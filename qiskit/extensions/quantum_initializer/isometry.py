# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Arbitrary isometry from m to n qubits.
"""

import math
import numpy as np
import scipy

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import CompositeGate
from qiskit.extensions.quantum_initializer.ucg import UCG
from qiskit.extensions.quantum_initializer.diag import DiagGate

import itertools


_EPS = 1e-10  # global variable used to chop very small numbers to zero


class Isometry(CompositeGate):  # pylint: disable=abstract-method
    """Decomposition of arbitrary isometries from m to n qubits.
    In particular, this allows to decompose unitaries (m=n) and to do state preparation (m=0).

    The decomposition is based on https://arxiv.org/abs/1501.06911.

    It inherits from CompositeGate in the same way that the Fredkin (cswap)
    gate does. Therefore self.data is the list of gates (in order) that must
    be applied to implement this meta-gate.

    params = an isometry from m to n qubits, i.e., a (complex) np.ndarray of dimension 2^n*2^m with orthonormal columns
            (given in the computational basis specified by the order of the ancillas and the input qubits,
            where the ancillas are considered to be more significant than the input qubits.)
    q_input = list of qubits where the input to the isometry is provided on (empty list for state preparation)
              The qubits are listed with increasing significance.
    q_ancilla = list of ancilla qubits (which are assumed to start in the zero state).
                The qubits are listed with increasing significance.
    circ = QuantumCircuit or CompositeGate containing this gate
    """

    """
    Notation: In the following decomposition we label the qubit by 
    0 - most significant one
    ...
    n - least significant one
    finally, we convert the labels back to the qubit numbering used in Qiskit (using: _get_qubits_by_label)
    """

    def __init__(self, isometry, q_input, q_ancillas_for_output, q_ancillas_zero=[], q_ancillas_dirty=[], circ=None):
        self.q_input = q_input
        self.q_ancillas_for_output = q_ancillas_for_output
        self.q_ancillas_zero = q_ancillas_zero
        self.q_ancillas_dirty = q_ancillas_dirty

        # Check if q_input and q_ancillas have type "list"
        if not type(q_input) == list:
            raise QiskitError("The input qubits must be provided as a list.")
        if not type(q_ancillas_for_output) == list:
            raise QiskitError("The ancilla qubits must be provided as a list.")
        # Check type of isometry
        if not type(isometry) == np.ndarray:
            raise QiskitError("The isometry is not of type numpy.ndarray.")

        # Check if the isometry has the right dimension and if the columns are orthonormal
        n = np.log2(isometry.shape[0])
        m = np.log2(isometry.shape[1])
        if not n == int(n):
            raise QiskitError("The number of rows of the isometry is not a power of 2.")
        if not m == int(m):
            raise QiskitError("The number of columns of the isometry is not a power of 2.")
        if m > n:
            raise QiskitError("The input matrix has more columns than rows and hence it can't be an isometry.")
        if not is_isometry(isometry, _EPS):
            raise QiskitError("The input matrix has non orthonormal columns and hence it is not an isometry.")

        # Check if the number of input qubits corresponds to the provided isometry
        if len(q_input) != m:
            raise QiskitError("The number of input qubits is not equal to log2(k), where k is the number of columns of the provided isometry.")
        # Check if there are enough ancilla qubits
        if len(q_input)+len(q_ancillas_for_output) < n:
            raise QiskitError("There are not enough ancilla qubits availabe to implement the isometry.")

        # Create new initialize composite gate.
        num_qubits = len(q_input) + len(q_ancillas_for_output) + len(q_ancillas_zero) + len(q_ancillas_dirty)
        self.num_qubits = int(num_qubits)
        qubits = q_input + q_ancillas_for_output + q_ancillas_zero + q_ancillas_dirty
        super().__init__("init", [isometry], qubits, circ)
        # Check if a qubit is provided as an input AND an ancilla qubit
        self._check_dups(qubits, message="There is an input qubit that is also listed as an ancilla qubit.")
        # call to generate the circuit that takes the desired vector to zero
        self._gates_to_uncompute()
        # invert the circuit to create the desired vector from zero (assuming
        # the qubits are in the zero state)
        self.inverse()
        # do not set the inverse flag, as this is the actual initialize gate
        # we just used inverse() as a method to obtain it
        self.inverse_flag = False

    def _gates_to_uncompute(self):
        """
        Call to populate the self.data list with gates that takes the
        desired isometry to the first 2^m columns of the 2^n*2^n identity matrix (see https://arxiv.org/abs/1501.06911)
        """

        # Copy the isometry (this is computationally expensive for large isometries but guarantees to keep a copy
        # of the input isometry)
        remaining_isometry = self.params[0].astype(complex)  # note that "astype" does copy the isometry
        diag = []
        m = int(np.log2((self.params[0]).shape[1]))
        for column_index in range(2**m):
            # decompose the column with index column_index and attache the gate to the CompositeGate object. Return the
            # isometry which is left to decompose, where the columns up to index column_index correspond to the first
            # few columns of the identity matrix up to diag, and hence we only have to save a list containing them.
            (diag, remaining_isometry) = self._decompose_column(diag, remaining_isometry, column_index)
            # extract phase of the state that was sent to the basis state ket(column_index)
            diag.append(remaining_isometry[column_index, 0])
            remaining_isometry = remaining_isometry[:, 1:]
        # ToDo: Implement diagonal gate for one diagonal entry (do nothing)
        if len(diag) > 1:
            self._attach(DiagGate(np.conj(diag).tolist(), self.q_input))

    def _decompose_column(self, diag, remaining_isometry, column_index):
        """
        Decomposes the column with index column_index.
        """
        n = int(np.log2(self.params[0].shape[0]))
        for s in range(n):
            (diag, remaining_isometry) = self._disentangle(diag, remaining_isometry,column_index, s)
        return diag, remaining_isometry

    def _disentangle(self, diag, remaining_isometry, column_index, s):
        """
        Disentangle the sth significant qubit (starting with s=0) into the zero or the one state
        (dependent on column_index)
        """
        k = column_index
        # k_prime is the index of the column with index column_index in the remaining isometry (note that we
        # remove columns of the isometry during the procedure)
        k_prime = 0
        v = remaining_isometry
        n = int(np.log2(self.params[0].shape[0]))
        """MCG"""
        index1 = 2*a(k,s+1)*2**s+b(k,s+1)
        index2 = (2*a(k,s+1)+1)*2**s+b(k,s+1)
        target_label = n - s - 1
        # Check if a MCG is required
        if k_s(k,s) == 0 and b(k,s+1) != 0 and np.abs(v[index2, k_prime]) > _EPS:
            # Find the MCG, decompose it and apply it to the remaining isometry
            gate = reverse_qubit_state([v[index1,k_prime],v[index2,k_prime]],0)
            control_labels = [i for i, x in enumerate(get_binary_rep_as_list(k, n)) if x == 1 and i != target_label]
            diagonal_mcg = self._attach_mcg_up_to_diagonal(gate, control_labels, target_label)
            # apply the MCG to the remaining isometry
            _apply_multi_controlled_gate(v, control_labels, target_label, gate)
            # correct for the implementation "up to diagonal"
            diag_mcg_inverse = np.conj(diagonal_mcg).tolist()
            _apply_diagonal_gate(v, control_labels + [target_label], diag_mcg_inverse)
            # update the diag according to the applied diagonal gate
            _apply_diagonal_gate_to_diag(diag, control_labels + [target_label], diag_mcg_inverse, n)
        """UCG"""
        # Find the UCG, decompose it and apply it to the remaining isometry
        single_qubit_gates = self._find_squs_for_disentangling(v, k, s)
        if not is_identity(single_qubit_gates):
            control_labels = list(range(target_label))
            diagonal_ucg = self._attach_ucg_up_to_diagonal(single_qubit_gates, control_labels, target_label)
            # merge the diagonal into the UCG for efficient appliaction of both together
            diagonal_ucg_inverse = np.conj(diagonal_ucg).tolist()
            single_qubit_gates = _merge_UCG_and_diag(single_qubit_gates, diagonal_ucg_inverse)
            # apply the UCG (with the merged diagonal gate) to the remaining isometry
            _apply_uniformly_controlled_gate(v, len(control_labels), single_qubit_gates)
            # update the diag according to the applied diagonal gate
            _apply_diagonal_gate_to_diag(diag, control_labels + [target_label], diagonal_ucg_inverse, n)
            # # correct for the implementation "up to diagonal"
            # diag_inv = np.conj(diag).tolist()
            # _apply_diagonal_gate(v, control_labels + [target_label], diag_inv)
        return diag, remaining_isometry

    def _find_squs_for_disentangling(self, v, k, s):
        k_prime = 0
        n = int(np.log2(self.params[0].shape[0]))
        if b(k,s+1) == 0:
            i_start = a(k,s+1)
        else:
            i_start = a(k,s+1) +1
        id_list = [np.eye(2,2) for i in range(i_start)]
        squs = [reverse_qubit_state([v[2*l*2**s+b(k,s),k_prime],v[(2*l+1)*2**s+b(k,s),k_prime]], k_s(k,s)) for l in range(i_start,2**(n-s-1))]
        return id_list + squs

    def _attach_ucg_up_to_diagonal(self, single_qubit_gates, control_labels, target_label):
        n = int(np.log2(self.params[0].shape[0]))
        qubits = self.q_input + self.q_ancillas_for_output
        # Note that we have to reverse the control labels, since controls are provided by increasing qubit number to
        # a UCG by convention
        ucg = UCG(single_qubit_gates, _reverse_qubit_oder(_get_qubits_by_label(control_labels, qubits, n)),
                  _get_qubits_by_label([target_label], qubits, n)[0], up_to_diagonal=True)
        self._attach(ucg)
        return ucg.diag

    def _attach_mcg_up_to_diagonal(self, gate, control_labels, target_label):
        n = int(np.log2(self.params[0].shape[0]))
        qubits = self.q_input + self.q_ancillas_for_output
        # ToDo: Keep this threshold updated such that the lowest gate count is achieves:
        # ToDo: we implement the MCG with a UCG up to diagonal for if the number of controls is smaller than the
        # ToDo: threshold. Otherwise, the best method for MCGs up to a diagonal should be used
        threshold = float("inf")
        if n < threshold:
            # Implement the MCG as a UCG (up to diagonal)
            gate_list = [np.eye(2,2) for i in range(2**len(control_labels))]
            gate_list[-1] = gate
            ucg = UCG(gate_list, _reverse_qubit_oder(_get_qubits_by_label(control_labels, qubits, n)),
                      _get_qubits_by_label([target_label],qubits, n)[0], up_to_diagonal=True)
            self._attach(ucg)
            return ucg.diag
        else:
            # ToDo: Use the best decomposition for MCGs up to diagonal gates here (with ancillas
            #  self.q_ancillas_zero that start in the zero state and  dirty ancillas self.q_ancillas_dirty)
            return range(len(control_labels)+1)


def _merge_UCG_and_diag(single_qubit_gates, diag):
    for i in range(len(single_qubit_gates)):
        single_qubit_gates[i] = np.array([[diag[2*i],0.],[0., diag[2*i+1]]]).dot(single_qubit_gates[i])
    return single_qubit_gates


def _get_qubits_by_label(labels, qubits, num_qubits):
    # note that we label them here with decreasing significance. So we have to transform the labels to be compatible
    # with the standard convention of Qiskit (and in particular, of the state vector simulator in Qiskit aer).
    return [qubits[num_qubits-label-1] for label in labels]


def _reverse_qubit_oder(qubits):
    return [q for q in reversed(qubits)]


def a(k, s):
    return k // 2**s


def b(k, s):
    return k - (a(k, s) * 2**s)


def k_s(k,s):
    if k == 0:
        return 0
    else:
        num_digits = s+1
        return get_binary_rep_as_list(k, num_digits)[0]

# Input: matrix m with 2^n rows (and arbitrary many columns). Think of the columns as states on n qubits. The method
# applies a uniformly controlled gate (UCG) to all the columns, where the UCG is specified by the inputs k and
# single_qubit_gates:

#  k = number of controls. We assume that the controls are on the k most significant qubits (and the target is on
#       the (k+1)th significant qubit
#  single_qubit_gates = [u_0,...,u_{2^k-1}], where the u_i are 2*2 unitaries (provided as numpy arrays)

# The order of the single-qubit unitaries is such that the first unitary u_0 is applied to the (k+1)th significant qubit
# if the control qubits are in the state ket(0...00), the gate u_1 is applied if the control qubits are in the state
# ket(0...01), and so on.

# The input matrix m and the single qubit gates have to be of dtype=complex
# The qubit labels are such that label 0 corresponds to the most significant qubit, label 1 to the second most
# significant qubit, and so on ...


def _apply_uniformly_controlled_gate(m, k, single_qubit_gates):
    # ToDo: Improve efficiency by parallelizing the gate application
    num_qubits = int(np.log2(m.shape[0]))
    num_col = m.shape[1]
    spacing = 2**(num_qubits - k -1)
    for j in range(2**(num_qubits-1)):
        i = (j //spacing) * spacing + j
        gate_index = i//(2**(num_qubits - k))
        for l in range(num_col):
            m[np.array([i,i+spacing]),np.array([l,l])] = \
                np.ndarray.flatten(single_qubit_gates[gate_index].dot(np.array([[m[i,l]],[m[i+spacing,l]]]))).tolist()
    return m

# The input matrix m has to be of dtype=complex
# The qubit labels are such that label 0 corresponds to the most significant qubit, label 1 to the second most
# significant qubit, and so on ...


def _apply_diagonal_gate(m, action_qubit_labels, diag):
    # ToDo: Improve efficiency
    num_qubits = int(np.log2(m.shape[0]))
    num_cols = m.shape[1]
    basis_states = list(itertools.product([0, 1], repeat=num_qubits))
    for state in basis_states:
        state_on_action_qubits = [state[i] for i in action_qubit_labels]
        diag_index = bin_to_int(state_on_action_qubits)
        i = bin_to_int(state)
        for j in range(num_cols):
            m[i,j] = diag[diag_index] * m[i, j]
    return m


def _apply_diagonal_gate_to_diag(m_diagonal, action_qubit_labels, diag, num_qubits):
    if len(m_diagonal) == 0:
        return m_diagonal
    basis_states = list(itertools.product([0, 1], repeat=num_qubits))
    for state in basis_states[:len(m_diagonal)]:
        state_on_action_qubits = [state[i] for i in action_qubit_labels]
        diag_index = bin_to_int(state_on_action_qubits)
        i = bin_to_int(state)
        m_diagonal[i] *= diag[diag_index]
    return m_diagonal

# The input matrix m and the gate have to be of dtype=complex
# The qubit labels are such that label 0 corresponds to the most significant qubit, label 1 to the second most
# significant qubit, and so on ...

# The diagonal in the output is with respect to the computational basis
# control_label[0] - most significant
# control_label[1] - second significant
# ...
# target_label - least significant


def _apply_multi_controlled_gate(m, control_labels, target_label, gate):
    # ToDo: Improve efficiency
    num_qubits = int(np.log2(m.shape[0]))
    num_cols = m.shape[1]
    control_labels.sort()
    free_qubits = num_qubits - len(control_labels) - 1
    basis_states_free = list(itertools.product([0, 1], repeat=free_qubits))
    for state_free in basis_states_free:
        (e1, e2) = _construct_basis_states(state_free, control_labels, target_label)
        for l in range(num_cols):
            m[np.array([e1, e2]), np.array([l, l])] = \
                np.ndarray.flatten(gate.dot(np.array([[m[e1, l]], [m[e2, l]]]))).tolist()
    return m


def _construct_basis_states(state_free, control_labels, target_label):
    e1 = []
    e2 = []
    j = 0
    for i in range(len(state_free) + len(control_labels) + 1):
        if i in control_labels:
            e1.append(1)
            e2.append(1)
        elif i == target_label:
            e1.append(0)
            e2.append(1)
        else:
            e1.append(state_free[j])
            e2.append(state_free[j])
            j += 1
    out1 = bin_to_int(e1)
    out2 = bin_to_int(e2)
    return out1, out2


# Convert list of binary digits to integer
def bin_to_int(binary_digits_as_list):
    return int("".join(str(x) for x in binary_digits_as_list), 2)


# Find special unitary matrix that maps [c0,c1] to [r,0] or [0,r] if basis_state=0 or basis_state=1 respectively
def reverse_qubit_state(state, basis_state):
    state= np.array(state)
    r = np.linalg.norm(state)
    if r < _EPS:
        return np.eye(2,2)
    if basis_state == 0:
        m = np.array([[np.conj(state[0]), np.conj(state[1])], [-state[1], state[0]]]) / r
    else:
        m = np.array([[-state[1], state[0]],[np.conj(state[0]), np.conj(state[1])]]) / r
    return m


def ct(m):
    return np.transpose(np.conjugate(m))


def is_isometry(m, eps):
    err = np.linalg.norm(np.dot(np.transpose(np.conj(m)), m) - np.eye(m.shape[1], m.shape[1]))
    return math.isclose(err, 0, abs_tol=eps)

# def does_same_qubit_appears_twice(qubit_list):
#     qubit_numbers = []
#     for qubit in qubit_list:
#         qubit_numbers.append(qubit[1])
#     return not (len(qubit_numbers) == len(set(qubit_numbers)))


def get_binary_rep_as_list(n, num_digits):
    binary_string = np.binary_repr(n).zfill(num_digits)
    binary = []
    for line in binary_string:
        for c in line:
            binary.append(int(c))
    return binary[-num_digits:]


def is_identity(single_qubit_gates):
    for gate in single_qubit_gates:
        if not np.allclose(gate,np.eye(2,2)):
            return False
    return True


def iso(self, isometry, q_input, q_ancillas_for_output, q_ancillas_zero=[], q_ancillas_dirty=[]):
    return self._attach(Isometry(isometry, q_input, q_ancillas_for_output, q_ancillas_zero, q_ancillas_dirty, self))


QuantumCircuit.iso = iso
CompositeGate.iso = iso