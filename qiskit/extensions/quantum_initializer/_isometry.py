# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# The structure of the code is based on Emanuel Malvetti's semester thesis at ETH in 2018, which was supervised by Raban Iten and Prof. Renato Renner.

"""
Arbitrary isometry from m to n qubits.
"""

import math
import numpy as np
import scipy

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.ry import RYGate
from qiskit.extensions.standard.rz import RZGate
import itertools



_EPS = 1e-10  # global variable used to chop very small numbers to zero


class Isometry(CompositeGate):  # pylint: disable=abstract-method
    """Decomposition of arbitrary isometries from m to n qubits.
    In particular, this allows to decompose unitaries (m=n) and to do state preparation (m=0).

    The decomposition is based on https://arxiv.org/abs/1501.06911.

    Additionally implements some extra optimizations: remove zero rotations and
    double cnots.`

    It inherits from CompositeGate in the same way that the Fredkin (cswap)
    gate does. Therefore self.data is the list of gates (in order) that must
    be applied to implement this meta-gate.

    params = an isometry from m to n qubits, i.e., a (complex) np.ndarray of dimension 2^n*2^m with orthonormal columns
            (given in the computational basis specified by the order of the ancilla and the input qubits,
            where the ancillas are considered to be more significant than the input qubits)
    q_input = list of qubits where the input to the isometry is provided on (empty list for state preparation)
    q_ancilla = list of ancilla qubits (which are assumed to start in the zero state)
    circ = QuantumCircuit or CompositeGate containing this gate
    """


    def __init__(self, isometry, q_ancillas, q_input, circ=None):
        self.q_controls = q_controls
        self.q_ancillas = q_ancillas

        # Check if q_input and q_ancillas have type "list"
        if not type(q_input) == list:
            raise QiskitError("The input qubits must be provided as a list.")
        if not type(q_ancillas) == list:
            raise QiskitError("The ancilla qubits must be provided as a list.")
        # Check type of isometry
        if not type(isometrty) == np.ndarray:
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
        if len(q_input)+len(q_ancillas) < n:
            raise QiskitError("There are not enough ancilla qubits availabe to implement the isometry.")

        # Create new initialize composite gate.
        num_qubits = len(q_input) + len(q_ancillas)
        self.num_qubits = int(num_qubits)
        qubits = q_input+q_ancillas
        super().__init__("init", isometry, qubits, circ)
        # Check if a qubit is provided as an input AND an ancilla qubit
        self._check_dups(qubits, message="There is an input qubit that is also listed as an ancilla qubit.")
        # call to generate the circuit that takes the desired vector to zero
        self._gates_to_uncompute()
        # remove zero rotations and double cnots
        self.optimize_gates()
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
        # Copy the isometry (this is computationally expensive for large isometries but garantees to keep a copy of the input isometry)
        remaining_isometry = np.copy(self.params)
        phases=[]
        n = np.log2(self.params.shape[0])
        m = np.log2(self.params.shape[1])

        for column_index in range(2**m):
            # decompose the column with index column_index and attache the gate to the CompositeGate object. Return the
            # isometry which is left to decompose, where the columns up to index column_index correspond to the first
            # few columns of the identity matrix up to phases, and hence we only have to save a list containing them.
            (phases,remaining_isometry) = _decompose_column(phases,remaining_isometry,column_index)

    def _decompose_column(self,phases,remaining_isometry,column_index):
        """
        Decomposes the column with index column_index.
        """



        # work out which rotations must be done to disentangle the LSB
        # qubit (we peel away one qubit at a time)
        (remaining_isometry, thetas, phis) = InitializeGate._rotations_to_disentangle(remaining_isometry)

        # perform the required rotations to decouple the LSB qubit (so that
        # it can be "factored" out, leaving a
        # shorter amplitude vector to peel away)
        self._attach(self._multiplex(RZGate, i, phis))
        self._attach(self._multiplex(RYGate, i, thetas))





    @staticmethod
    def _rotations_to_disentangle(local_param):
        """
        Static internal method to work out Ry and Rz rotation angles used
        to disentangle the LSB qubit.
        These rotations make up the block diagonal matrix U (i.e. multiplexor)
        that disentangles the LSB.

        [[Ry(theta_1).Rz(phi_1)  0   .   .   0],
         [0         Ry(theta_2).Rz(phi_2) .  0],
                                    .
                                        .
          0         0           Ry(theta_2^n).Rz(phi_2^n)]]
        """
        remaining_vector = []
        thetas = []
        phis = []

        param_len = len(local_param)

        for i in range(param_len // 2):
            # Ry and Rz rotations to move bloch vector from 0 to "imaginary"
            # qubit
            # (imagine a qubit state signified by the amplitudes at index 2*i
            # and 2*(i+1), corresponding to the select qubits of the
            # multiplexor being in state |i>)
            (remains,
             add_theta,
             add_phi) = InitializeGate._bloch_angles(
                 local_param[2*i: 2*(i + 1)])

            remaining_vector.append(remains)

            # rotations for all imaginary qubits of the full vector
            # to move from where it is to zero, hence the negative sign
            thetas.append(-add_theta)
            phis.append(-add_phi)

        return remaining_vector, thetas, phis

    @staticmethod
    def _bloch_angles(pair_of_complex):
        """
        Static internal method to work out rotation to create the passed in
        qubit from the zero vector.
        """
        [a_complex, b_complex] = pair_of_complex
        # Force a and b to be complex, as otherwise numpy.angle might fail.
        a_complex = complex(a_complex)
        b_complex = complex(b_complex)
        mag_a = np.absolute(a_complex)
        final_r = float(np.sqrt(mag_a ** 2 + np.absolute(b_complex) ** 2))
        if final_r < _EPS:
            theta = 0
            phi = 0
            final_r = 0
            final_t = 0
        else:
            theta = float(2 * np.arccos(mag_a / final_r))
            a_arg = np.angle(a_complex)
            b_arg = np.angle(b_complex)
            final_t = a_arg + b_arg
            phi = b_arg - a_arg

        return final_r * np.exp(1.J * final_t/2), theta, phi


def initialize(self, params, qubits):
    """Apply initialize to circuit."""
    # TODO: make initialize an Instruction, and insert reset
    # TODO: avoid explicit reset if compiler determines a |0> state
    return self._attach(InitializeGate(params, qubits, self))


    # Input: matrix m with 2^n rows (and arbitrary many columns). Think of the columns as states on n qubits. The method
    # applies a uniformly controlled gate (UCG) to all the columns, where the UCG is specified by the inputs k and
    # single_qubit_gates:

    #  k = number of controls. We assume that the controls are on the k most significant qubits (and the target is on
    #       the (k+1)th significant qubit
    #  single_qubit_gates = [u_0,...,u_{2^k-1}], where the u_i are 2*2 unitaries (provided as numpy arrays)
    #

    # The order of the single-qubit unitaries is such that the first unitary u_0 is applied to the (k+1)th significant qubit
    # if the control qubits are in the state ket(0...00), the gate u_1 is applied if the control qubits are in the state
    # ket(0...01), and so on.


def a(k, s):
    return k // 2**s


def b(k, s):
    return k - (a(k, s) * 2**s)

def _apply_uniformly_controlled_gate(m, k, single_qubit_gates):
    # ToDo: Improve efficiency by parallelizing the gate application
    num_qubits = int(np.log2(m.shape[0]))
    num_col= m.shape[1]
    spacing = 2**(num_qubits - k -1)
    for j in range(2**(num_qubits-1)):
        i = (j //spacing) * spacing + j
        gate_index = i//(2**(num_qubits - k))
        for l in range(num_col):
            m[np.array([i,i+spacing]),np.array([l,l])] = \
                np.ndarray.flatten(single_qubit_gates[gate_index].dot(np.array([[m[i,l]],[m[i+spacing,l]]]))).tolist()

def _apply_diagonal_gate(m, controls, target, gate):
    # ToDo: Improve efficiency
    num_qubits = int(np.log2(m.shape[0]))
    num_cols = m.shape[1]
    controls.sort()
    free_qubits = num_qubits - len(controls) - 1
    basis_states_free = list(itertools.product([0, 1], repeat=free_qubits))
    for state_free in basis_states_free:
        (e1, e2) = _construct_basis_states(state_free, controls,target)
        for l in range(num_cols):
            m[np.array([e1, e2]), np.array([l, l])] = \
                np.ndarray.flatten(gate.dot(np.array([[m[e1, l]], [m[e2, l]]]))).tolist()
    return m

def _apply_multi_controlled_gate(m, controls, target, gate):
    # ToDo: Improve efficiency
    num_qubits = int(np.log2(m.shape[0]))
    num_cols = m.shape[1]
    controls.sort()
    free_qubits = num_qubits - len(controls) - 1
    basis_states_free = list(itertools.product([0, 1], repeat=free_qubits))
    for state_free in basis_states_free:
        (e1, e2) = _construct_basis_states(state_free, controls,target)
        for l in range(num_cols):
            m[np.array([e1, e2]), np.array([l, l])] = \
                np.ndarray.flatten(gate.dot(np.array([[m[e1, l]], [m[e2, l]]]))).tolist()
    return m

def _construct_basis_states(state_free, controls, target):
    e1 =[]
    e2 =[]
    j = 0
    for i in range(len(state_free)+len(controls)+1):
        if i in controls:
            e1.append(1)
            e2.append(1)
        elif i == target:
            e1.append(0)
            e2.append(1)
        else:
            e1.append(state_free[j])
            e2.append(state_free[j])
            j += 1
    # Convert list of binary digits to integer
    out1 = int("".join(str(x) for x in e1), 2)
    out2 = int("".join(str(x) for x in e2), 2)
    return out1, out2

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

