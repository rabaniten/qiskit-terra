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

    def _multiplex(self, bottom_gate, bottom_qubit_index, list_of_angles):
        """
        Internal recursive method to create gates to perform rotations on the
        imaginary qubits: works by rotating LSB (and hence ALL imaginary
        qubits) by combo angle and then flipping sign (by flipping the bit,
        hence moving the complex amplitudes) of half the imaginary qubits
        (CNOT) followed by another combo angle on LSB, therefore executing
        conditional (on MSB) rotations, thereby disentangling LSB.
        """
        list_len = len(list_of_angles)
        target_qubit = self.nth_qubit_from_least_sig_qubit(bottom_qubit_index)

        # Case of no multiplexing = base case for recursion
        if list_len == 1:
            return bottom_gate(list_of_angles[0], target_qubit)

        local_num_qubits = int(math.log2(list_len)) + 1
        control_qubit = self.nth_qubit_from_least_sig_qubit(
            local_num_qubits - 1 + bottom_qubit_index)

        # calc angle weights, assuming recursion (that is the lower-level
        # requested angles have been correctly implemented by recursion
        angle_weight = scipy.kron([[0.5, 0.5], [0.5, -0.5]],
                                  np.identity(2 ** (local_num_qubits - 2)))

        # calc the combo angles
        list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()
        combine_composite_gates = CompositeGate(
            "multiplex" + local_num_qubits.__str__(), [], self.qargs)

        # recursive step on half the angles fulfilling the above assumption
        combine_composite_gates._attach(
            self._multiplex(bottom_gate, bottom_qubit_index,
                            list_of_angles[0:(list_len // 2)]))

        # combine_composite_gates.cx(control_qubit,target_qubit) -> does not
        # work as expected because checks circuit
        # so attach CNOT as follows, thereby flipping the LSB qubit
        combine_composite_gates._attach(CnotGate(control_qubit, target_qubit))

        # implement extra efficiency from the paper of cancelling adjacent
        # CNOTs (by leaving out last CNOT and reversing (NOT inverting) the
        # second lower-level multiplex)
        sub_gate = self._multiplex(
            bottom_gate, bottom_qubit_index, list_of_angles[(list_len // 2):])
        if isinstance(sub_gate, CompositeGate):
            combine_composite_gates._attach(sub_gate.reverse())
        else:
            combine_composite_gates._attach(sub_gate)

        # outer multiplex keeps final CNOT, because no adjacent CNOT to cancel
        # with
        if self.num_qubits == local_num_qubits + bottom_qubit_index:
            combine_composite_gates._attach(CnotGate(control_qubit,
                                                     target_qubit))

        return combine_composite_gates

    @staticmethod
    def chop_num(numb):
        """
        Set very small numbers (as defined by global variable _EPS) to zero.
        """
        return 0 if abs(numb) < _EPS else numb


# ###############################################################
# Add needed functionality to other classes (it feels
# weird following the Qiskit convention of adding functionality to other
# classes like this ;),
#  TODO: multiple inheritance might be better?)


def reverse(self):
    """
    Reverse (recursively) the sub-gates of this CompositeGate. Note this does
    not invert the gates!
    """
    new_data = []
    for gate in reversed(self.data):
        if isinstance(gate, CompositeGate):
            new_data.append(gate.reverse())
        else:
            new_data.append(gate)
    self.data = new_data

    # not just a high-level reverse:
    # self.data = [gate for gate in reversed(self.data)]

    return self


QuantumCircuit.reverse = reverse
CompositeGate.reverse = reverse


def optimize_gates(self):
    """Remove Zero rotations and Double CNOTS."""
    self.remove_zero_rotations()
    while self.remove_double_cnots_once():
        pass


QuantumCircuit.optimize_gates = optimize_gates
CompositeGate.optimize_gates = optimize_gates


def remove_zero_rotations(self):
    """
    Remove Zero Rotations by looking (recursively) at rotation gates at the
    leaf ends.
    """
    # Removed at least one zero rotation.
    zero_rotation_removed = False
    new_data = []
    for gate in self.data:
        if isinstance(gate, CompositeGate):
            zero_rotation_removed |= gate.remove_zero_rotations()
            if gate.data:
                new_data.append(gate)
        else:
            if ((not isinstance(gate, Gate)) or
                    (not (gate.name == "rz" or gate.name == "ry" or
                          gate.name == "rx") or
                     (InitializeGate.chop_num(gate.params[0]) != 0))):
                new_data.append(gate)
            else:
                zero_rotation_removed = True

    self.data = new_data

    return zero_rotation_removed


QuantumCircuit.remove_zero_rotations = remove_zero_rotations
CompositeGate.remove_zero_rotations = remove_zero_rotations


def number_atomic_gates(self):
    """Count the number of leaf gates. """
    num = 0
    for gate in self.data:
        if isinstance(gate, CompositeGate):
            num += gate.number_atomic_gates()
        else:
            if isinstance(gate, Gate):
                num += 1
    return num


QuantumCircuit.number_atomic_gates = number_atomic_gates
CompositeGate.number_atomic_gates = number_atomic_gates


def remove_double_cnots_once(self):
    """
    Remove Double CNOTS paying attention that gates may be neighbours across
    Composite Gate boundaries.
    """
    num_high_level_gates = len(self.data)

    if num_high_level_gates == 0:
        return False
    else:
        if num_high_level_gates == 1 and isinstance(self.data[0],
                                                    CompositeGate):
            return self.data[0].remove_double_cnots_once()

    # Removed at least one double cnot.
    double_cnot_removed = False

    # last gate might be composite
    if isinstance(self.data[num_high_level_gates - 1], CompositeGate):
        double_cnot_removed = \
            double_cnot_removed or\
            self.data[num_high_level_gates - 1].remove_double_cnots_once()

    # don't start with last gate, using reversed so that can del on the go
    for i in reversed(range(num_high_level_gates - 1)):
        if isinstance(self.data[i], CompositeGate):
            double_cnot_removed =\
                double_cnot_removed \
                or self.data[i].remove_double_cnots_once()
            left_gate_host = self.data[i].last_atomic_gate_host()
            left_gate_index = -1
            # TODO: consider adding if semantics needed:
            # to remove empty composite gates
            # if left_gate_host == None: del self.data[i]
        else:
            left_gate_host = self.data
            left_gate_index = i

        if ((left_gate_host is not None) and
                left_gate_host[left_gate_index].name == "cx"):
            if isinstance(self.data[i + 1], CompositeGate):
                right_gate_host = self.data[i + 1].first_atomic_gate_host()
                right_gate_index = 0
            else:
                right_gate_host = self.data
                right_gate_index = i + 1

            if (right_gate_host is not None) \
                    and right_gate_host[right_gate_index].name == "cx" \
                    and (left_gate_host[left_gate_index].qargs ==
                         right_gate_host[right_gate_index].qargs):
                del right_gate_host[right_gate_index]
                del left_gate_host[left_gate_index]
                double_cnot_removed = True

    return double_cnot_removed


QuantumCircuit.remove_double_cnots_once = remove_double_cnots_once
CompositeGate.remove_double_cnots_once = remove_double_cnots_once


def first_atomic_gate_host(self):
    """Return the host list of the leaf gate on the left edge."""
    if self.data:
        if isinstance(self.data[0], CompositeGate):
            return self.data[0].first_atomic_gate_host()
        return self.data

    return None


QuantumCircuit.first_atomic_gate_host = first_atomic_gate_host
CompositeGate.first_atomic_gate_host = first_atomic_gate_host


def last_atomic_gate_host(self):
    """Return the host list of the leaf gate on the right edge."""
    if self.data:
        if isinstance(self.data[-1], CompositeGate):
            return self.data[-1].last_atomic_gate_host()
        return self.data

    return None


QuantumCircuit.last_atomic_gate_host = last_atomic_gate_host
CompositeGate.last_atomic_gate_host = last_atomic_gate_host


def initialize(self, params, qubits):
    """Apply initialize to circuit."""
    # TODO: make initialize an Instruction, and insert reset
    # TODO: avoid explicit reset if compiler determines a |0> state
    return self._attach(InitializeGate(params, qubits, self))


def is_isometry(m, eps):
    err = np.linalg.norm(np.dot(np.transpose(np.conj(m)), m) - np.eye(m.shape[1], m.shape[1]))
    return math.isclose(err, 0, abs_tol=eps)


# def does_same_qubit_appears_twice(qubit_list):
#     qubit_numbers = []
#     for qubit in qubit_list:
#         qubit_numbers.append(qubit[1])
#     return not (len(qubit_numbers) == len(set(qubit_numbers)))

