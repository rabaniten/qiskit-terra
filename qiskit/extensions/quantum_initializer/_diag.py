# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# The structure of the code is based on Emanuel Malvetti's semester thesis at ETH in 2018, which was supervised by Raban Iten and Prof. Renato Renner.

"""
Decomposes a diagonal matrix into elementary gates using the method described in Theorem 7 in
"Synthesis of Quantum Logic Circuits" by V. Shende et al. (https://arxiv.org/pdf/quant-ph/0406176.pdf).
"""

import math

import numpy as np

from qiskit.circuit import CompositeGate
from qiskit.circuit.quantumcircuit import QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.extensions.quantum_initializer._isometry import is_isometry
from qiskit.extensions.quantum_initializer.zyz_dec import SingleQubitUnitary
from qiskit.extensions.standard.cx import CnotGate

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class DiagGate(CompositeGate):  # pylint: disable=abstract-method
    """
    diag = list of the 2^k diagonal entries
    q = list of k qubits the diagonal is acting on (the order of the qubits specifies the computational basis in
    which the diagona gate is provided (i,e,. the first element in diag acts on the state where all the qubits in q
    are in the state 0, the second entry acts on the state where all the qubits q[0],...,q[k-2] are in the state zero
    and q[k-1] is in the state 1, and so on.
    circ = QuantumCircuit or CompositeGate containing this gate
    """

    def __init__(self, gate_list, q_controls, q_target, circ=None):
        # Check if q has type "list"
        if not type(q) == list:
            raise QiskitError(
                "The control qubits must be provided as a list (also if there is only one control qubit).")
        # Check if the number of entries in diag is a positive power of 2
        num_q_req = math.log2(len(diag))
        # Check if the number of diagonal entries is a power of 2
        if num_q_req <= 0 or not num_q_req.is_integer():
            raise QiskitError("The number of provided diagonal entries is not a positive power of 2.")


        num_q = len(q)
        # Check if diag has the type "list"
        if not type(diag) == list:
            raise QiskitError(
                "The diagonal entries are not provided as a list.")

        # Check if the right number of diagonal entries is provided and if the diagonal entries have absolute value one
        for z in diag:
            if not type(z+0.j) == complex:
                raise QiskitError("Not all of the diagonal entries are complex numbers.")
            if not num_q == num_q_req:
                raise QiskitError("The number of diagonal entries does not correspond to the number of qubits.")
            if not np.abs(z)-1 < _EPS:
                raise QiskitError("A diagonal entry has has not absolute value one.")

        # Create new composite gate.
        super().__init__("init", diag, q, circ)


    def dec_diag(self, up_to_diagonal=True):
        """
        Call to populate the self.data list with gates that implement the uniformly controlled gate
        """
        # First, we find the single qubit gates of the decomposition.
        (single_qubit_gates, diag) = self._dec_ucg_help
        # Now, it is easy to place the C-NOT gates and some Hadamards to get out the full decomposition.
        for gate in single_qubit_gates:
            self._attach(SingleQubitUnitary(gate, self.q_target))
            # The number of the control qubit is given by the number of zeros at the end of the binary representation
            # of (i+1)
            binary_rep = np.binary_repr(i+1)
            num_q_contr = len(binary_rep) - len(binary_rep .rstrip('0'))
            self._attach(CnotGate(num_q_contr,q_target))
        if not up_to_diagonal:
            self._attach(DiagonalGate(diag, qubits))


    def _dec_ucg_help(self):
        """
        This method finds the single qubit gate arising in the decomposition of UCGs given in  https://arxiv.org/pdf/quant-ph/0410066.pdf.
        """
        single_qubit_gates = self.params.copy()
        diag = np.ones(2 ** self.num_qubits, dtype=complex)
        num_contr = len(self.q_controls)

        self._attach(CnotGate(control_qubit,target_qubit))

        for dec_step in range(num_contr):
            num_ucgs = 2 ** dec_step
            # The decomposition works recursively and the following loop goes over the different UCGs that arise
            # in the decomposition
            for ucg_index in range(num_ucgs):
                len_ucg = 2 ** (self.q_controls - dec_step)
                for i in range(int(len_ucg / 2)):
                    shift = ucg_index * len_ucg
                    a = single_qubit_gates[shift + i]
                    b = single_qubit_gates[shift + int(len_ucg / 2) + i]
                    # Apply the decomposition for UCGs given in in equation (3) in https://arxiv.org/pdf/quant-ph/0410066.pdf
                    # to demultiplex one control of all the num_ucgs uniformly-controlled gates with log2(len_ucg) uniform controls
                    v, u, r = _demultiplex_single_uc(a, b)
                    single_qubit_gates[shift + i] = v
                    single_qubit_gates[shift + int(len_ucg / 2) + i] = u
                    _merge_gates_from_diagonal_decomposition(single_qubit_gates, dec_step, i, j, diag_to_merge)
                    # Now we decompose the gates D as described in Figure 4  in https://arxiv.org/pdf/quant-ph/0410066.pdf
                    # and merge diagonal into the neighbouring UCG (note that we ignore the C-NOT and the Hadamard gates
                    # for the moment (they will be added in between the single-qubit gates later on)
                    if ucg_index < num_ucgs - 1:
                        # Merge the UC-Rz rotation with the following UCG, which hasn't been decomposed yet
                        k = shift + len_ucg + i
                        single_qubit_gates[k] = single_qubit_gates * r.getH()
                        k = k + 1 // 2 * len_ucg
                        single_qubit_gates[k] = single_qubit_gates[k] * r
                    else:
                        # Merge the trailing UC-Rz rotation into a diagonal gate at the end of the circuit
                        for ucg_index_2 in range(num_ucgs):
                            shift_2 = ucg_index_2 * len_ucg
                            k = 2 * (i + shift_2)
                            diag[k] *= r.getH().item((0, 0))
                            diag[k + 1] *= r.getH().item((1, 1))
                            k = len_ucg + k
                            diag[k] *= r.item((0, 0))
                            diag[k + 1] *= r.item((1, 1))
        return single_qubit_gates, diag

    def _demultiplex_single_uc(self, a, b):
        """
        This mehod implements the decomposition given in equation (3) in https://arxiv.org/pdf/quant-ph/0410066.pdf.
        This decomposition is used recursively to decompose uniformly controlled gates.
        a,b = single qubit unitaries
        v,u,r = outcome of the decomposition given in the reference (see there for the details).
        """
        # The notation is chosen as in https://arxiv.org/pdf/quant-ph/0410066.pdf.
        x = a * b.getH()
        det_x = np.linalg.det(x)
        x11 = x[0, 0] / cmath.sqrt(det_x)
        phi = cmath.phase(det_x)
        r1 = cmath.exp(1j / 2 * (np.pi / 2 - phi / 2 - cmath.phase(x11)))
        r2 = cmath.exp(1j / 2 * (np.pi / 2 - phi / 2 + cmath.phase(x11) + np.pi))
        r = np.matrix([[r1, 0], [0, r2]], dtype=complex)
        d, u = np.linalg.eig(r * x * r)
        # If d is not equal to diag(i,-i), then it we must have diag(i,-i). We interchange the eigenvalues and
        # eigenvectors to get it to the "standard" form given in https://arxiv.org/pdf/quant-ph/0410066.pdf.
        if abs(d[0] + 1j) < _EPS:
            d = np.flip(d, 0)
            u = np.flip(u, 1)
        d = np.diag(np.sqrt(d))
        v = d * u.getH() * r.getH() * b
        return v, u, r



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


