# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# The structure of the code is based on Emanuel Malvetti's semester thesis at ETH in 2018, which was supervised by Raban Iten and Prof. Renato Renner.

"""
(Abstract) base class for uniformly controlled (also called multiplexed) single-qubit rotations R_t.
This class provides a basis for the decomposition of uniformly controlled R_x and R_z gates (i.e., for t=x,z).
These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state ket(i) (in the computational bases),
a single-qubit rotation R_t(a_i) is applied to the target qubit.
"""

import math

import numpy as np

from qiskit.circuit import CompositeGate
from qiskit.circuit.quantumcircuit import QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.rz import RZGate
from qiskit.extensions.standard.ry import RYGate

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class UCRot(CompositeGate):  # pylint: disable=abstract-method
    """Uniformly controlled rotations (also called multiplexed rotations). The decomposition is based on
    "Synthesis of Quantum Logic Circuits" by V. Shende et al. (https://arxiv.org/pdf/quant-ph/0406176.pdf)
    angle_list = list (real) rotation angles [a_0,...,a_{2^k-1}]
    q_controls = list of control k qubits (or empty list if no controls). The qubits are ordered according to their
                    significance in in increasing order.
                    For example if q_controls=[q[2],q[1]] (with q = QuantumRegister(2)), the rotation R_t(a_0)is
                    performed if q[2] and q[1] are in the state zero,
                    R_t(a_1) is performed if q[2] is in the state one and q[1] is in the state zero, and so on.
    q_target = target qubit, where we act on with the single-qubit gates.
    circ = QuantumCircuit or CompositeGate containing this gate
    """

    def __init__(self, angle_list, q_controls, q_target, rot_axes, circ=None):
        self.q_controls = q_controls
        self.q_target = q_target
        self.rot_axes = rot_axes
        """Check types"""
        # Check if q_controls has type "list"
        if not type(q_controls) == list:
            raise QiskitError(
                "The control qubits must be provided as a list (also if there is only one control qubit).")
        # Check if the entries in q_controls are qubits
        for qu in q_controls:
            if not (type(qu) == tuple and type(qu[0]) == QuantumRegister):
                raise QiskitError("Wrong type: there is a control qubit which is not part of a QuantumRegister.")
        # Check if angle_list has type "list"
        if not type(angle_list) == list:
            raise QiskitError(
                "The angles are not provided in a list.")
        # Check if the angles in angle_list are real numbers
        for a in angle_list:
            try:
                float(a)
            except:
                raise QiskitError("An angle cannot be converted to type float.")
        # Check if there is one target qubit provided
        if not (type(q_target) == tuple and type(q_target[0]) == QuantumRegister):
            raise QiskitError("The target qubit is not a single qubit from a QuantumRegister.")

        """Check input form"""
        num_contr = math.log2(len(angle_list))
        if num_contr < 0 or not num_contr.is_integer():
            raise QiskitError("The number of controlled rotation gates is not a non-negative power of 2.")
        # Check if number of control qubits does correspond to the number of rotations
        if num_contr != len(q_controls):
            raise QiskitError("Number of controlled rotations does not correspond to the number of control-qubits.")

        # Create new composite gate.
        num_qubits = len(q_controls) + len(q_target)
        self.num_qubits = int(num_qubits)
        qubits = q_controls + [q_target]
        super().__init__("init", angle_list, qubits, circ)
        # Check if the target qubit is also a control qubit
        self._check_dups(qubits, message="The target qubit cannot also be listed as a control qubit.")

    def _dec_uc_rot_gate(self):
        """
        Call to populate the self.data list with gates that implement the uniformly controlled gate
        """
        if len(self.q_controls) == 0:
            self._attach(RZGate(self.params[0], self.q_target))
        else:
            # First, we find the rotation angles of the single-qubit rotations acting on the target qubit
            angles = self.params.copy()
            self._dec_uc_rotations(angles, 0, len(angles), False)
            # Now, it is easy to place the C-NOT gates to get out the full decomposition.
            for i in range(len(angles)):
                if self.rot_axes == "Z":
                    if (angles[i] % 4 * np.pi) > _EPS:
                        self._attach(RZGate(angles[i], self.q_target))
                if self.rot_axes == "Y":
                    if (angles[i] % 4 * np.pi) > _EPS:
                        self._attach(RYGate(angles[i], self.q_target))
                # The number of the control qubit (labeling the control qubits from the bottom to the top,
                # starting with zero) is given by the number of zeros at the end of the binary representation of (i+1)
                if not i == len(angles) - 1:
                    binary_rep = np.binary_repr(i + 1)
                    num_trailing_zeros = len(binary_rep) - len(binary_rep.rstrip('0'))
                    q_contr_index = num_trailing_zeros
                else:
                    # Handle special case:
                    q_contr_index = len(self.q_controls) - 1
                self._attach(CnotGate(self.q_controls[q_contr_index], self.q_target))

    # Calculates rotation angles for a uniformly controlled R_t gate with a C-NOT gate at the end of the circuit.
    # If reversed == True, it decomposes the gate such that there is a C-NOT gate at the start (in fact, the circuit
    # topology for the reversed decomposition is the reversed one of the original decomposition)

    def _dec_uc_rotations(self, angles, start_index, end_index, reversedDec):
        interval_len_half = (end_index - start_index)//2
        for i in range(start_index, start_index + interval_len_half):
            if not reversedDec:
                angles[i], angles[i + interval_len_half] = self._update_angles(angles[i], angles[i + interval_len_half])
            else:
                angles[i + interval_len_half], angles[i] = self._update_angles(angles[i], angles[i + interval_len_half])
        if interval_len_half <= 1:
            return
        else:
            self._dec_uc_rotations(angles, start_index, start_index + interval_len_half, False)
            self._dec_uc_rotations(angles, start_index + interval_len_half, end_index, True)

    # Calculate new rotation angles according to Shende's decomposition

    def _update_angles(self, a1, a2):
        return (a1 + a2) / 2.0, (a1 - a2) / 2.0
