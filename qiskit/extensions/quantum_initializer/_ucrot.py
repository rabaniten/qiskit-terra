# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# The structure of the code is based on Emanuel Malvetti's semester thesis at ETH in 2018,
# which was supervised by Raban Iten and Prof. Renato Renner.

"""
(Abstract) base class for uniformly controlled (also called multiplexed) single-qubit rotations R_t.
This class provides a basis for the decomposition of uniformly controlled R_y and R_z gates (i.e., for t=y,z).
These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state ket(i) (in the computational bases),
a single-qubit rotation R_t(a_i) is applied to the target qubit for a (real) angle a_i.
"""

import math

import numpy as np

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.quantumcircuit import QuantumRegister
from qiskit.exceptions import QiskitError

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class UCRot(Gate):
    """
    Uniformly controlled rotations (also called multiplexed rotations). The decomposition is based on
    'Synthesis of Quantum Logic Circuits' by V. Shende et al. (https://arxiv.org/pdf/quant-ph/0406176.pdf)

    Input:
    angle_list = list of (real) rotation angles [a_0,...,a_{2^k-1}]. Must have at least one entry.

    rot_axis = rotation axis for the single qubit rotations (currently, "Z" and "Y" are supported)
    """

    def __init__(self, angle_list, rot_axis):
        self.rot_axes = rot_axis

        """Check types"""
        # Check if angle_list has type "list"
        if not type(angle_list) == list:
            raise QiskitError("The angles are not provided in a list.")
        # Check if the angles in angle_list are real numbers
        for a in angle_list:
            try:
                float(a)
            except:
                raise QiskitError("An angle cannot be converted to type float (real angles are expected).")

        """Check input form"""
        num_contr = math.log2(len(angle_list))
        if num_contr < 0 or not num_contr.is_integer():
            raise QiskitError("The number of controlled rotation gates is not a non-negative power of 2.")
        if rot_axis != "Y" and rot_axis != "Z":
            raise QiskitError("Rotation axis is not supported.")
        # Create new gate.
        num_qubits = int(num_contr) + 1
        super().__init__("UC" + rot_axis, num_qubits, angle_list)

    def _define(self):
        ucr_circuit = self._dec_ucrot()
        gate_num = len(ucr_circuit.data)
        gate = ucr_circuit.to_instruction()
        q = QuantumRegister(self.num_qubits)
        ucr_circuit = QuantumCircuit(q)
        if gate_num == 0:
            # ToDo: if we would not add the identity here, this would lead to troubles simulating the circuit afterwards.
            #  this should probably be fixed in the bahaviour of QuantumCircuit.
            ucr_circuit.iden(q[0])
        else:
            ucr_circuit.append(gate, q[:])
        self.definition = ucr_circuit.data

    """
    finds a decomposition of a UC rotation gate into elementary gates (C-NOTs and single-qubit rotations).
    """

    def _dec_ucrot(self):
        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q)
        q_target = q[0]
        q_controls = q[1:]
        if len(q_controls) == 0:
            if self.rot_axes == "Z":
                if np.abs(self.params[0]) > _EPS:
                    circuit.rz(self.params[0], q_target)
            if self.rot_axes == "Y":
                if np.abs(self.params[0]) > _EPS:
                    circuit.ry(self.params[0], q_target)
        else:
            # First, we find the rotation angles of the single-qubit rotations acting on the target qubit
            angles = self.params.copy()
            _dec_uc_rotations(angles, 0, len(angles), False)
            # Now, it is easy to place the C-NOT gates to get back the full decomposition.
            for i in range(len(angles)):
                if self.rot_axes == "Z":
                    if np.abs(angles[i]) > _EPS:
                        circuit.rz(angles[i], q_target)
                if self.rot_axes == "Y":
                    if np.abs(angles[i]) > _EPS:
                        circuit.ry(angles[i], q_target)
                # Determine the index of the qubit we want to control the C-NOT gate. Note that it corresponds
                # to the number of trailing zeros in the binary representaiton of i+1
                if not i == len(angles) - 1:
                    binary_rep = np.binary_repr(i + 1)
                    q_contr_index = len(binary_rep) - len(binary_rep.rstrip('0'))
                else:
                    # Handle special case:
                    q_contr_index = len(q_controls) - 1
                circuit.cx(q_controls[q_contr_index], q_target)
        return circuit


# Calculates rotation angles for a uniformly controlled R_t gate with a C-NOT gate at the end of the circuit.
# The rotation angles of the gate R_t are stored in angles[start_index:end_index].
# If reversed == True, it decomposes the gate such that there is a C-NOT gate at the start of the circuit
# (in fact, the circuit topology for the reversed decomposition is the reversed one of the original decomposition)
def _dec_uc_rotations(angles, start_index, end_index, reversedDec):
    interval_len_half = (end_index - start_index) // 2
    for i in range(start_index, start_index + interval_len_half):
        if not reversedDec:
            angles[i], angles[i + interval_len_half] = _update_angles(angles[i], angles[i + interval_len_half])
        else:
            angles[i + interval_len_half], angles[i] = _update_angles(angles[i], angles[i + interval_len_half])
    if interval_len_half <= 1:
        return
    else:
        _dec_uc_rotations(angles, start_index, start_index + interval_len_half, False)
        _dec_uc_rotations(angles, start_index + interval_len_half, end_index, True)


# Calculate the new rotation angles according to Shende's decomposition

def _update_angles(a1, a2):
    return (a1 + a2) / 2.0, (a1 - a2) / 2.0
