# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Implementation of the abstract class UCRot for uniformly controlled (also called multiplexed) single-qubit rotations
around the X-axes (i.e., uniformly controlled R_x rotations).
These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state ket(i) (in the computational bases),
a single-qubit rotation R_x(a_i) is applied to the target qubit.
"""
import math

from qiskit import QuantumRegister, QiskitError
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.extensions.quantum_initializer._ucrot import UCRot


class UCX(UCRot):
    """
    Uniformly controlled rotations (also called multiplexed rotations). The decomposition is based on
    'Synthesis of Quantum Logic Circuits' by V. Shende et al. (https://arxiv.org/pdf/quant-ph/0406176.pdf)

    Input:
    angle_list = list of (real) rotation angles [a_0,...,a_{2^k-1}]
    """

    def __init__(self, angle_list):
        super().__init__(angle_list, "X")


"""
Attach a uniformly controlled (also called multiplexed gates) Rx rotation gate to a circuit.
The decomposition is base on https://arxiv.org/pdf/quant-ph/0406176.pdf by Shende et al.

    Args:
        angle_list (list[numbers): list of (real) rotation angles [a_0,...,a_{2^k-1}]
        q_controls (QautnumRegister|list[(QuantumRegister,int)]): list of k control qubits (or empty list if
            no controls). The control qubits are ordered according to their significance in increasing order:
            For example if q_controls=[q[1],q[2]] (with q = QuantumRegister(2)), the rotation Rx(a_0)is
            performed if q[1] and q[2] are in the state zero, the rotation  Rx(a_1) is performed if
            q[1] is in the state one and q[2] is in the state zero, and so on.
        q_target (QautnumRegister|(QuantumRegister,int)):  target qubit, where we act on with the single-qubit
            rotation gates.

    Returns:
        QuantumCircuit: the uniformly controlled rotation gate is attached to the circuit.

    Raises:
        QiskitError: if the list number of control qubits does not correspond to the provided number of single-qubit 
            unitaries; if an input is of the wrong type
"""


def ucx(self, angle_list, q_controls, q_target):
    if isinstance(q_controls, QuantumRegister):
        q_controls = q_controls[:]
    if isinstance(q_target, QuantumRegister):
        q_target = q_target[:]
        if len(q_target) == 1:
            q_target = q_target[0]
        else:
            raise QiskitError("The target qubit is a QuantumRegister containing more than one qubits.")
    # Check if q_controls has type "list"
    if not type(angle_list) == list:
        raise QiskitError("The angles must be provided as a list.")
    num_contr = math.log2(len(angle_list))
    if num_contr < 0 or not num_contr.is_integer():
        raise QiskitError("The number of controlled rotation gates is not a non-negative power of 2.")
    # Check if number of control qubits does correspond to the number of rotations
    if num_contr != len(q_controls):
        raise QiskitError("Number of controlled rotations does not correspond to the number of control-qubits.")
    return self.append(UCX(angle_list), [q_target] + q_controls, [])


QuantumCircuit.ucx = ucx