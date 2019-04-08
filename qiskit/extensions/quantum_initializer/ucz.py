# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# The structure of the code is based on Emanuel Malvetti's semester thesis at ETH in 2018,
# which was supervised by Raban Iten and Prof. Renato Renner.

"""
Implementation of the abstract class UCRot for uniformly controlled (also called multiplexed) single-qubit rotations
around the Z-axes (i.e., uniformly controlled R_z rotations).
These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state ket(i) (in the computational bases),
a single-qubit rotation R_z(a_i) is applied to the target qubit.
"""

import math

import numpy as np

from qiskit.extensions.quantum_initializer._ucrot import UCRot
from qiskit.circuit import CompositeGate
from qiskit.circuit.quantumcircuit import QuantumCircuit

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class UCZ(UCRot):  # pylint: disable=abstract-method
    """Uniformly controlled R_z rotations (also called multiplexed rotations). The decomposition is based on
    "Synthesis of Quantum Logic Circuits" by V. Shende et al. (https://arxiv.org/pdf/quant-ph/0406176.pdf)
    angle_list = list (real) rotation angles [a_0,...,a_{2^k-1}]
    q_controls =  list of control k qubits (at least of length one). The qubits are ordered according to their
                    significance in in increasing order.
                    For example if q_controls=[q[2],q[1]] (with q = QuantumRegister(2)), the rotation R_z(a_0)is
                    performed if q[2] and q[1] are in the state zero,
                    R_z(a_1) is performed if q[2] is in the state one and q[1] is in the state zero, and so on.
    q_target = target qubit, where we act on with the single-qubit gates.
    circ = QuantumCircuit or CompositeGate containing this gate
    """

    def __init__(self, angle_list, q_controls, q_target, circ=None):
        super().__init__(angle_list, q_controls, q_target, "Z", circ)
        # call to generate the circuit that takes the desired vector to zero
        self._dec_uc_rot_gate()


def ucz(self, angle_list, q_controls, q_target):
    return self._attach(UCZ(angle_list, q_controls, q_target, self))

QuantumCircuit.ucz = ucz
CompositeGate.ucz = ucz
