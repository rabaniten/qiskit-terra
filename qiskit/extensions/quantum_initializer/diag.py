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
import cmath
import math

import numpy as np

from qiskit.circuit import CompositeGate
from qiskit.exceptions import QiskitError
from qiskit.extensions.quantum_initializer.ucz import UCZ
from qiskit.circuit.quantumcircuit import QuantumCircuit, QuantumRegister

_EPS = 1e-10  # global variable used to chop very small numbers to zero

#ToDo: We could also input the diagonal gate by only providing the phases of the entries. This would be more efficient,
#ToDo: however, maybe a bit less user friendly.

class DiagGate(CompositeGate):  # pylint: disable=abstract-method
    """
    diag = list of the 2^k diagonal entries
    q = list of k qubits the diagonal is acting on (the order of the qubits specifies the computational basis in
    which the diagonal gate is provided (i,e,. the first element in diag acts on the state where all the qubits in q
    are in the state 0, the second entry acts on the state where all the qubits q[1],...,q[k-1] are in the state zero
    and q[0] is in the state 1, and so on.
    circ = QuantumCircuit or CompositeGate containing this gate
    """

    def __init__(self, diag, q, circ=None):
        """Check types"""
        # Check if q has type "list"
        if not type(q) == list:
            raise QiskitError(
                "The qubits must be provided as a list (also if there is only one qubit).")
        # Check if the entries in q are qubits
        for qu in q:
            if not (type(qu) == tuple and type(qu[0]) == QuantumRegister):
                raise QiskitError("Wrong type: there is a qubit which is not of the type part of a QauntumRegister.")
        # Check if diag has type "list"
        if not type(diag) == list:
            raise QiskitError(
                "The diagonal entries are not provided in a list.")

        """Check input form"""
        # Check if the right number of diagonal entries is provided and if the diagonal entries have absolute value one
        num_action_qubits = math.log2(len(diag))
        if num_action_qubits <= 0 or not num_action_qubits.is_integer():
            raise QiskitError("The number of diagonal entrie is not a positive power of 2.")
        for z in diag:
            try:
                complex(z)
            except:
                raise QiskitError("Not all of the diagonal entries can be converted to complex numbers.")
            if not len(q) == num_action_qubits:
                raise QiskitError("The number of diagonal entries does not correspond to the number of qubits.")
            if not np.abs(z) - 1 < _EPS:
                raise QiskitError("A diagonal entry has has not absolute value one.")


        # Create new composite gate.
        super().__init__("init", diag, q, circ)
        # call to generate the circuit corresponding to the diagonal gate
        self.dec_diag()

    def dec_diag(self):
        """
        Call to populate the self.data list with gates that implement the diagonal gate
        """
        n = len(self.params)
        num_qubits = int(np.log2(n))
        # Since the diagonal is a unitary, all its entries have absolute value one and the diagonal is fully specified
        #  by its phases
        diag_phases = [cmath.phase(z) for z in self.params]
        while n >= 2:
            angles_rz = []
            for i in range(0, n, 2):
                diag_phases[i // 2], rz_angle = _extract_rz(diag_phases[i], diag_phases[i + 1])
                angles_rz.append(rz_angle)
            num_act_qubits = int(np.log2(n))
            contr_qubits = self.qargs[num_qubits-num_act_qubits + 1:num_qubits]
            target_qubit = self.qargs[num_qubits-num_act_qubits]
            self._attach(UCZ(angles_rz, contr_qubits, target_qubit))
            n //= 2


# extract a Rz rotation (angle given by first output) such that exp(j*phase)*Rz(z_angle) is equal to the diagonal matrix
# with entriew exp(1j*ph1) and exp(1j*ph2)
def _extract_rz(phi1, phi2):
    phase = (phi1 + phi2) / 2.0
    z_angle = phi2 - phi1
    return phase, z_angle


def diag_gate(self, diag, q):
    return self._attach(DiagGate(diag, q, self))

QuantumCircuit.diag_gate = diag_gate
CompositeGate.diag_gate = diag_gate