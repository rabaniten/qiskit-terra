# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

"""
ZYZ decomposition for single-qubit unitary (CompositeGate instance) test.
"""

import unittest

from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import execute, BasicAer
from qiskit.quantum_info import state_fidelity
from qiskit.test import QiskitTestCase
import numpy as np
from scipy.stats import unitary_group
from qiskit.extensions.quantum_initializer.zyz_dec import SingleQubitUnitary


class TestZYZ(QiskitTestCase):
    """Qiskit ZYZ-decomposition tests."""

    _desired_fidelity = 0.99

    def test_identity(self):
        fidelity = _get_fidelity_zyz_dec(np.eye(2, 2))
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_NOT(self):
        fidelity = _get_fidelity_zyz_dec(np.array([[0., 1.], [1., 0.]]))
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_Hadamard(self):
        fidelity = _get_fidelity_zyz_dec(1/np.sqrt(2)*np.array([[1., 1.], [-1., 1.]]))
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_random(self):
        u = unitary_group.rvs(2)
        fidelity = _get_fidelity_zyz_dec(u)
        print(fidelity)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))

    def test_random_up_to_diagonal(self):
        u = unitary_group.rvs(2)
        fidelity = _get_fidelity_zyz_dec(u, up_to_diagonal=True)
        print(fidelity)
        self.assertGreater(
            fidelity, self._desired_fidelity,
            "Initializer has low fidelity {0:.2g}.".format(fidelity))


def _get_fidelity_zyz_dec(u, up_to_diagonal=False):
    qr = QuantumRegister(2, "qr")
    qc = QuantumCircuit(qr)
    # To test a unitary, we apply it to the maximally entangled state. This lead to the Choi state of the unitary,
    # which we then compare to the Choi state of the desired unitary.
    desired_vector = choi_state(u)
    create_maximally_entangled_state(qr, qc)
    qc.zyz(u, qr[0])
    job = execute(qc, BasicAer.get_backend('statevector_simulator'))
    result = job.result()
    statevector = result.get_statevector()
    if up_to_diagonal:
        squ = SingleQubitUnitary(u, qr[0], up_to_diagonal=True)
        statevector[0] *= squ.diag[0]
        statevector[1] *= squ.diag[1]
        fidelity = state_fidelity(statevector, desired_vector)
    else:
        fidelity = state_fidelity(statevector, desired_vector)
    return fidelity


def choi_state(u):
    vector = np.dot(np.kron(np.eye(2, 2), u), 1/np.sqrt(2)*np.array([[1], [0], [0], [1]]))
    return vector[:, 0]

# takes a qubit register with an even number of qubits and a circuit as an input. Creates the maximally entangled
# state between the first half and the second half of the qubits.


def create_maximally_entangled_state(qr, qc):
    for i in range(len(qr)//2):
        qc.h(qr[i])
        qc.cx(qr[i], qr[len(qr)//2+i])


if __name__ == '__main__':
    unittest.main()
