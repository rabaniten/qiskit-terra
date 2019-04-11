# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Test decomposition of uniformly ccontrolled R_y rotations.
"""

_EPS = 1e-10  # global variable used to chop very small numbers to zero

import unittest

import numpy as np
from parameterized import parameterized

from qiskit import BasicAer
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import execute as q_execute
from qiskit.test import QiskitTestCase


class TestUCY(QiskitTestCase):
    """Qiskit tests for UCY gates."""

    @parameterized.expand(
        [[[0]], [[0.4]], [[0, 0]], [[0, 0.8]], [[0, 0, 1, 1]], [[0, 1, 0.5, 1]],
         [(2 * np.pi * np.random.rand(2 ** 3)).tolist()],
         [(2 * np.pi * np.random.rand(2 ** 4)).tolist()], [(2 * np.pi * np.random.rand(2 ** 5)).tolist()]]
    )
    def test_ucy(self, angles):
        num_contr = int(np.log2(len(angles)))
        q = QuantumRegister(num_contr + 1)
        # test the UC R_y gate for all possible basis states.
        for i in range(2 ** (num_contr + 1)):
            qc = QuantumCircuit(q)
            # ToDo: the next line can be removed after fixing the state simulator (can't simulate the empty circuit
            # ToDo: at the moment.
            qc.iden(q[0])
            binary_rep = _get_binary_rep_as_list(i, num_contr + 1)
            for j in range(len(binary_rep)):
                if binary_rep[j] == 1:
                    qc.x(q[- (j + 1)])
            qc.ucy(angles, q[1:num_contr + 1], q[0])
            # ToDo: improve efficiency here by allowing to execute circuit on several states in parallel (this would
            # ToDo: in particular allow to get out the isometry the circuit is implementing by applying it to the first
            # ToDo: few basis vectors
            vec_out = np.asarray(q_execute(qc, BasicAer.get_backend(
                'statevector_simulator')).result().get_statevector(qc, decimals=16))
            vec_desired = _apply_ucy_to_basis_state(angles, i)
            # It is fine if the gate is implemented up to a global phase (however, the phases between the different
            # outputs for different bases states must be correct!)
            if i == 0:
                global_phase = _get_global_phase(vec_out, vec_desired)
            vec_desired = (global_phase * vec_desired).tolist()
            dist = np.linalg.norm(np.array(vec_desired - vec_out))
            self.assertAlmostEqual(dist, 0)


def _apply_ucy_to_basis_state(angles, basis_state):
    num_qubits = int(np.log2(len(angles)) + 1)
    angle = angles[basis_state // 2]
    ry = np.array([[np.cos(angle / 2), - np.sin(angle / 2)], [np.sin(angle / 2), np.cos(angle / 2)]])
    state = np.zeros(2 ** num_qubits, dtype=complex)
    if basis_state / 2. == float(basis_state // 2):
        target_state = np.dot(ry, np.array([[1], [0]]))
        state[basis_state] = target_state[0, 0]
        state[basis_state + 1] = target_state[1, 0]
    else:
        target_state = np.dot(ry, np.array([[0], [1]]))
        state[basis_state - 1] = target_state[0, 0]
        state[basis_state] = target_state[1, 0]
    return state


def _get_global_phase(a, b):
    for i in range(len(b)):
        if abs(b[i]) > _EPS:
            return a[i] / b[i]


def _get_binary_rep_as_list(n, num_digits):
    binary_string = np.binary_repr(n).zfill(num_digits)
    binary = []
    for line in binary_string:
        for c in line:
            binary.append(int(c))
    return binary


if __name__ == '__main__':
    unittest.main()
