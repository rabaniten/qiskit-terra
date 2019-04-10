# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Decomposition for uniformly controlled R_z rotation test.
"""

import unittest

from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import BasicAer
import numpy as np
from parameterized import parameterized
from qiskit import execute as q_execute
from qiskit.test import QiskitTestCase


class TestUCZ(QiskitTestCase):
    """Qiskit ZYZ-decomposition tests."""
    @parameterized.expand(
        [[[0,0]],[[0,0.8]],[[0,0,1,1]],[[0,1,0.5,1]],[(2*np.pi*np.random.rand(2**3)).tolist()],
        [(2 * np.pi * np.random.rand(2 ** 4)).tolist()],[(2*np.pi*np.random.rand(2**5)).tolist()]]
    )
    def test_ucz(self, angles):
        num_con = int(np.log2(len(angles)))
        q = QuantumRegister(num_con+1, name='c')
        # test the UC R_z gate for all possible basis states.
        for i in range(2**(num_con+1)):
            qc = QuantumCircuit(q)
            binary_rep = get_binary_rep_as_list(i, num_con+1)
            for j in range(len(binary_rep)):
                if binary_rep[j] == 1:
                    qc.x(q[- (j+1)])
            qc.ucz(angles, q[1:num_con+1], q[0])
            # ToDo: improve efficiency here by allowing to execute circuit on several states in parallel (this would
            # ToDo: in particular allow to get out the isometry the circuit is implementing by applying it to the first
            # ToDo: few basis vectors
            vec_out = np.asarray(q_execute(qc, BasicAer.get_backend(
                'statevector_simulator')).result().get_statevector(qc, decimals=16))
            vec_desired = apply_ucz_to_basis_state(angles, i)
            if i == 0:
                global_phase = vec_out[0]/vec_desired[0]
            vec_desired = (global_phase*vec_desired).tolist()
            dist = np.linalg.norm(np.array(vec_desired - vec_out))
            self.assertAlmostEqual(dist, 0)


def apply_ucz_to_basis_state(angles, basis_state):
    num_qubits = int(np.log2(len(angles))+1)
    angle = angles[basis_state//2]
    rz = np.array([[np.exp(-1.j*angle/2), 0], [0, np.exp(1.j*angle/2)]])
    state = np.zeros(2 ** num_qubits, dtype=complex)
    if basis_state/2. == float(basis_state//2):
        target_state = np.dot(rz, np.array([[1], [0]]))
        state[basis_state] = target_state[0, 0]
        state[basis_state+1] = target_state[1, 0]
    else:
        target_state = np.dot(rz, np.array([[0], [1]]))
        state[basis_state-1] = target_state[0, 0]
        state[basis_state] = target_state[1, 0]
    return state


def get_binary_rep_as_list(n, num_digits):
    binary_string = np.binary_repr(n).zfill(num_digits)
    binary = []
    for line in binary_string:
        for c in line:
            binary.append(int(c))
    return binary

if __name__ == '__main__':
    unittest.main()
