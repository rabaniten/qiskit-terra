# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================



# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

"""
Decomposition for uniformly conteolled R_z rotation test.
"""

import unittest

from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import BasicAer
import numpy as np
from parameterized import parameterized
from qiskit import execute as q_execute
from qiskit.test import QiskitTestCase


class TestDiagGate(QiskitTestCase):
    """Diagonal gate tests."""
    @parameterized.expand(
        [[[0,0]],[[0,0.8]],[[0,0,1,1]],[[0,1,0.5,1]],[(2*np.pi*np.random.rand(2**3)).tolist()],
        [(2 * np.pi * np.random.rand(2 ** 4)).tolist()],[(2*np.pi*np.random.rand(2**5)).tolist()]]
    )
    def test_diag_gate(self, phases):
        diag = [np.exp(1j*ph) for ph in phases]
        num_qubits = int(np.log2(len(diag)))
        q = QuantumRegister(num_qubits)
        # test the diagonal gate for all possible basis states.
        for i in range(2**(num_qubits)):
            qc = QuantumCircuit(q)
            binary_rep = get_binary_rep_as_list(i, num_qubits)
            for j in range(len(binary_rep)):
                if binary_rep[j] == 1:
                    qc.x(q[- (j+1)])
            qc.diag_gate(diag, q[0:num_qubits])
            vec_out = np.asarray(q_execute(qc, BasicAer.get_backend(
                'statevector_simulator')).result().get_statevector(qc, decimals=16))
            vec_desired = apply_diag_gate_to_basis_state(phases, i)
            if i == 0:
                global_phase = vec_out[0]/vec_desired[0]
            vec_desired = (global_phase*vec_desired).tolist()
            # Remark: We should not take the fidelity to measure the overlap over the states, since the fidelity ignores
            # the global phase (and hence the phase relation between the different columns of the unitary that the gate
            # should implement)
            dist = np.linalg.norm(np.array(vec_desired - vec_out))
            self.assertAlmostEqual(dist, 0)


if __name__ == '__main__':
    unittest.main()


def apply_diag_gate_to_basis_state(phases, basis_state):
    # ToDo: improve efficiency here by implementing a simulation for diagonal gates
    num_qubits = int(np.log2(len(phases)))
    ph = phases[basis_state]
    state = np.zeros(2 ** num_qubits, dtype=complex)
    state[basis_state] = np.exp(1j*ph)
    return state


def get_binary_rep_as_list(n, num_digits):
    binary_string = np.binary_repr(n).zfill(num_digits)
    binary = []
    for line in binary_string:
        for c in line:
            binary.append(int(c))
    return binary
