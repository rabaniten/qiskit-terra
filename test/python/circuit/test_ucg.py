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
Decomposition for uniformly controlled single-qubit unitaries test.
"""

import unittest

from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import BasicAer
import numpy as np
from parameterized import parameterized
from qiskit import execute as q_execute
from qiskit.test import QiskitTestCase
from scipy.stats import unitary_group

_id = np.eye(2,2)
_not = np.matrix([[0,1],[1,0]])

class TestUCG(QiskitTestCase):
    """Qiskit UCG tests."""
    @parameterized.expand(
        [[[_id,_id]],[[_id,1j*_id]],[[_id,_not,_id,_not]],[[unitary_group.rvs(2) for i in range(2**2)]],
         [[unitary_group.rvs(2) for i in range(2**3)]],[[unitary_group.rvs(2) for i in range(2**4)]]]
    )
    def test_ucg(self, squs):
        num_con = int(np.log2(len(squs)))
        q = QuantumRegister(num_con+1, name='c')
        # test the UC gate for all possible basis states.
        for i in range(2**(num_con+1)):
            qc = QuantumCircuit(q)
            binary_rep = get_binary_rep_as_list(i, num_con+1)
            for j in range(len(binary_rep)):
                if binary_rep[j] == 1:
                    qc.x(q[- (j+1)])
            qc.ucg(squs, q[1:num_con + 1], q[0])
            # ToDo: improve efficiency here by allowing to execute circuit on several states in parallel (this would
            # ToDo: in particular allow to get out the isometry the circuit is implementing by applying it to the first
            # ToDo: few basis vectors
            vec_out = np.asarray(q_execute(qc, BasicAer.get_backend(
                'statevector_simulator')).result().get_statevector(qc, decimals=16))
            vec_desired = apply_squ_to_basis_state(squs, i)
            # It is fine if the gate is implemented up to a global phase (however, the phases between the different
            # outputs for different bases states must be correct!
            if i == 0:
                global_phase = vec_out[0]/vec_desired[0]
            vec_desired = (global_phase*vec_desired).tolist()
            # Remark: We should not take the fidelity to measure the overlap over the states, since the fidelity ignores
            # the global phase (and hence the phase relation between the different columns of the unitary that the gate
            # should implement)
            dist = np.linalg.norm(np.array(vec_desired - vec_out))
            self.assertAlmostEqual(dist, 0)

def apply_squ_to_basis_state(squs, basis_state):
    num_qubits = int(np.log2(len(squs)) + 1)
    sqg = squs[basis_state // 2]
    state = np.zeros(2 ** num_qubits, dtype=complex)
    if basis_state/2. == float(basis_state//2):
        target_state = np.dot(sqg, np.array([[1], [0]]))
        state[basis_state] = target_state[0, 0]
        state[basis_state+1] = target_state[1, 0]
    else:
        target_state = np.dot(sqg, np.array([[0], [1]]))
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
