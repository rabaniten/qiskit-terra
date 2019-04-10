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

_EPS = 1e-10  # global variable used to chop very small numbers to zero

class TestUCG(QiskitTestCase):
    """Qiskit UCG tests."""
    @parameterized.expand(
        [[np.eye(4,4)],[unitary_group.rvs(4)[:,0:2]],[np.eye(4,4)[:,0:2]],[unitary_group.rvs(4)],
         [unitary_group.rvs(8)[:,0:4]],[unitary_group.rvs(8)],[unitary_group.rvs(16)],[unitary_group.rvs(16)[:,0:8]]]
    )
    def test_isometry(self, iso):
        num_q_input = int(np.log2(iso.shape[1]))
        num_q_ancilla_for_output = int(np.log2(iso.shape[0])) - num_q_input
        n = num_q_input + num_q_ancilla_for_output
        q = QuantumRegister(n)
        # test the isometry for all possible input basis states.
        for i in range(2**num_q_input):
            qc = QuantumCircuit(q)
            binary_rep = get_binary_rep_as_list(i, n)
            for j in range(len(binary_rep)):
                if binary_rep[j] == 1:
                    qc.x(q[- (j+1)])
            qc.iso(iso, q[:num_q_input], q[num_q_input:])
            # ToDo: improve efficiency here by allowing to execute circuit on several states in parallel (this would
            # ToDo: in particular allow to get out the isometry the circuit is implementing by applying it to the first
            # ToDo: few basis vectors
            vec_out = np.asarray(q_execute(qc, BasicAer.get_backend(
                'statevector_simulator')).result().get_statevector(qc, decimals=16))
            vec_desired = apply_isometry_to_basis_state(iso, i)
            # It is fine if the gate is implemented up to a global phase (however, the phases between the different
            # outputs for different bases states must be correct!
            if i == 0:
                global_phase = get_global_phase(vec_out, vec_desired)
            vec_desired = (global_phase*vec_desired).tolist()
            # Remark: We should not take the fidelity to measure the overlap over the states, since the fidelity ignores
            # the global phase (and hence the phase relation between the different columns of the unitary that the gate
            # should implement)
            dist = np.linalg.norm(np.array(vec_desired - vec_out))
            self.assertAlmostEqual(dist, 0)

 # ToDo: check "up to diagonal" option

def apply_isometry_to_basis_state(iso, basis_state):
    return iso[:,basis_state]


def get_binary_rep_as_list(n, num_digits):
    binary_string = np.binary_repr(n).zfill(num_digits)
    binary = []
    for line in binary_string:
        for c in line:
            binary.append(int(c))
    return binary


def get_global_phase(a, b):
    for i in range(len(b)):
        if abs(b[i]) > _EPS:
            return a[i]/b[i]

if __name__ == '__main__':
    unittest.main()
