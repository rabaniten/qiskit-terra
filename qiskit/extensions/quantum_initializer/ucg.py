# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# The structure of the code is based on Emanuel Malvetti's semester thesis at ETH in 2018,
# which was supervised by Raban Iten and Prof. Renato Renner.

"""
Uniformly controlled gates (also called multiplexed gates). These gates can have several control qubits and a single target qubit.
If the k control qubits are in the state ket(i) (in the computational bases), a single-qubit unitary U_i is applied to the target qubit.
"""

import math

import numpy as np

from qiskit.circuit import CompositeGate
from qiskit.circuit.quantumcircuit import QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.extensions.quantum_initializer._isometry import is_isometry
from qiskit.extensions.quantum_initializer.diag import DiagGate
from qiskit.extensions.quantum_initializer.zyz_dec import SingleQubitUnitary
from qiskit.extensions.standard.cx import CnotGate
import cmath

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class UCG(CompositeGate):  # pylint: disable=abstract-method
    """Uniformly controlled gates (also called multiplexed gates). The decomposition is based on: https://arxiv.org/pdf/quant-ph/0410066.pdf.
    gate_list = list of two qubit unitaries [U_0,...,U_{2^k-1}], where each single-qubit unitary U_i is a given as a 2*2 numpy array.
    q_controls = list of control k qubits. The qubits are ordered according to their significance in the computational basis.
                    For example if q_controls=[q[2],q[1]] (with q = QuantumRegister(2)), the unitary U_0 is performed if q[2] and q[1] are in the state zero,
                    U_1 is performed if q[2] is in the state zero and q[1] is in the state one, and so on.
    q_target = target qubit, where we act on with the single-qubit gates.
    circ = QuantumCircuit or CompositeGate containing this gate
    """

    def __init__(self, gate_list, q_controls, q_target, up_to_diagonal=False, circ=None):
        self.q_controls = q_controls
        self.q_target = q_target
        """Check types"""
        # Check if q_controls has type "list"
        if not type(q_controls) == list:
            raise QiskitError(
                "The control qubits must be provided as a list (also if there is only one control qubit).")
        # Check if the entries in q_controls are qubits
        for qu in q_controls:
            if not (type(qu) == tuple and type(qu[0]) == QuantumRegister):
                raise QiskitError("There is a control qubit which is not of the type used for qubits.")
        # Check if gate_list has type "list"
        if not type(gate_list) == list:
            raise QiskitError(
                "The single-qubit unitaries are not provided in a list.")
        # Check if the gates in gate_list are numpy arrays
        for gate in gate_list:
            # if not type(gate) == np.matrix:
            #     raise QiskitError("A single-qubit gate is not of type numpy.matrix.")
            if not gate.shape == (2, 2):
                raise QiskitError("The dimension of a controlled gate is not equal to (2,2).")
        # Check if there is one target qubit provided
        if not (type(q_target) == tuple and type(q_target[0]) == QuantumRegister):
            raise QiskitError("The target qubit is not a single qubit from a QuantumRegister.")

        """Check if the input has the correct form"""
        # Check if number of gates in gate_list is a positive power of two
        num_contr = math.log2(len(gate_list))
        if num_contr < 0 or not num_contr.is_integer():
            raise QiskitError("The number of controlled single-qubit gates is not a non negative power of 2.")
        # Check if number of control qubits does correspond to the number of single-qubit rotations
        if num_contr != len(q_controls):
            raise QiskitError("Number of controlled gates does not correspond to the number of control-qubits.")
        # Check if the single-qubit gates have the right dimensions and if they are unitaries
        for gate in gate_list:
            if not is_isometry(gate, _EPS):
                raise QiskitError("A controlled gate is not unitary.")

        # Create new composite gate.
        num_qubits = len(q_controls) + 1
        self.num_qubits = int(num_qubits)
        self.diag = np.ones(self.num_qubits)
        qubits = [q_target] + q_controls
        super().__init__("init", gate_list, qubits, circ)
        # Check if the target qubit is also a control qubit
        self._check_dups(qubits, message="The target qubit cannot also be listed as a control qubit.")
        # call to generate the circuit that implements the UCG
        self.dec_ucg(up_to_diagonal)

    def dec_ucg(self, up_to_diagonal):
        """
        Call to populate the self.data list with gates that implement the uniformly controlled gate
        """
        if len(self.q_controls) == 0:
            if up_to_diagonal:
                sqg = SingleQubitUnitary(self.params[0], self.q_target, up_to_diagonal=True)
                self._attach(sqg)
                self.diag = squ.diag
            else:
                sqg = SingleQubitUnitary(self.params[0], self.q_target)
                self._attach(sqg)
            return None
        # First, we find the single qubit gates of the decomposition.
        (single_qubit_gates, diag) = self._dec_ucg_help()
        # Now, it is easy to place the C-NOT gates and some Hadamards and Rz(pi/2) gates (which are absorbed into the
        # single-qubit unitaries) to get out the full decomposition.
        for i in range(len(single_qubit_gates)):
            # Absorb Hadamards and Rz(pi/2) gates
            if i == 0:
                squ = h().dot(single_qubit_gates[i])
            elif i == len(single_qubit_gates)-1:
                squ = single_qubit_gates[i].dot(rz(np.pi/2)).dot(h())
            else:
                squ = h().dot(single_qubit_gates[i].dot(rz(np.pi/2))).dot(h())
            self._attach(SingleQubitUnitary(squ, self.q_target))
            # The number of the control qubit (labeling the control qubits from the bottom to the top,
            # starting with zero) is given by the number of zeros at the end of the binary representation of (i+1)
            binary_rep = np.binary_repr(i + 1)
            num_trailing_zeros = len(binary_rep) - len(binary_rep.rstrip('0'))
            q_contr_index = num_trailing_zeros
            if not i == len(single_qubit_gates)-1:
                self._attach(CnotGate(self.q_controls[q_contr_index], self.q_target))
        if up_to_diagonal:
            self.diag = diag
        else:
            # The diagonal gate is given in the basis [c_k,...,c_0,t], where c_i are the control qubits and t denotes
            # the target qubit
            self._attach(DiagGate(diag.tolist(), [self.q_target] + self.q_controls))


    def _dec_ucg_help(self):
        """
        This method finds the single qubit gate arising in the decomposition of UCGs given in  
        https://arxiv.org/pdf/quant-ph/0410066.pdf.
        """
        single_qubit_gates = [gate.astype(complex) for gate in self.params]
        diag = np.ones(2 ** self.num_qubits, dtype=complex)
        num_contr = len(self.q_controls)
        for dec_step in range(num_contr):
            num_ucgs = 2 ** dec_step
            # The decomposition works recursively and the following loop goes over the different UCGs that arise
            # in the decomposition
            for ucg_index in range(num_ucgs):
                len_ucg = 2 ** (num_contr - dec_step)
                for i in range(int(len_ucg / 2)):
                    shift = ucg_index * len_ucg
                    a = single_qubit_gates[shift + i]
                    b =single_qubit_gates[shift + len_ucg // 2 + i]
                    # Apply the decomposition for UCGs given in in equation (3) in
                    # https://arxiv.org/pdf/quant-ph/0410066.pdf
                    # to demultiplex one control of all the num_ucgs uniformly-controlled gates
                    #  with log2(len_ucg) uniform controls
                    v, u, r = self._demultiplex_single_uc(a, b)
                    single_qubit_gates[shift + i] = v
                    single_qubit_gates[shift + len_ucg //2  + i] = u
                    # Now we decompose the gates D as described in Figure 4  in
                    # https://arxiv.org/pdf/quant-ph/0410066.pdf and merge some of the gates into the UCGs and the
                    # diagonal at the end of the circuit

                    # Remark: The Rz(pi/2) rotation acting on the target qubit and the Hadamard gates arising in the
                    # decomposition of D are ignored for the moment (they will be added together with the C-NOT
                    # gates at the end of the decomposition)
                    if ucg_index < num_ucgs - 1:
                        # Absorb the Rz(pi/2) rotation on the control into the UC-Rz gate and
                        # merge the UC-Rz rotation with the following UCG, which hasn't been decomposed yet.
                        k = shift + len_ucg + i
                        single_qubit_gates[k] = single_qubit_gates[k].dot(ct(r))*rz(np.pi/2).item((0,0))
                        k = k + len_ucg // 2
                        single_qubit_gates[k] = single_qubit_gates[k].dot(r)*rz(np.pi/2).item((1,1))
                    else:
                        # Absorb the Rz(pi/2) rotation on the control into the UC-Rz gate and
                        # Merge the trailing UC-Rz rotation into a diagonal gate at the end of the circuit
                        for ucg_index_2 in range(num_ucgs):
                            shift_2 = ucg_index_2 * len_ucg
                            k = 2 * (i + shift_2)
                            diag[k] = diag[k] * ct(r).item((0, 0)) * rz(np.pi / 2).item((0, 0))
                            diag[k + 1] = diag[k + 1] * ct(r).item((1, 1)) * rz(np.pi / 2).item((0, 0))
                            k = len_ucg + k
                            diag[k] *= r.item((0, 0)) * rz(np.pi / 2).item((1, 1))
                            diag[k + 1] *= r.item((1, 1)) * rz(np.pi / 2).item((1, 1))
        return single_qubit_gates, diag

    def _demultiplex_single_uc(self, a, b):
        """
        This mehod implements the decomposition given in equation (3) in https://arxiv.org/pdf/quant-ph/0410066.pdf.
        This decomposition is used recursively to decompose uniformly controlled gates.
        a,b = single qubit unitaries
        v,u,r = outcome of the decomposition given in the reference (see there for the details).
        """
        # The notation is chosen as in https://arxiv.org/pdf/quant-ph/0410066.pdf.
        x = a.dot(ct(b))
        det_x = np.linalg.det(x)
        x11 = x.item((0,0))/ cmath.sqrt(det_x)
        phi = cmath.phase(det_x)
        r1 = cmath.exp(1j / 2 * (np.pi / 2 - phi / 2 - cmath.phase(x11)))
        r2 = cmath.exp(1j / 2 * (np.pi / 2 - phi / 2 + cmath.phase(x11) + np.pi))
        r = np.array([[r1, 0], [0, r2]], dtype=complex)
        d, u = np.linalg.eig(r.dot(x).dot(r))
        # If d is not equal to diag(i,-i), then it we must have diag(i,-i). We interchange the eigenvalues and
        # eigenvectors to get it to the "standard" form given in https://arxiv.org/pdf/quant-ph/0410066.pdf.
        if abs(d[0] + 1j) < _EPS:
            d = np.flip(d, 0)
            u = np.flip(u, 1)
        d = np.diag(np.sqrt(d))
        v = d.dot(ct(u)).dot(ct(r)).dot(b)
        return v, u, r

def ct(m):
    return np.transpose(np.conjugate(m))

def h():
    return 1/np.sqrt(2)*np.array([[1,1],[1,-1]])

def rz(alpha):
    return np.array([[np.exp(1j*alpha/2),0],[0,np.exp(-1j*alpha/2)]])

def ucg(self, gate_list, q_controls, q_target, up_to_diagonal = False):
    return self._attach(UCG(gate_list, q_controls, q_target, up_to_diagonal, self))

QuantumCircuit.ucg = ucg
CompositeGate.ucg = ucg
