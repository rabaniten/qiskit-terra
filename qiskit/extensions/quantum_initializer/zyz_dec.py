# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Decompose an arbitrary 2*2 unitary into three rotation gates: U=R_zR_yR_z. Note that the decomposition is up to
a global phase shift.
(This is a well known decomposition, which can be found for example in Nielsen and Chuang's book
"Quantum computation and quantum information".)
"""
import cmath
import math

import numpy as np

from qiskit.circuit import CompositeGate
from qiskit.circuit.quantumcircuit import QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.extensions.standard.ry import RYGate
from qiskit.extensions.standard.rz import RZGate

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class SingleQubitUnitary(CompositeGate):  # pylint: disable=abstract-method
    """
    u = 2*2 unitary (given as an (complex) numpy.ndarray)
    q = single qubit, where the unitary is acting on
    circ = QuantumCircuit or CompositeGate containing this gate
    """

    def __init__(self, u, q, up_to_diagonal = False, circ=None):
        # Check if the matrix u has the right dimensions and if it is a unitary
        # if not type(u) == np.ndarray:
        #     raise QiskitError("The input matrix u is not of type numpy.ndarray.")
        if not u.shape == (2, 2):
            raise QiskitError("The dimension of the input matrix is not equal to (2,2).")
        if not is_isometry(u, _EPS):
            raise QiskitError("The 2*2 matrix is not unitary.")

        # Check if there is one qubit provided
        if not (type(q) == tuple and type(q[0]) == QuantumRegister):
            raise QiskitError("The provided qubit is not a single qubit from a QuantumRegister.")

        # Create new composite gate.
        super().__init__("init", [u], [q], circ)
        # Decompose the single-qubit unitary (and save the elementary gate into self.data)
        self._dec_single_qubit_unitary(up_to_diagonal)
        self.diag = [1, 1]

    def _dec_single_qubit_unitary(self, up_to_diagonal=False):
        """
        Call to populate the self.data list with gates that implement the single qubit unitary
        """
        # First, we find the rotation angles (where we can ignore the global phase)
        (a, b, c, _) = self._zyz_dec()
        # Add the gates to the composite gate
        if (a % 4 * np.pi) > _EPS:
            self._attach(RZGate(a, self.qargs[0]))
        if (b % 4 * np.pi) > _EPS:
            self._attach(RYGate(b,  self.qargs[0]))
        if (c % 4 * np.pi) > _EPS:
            if up_to_diagonal:
                self.diag = [np.exp(-1j*c/2.), np.exp(1j*c/2.)]
            else:
                self._attach(RZGate(c, self.qargs[0]))

    def _zyz_dec(self):
        """
        This method finds rotation angles (a,b,c,d) in the decomposition u=exp(id)*Rz(c).Ry(b).Rz(a) (where "." denotes
        matrix multiplication)
        """
        u = self.params[0]
        u00 = u.item(0, 0)
        u01 = u.item(0, 1)
        u10 = u.item(1, 0)
        u11 = u.item(1, 1)
        # Handle special case if the entry (0,0) of the unitary is equal to zero
        if np.abs(u00) < _EPS:
            # Note that u10 can't be zero, since u is unitary (and u00 == 0)
            c = cmath.phase(-u01 / u10)
            d = cmath.phase(u01 * np.exp(-1j * c / 2))
            return 0., np.pi, c, d
        # Handle special case if the entry (0,1) of the unitary is equal to zero
        if np.abs(u01) < _EPS:
            # Note that u11 can't be zero, since u is unitary (and u01 == 0)
            c = cmath.phase(u00 / u11)
            d = cmath.phase(u00 * np.exp(-1j * c / 2))
            return 0, 0, c, d
        b = 2 * np.arccos(np.abs(u00))
        if 0 < np.sin(b / 2) - np.cos(b / 2):
            c = cmath.phase(-u00 / u10)
            a = cmath.phase(u00 / u01)
        else:
            c = -cmath.phase(-u10 / u00)
            a = -cmath.phase(u01 / u00)
        d = cmath.phase(u00 * np.exp(-1j * (a + c) / 2))
        # The decomposition works with another convention for the rotation gates (the one using negative angles).
        # Therefore, we have to take the inverse of the angles at the end.
        return -a, -b, -c, d


def is_isometry(m, eps):
    err = np.linalg.norm(np.dot(np.transpose(np.conj(m)), m) - np.eye(m.shape[1], m.shape[1]))
    return math.isclose(err, 0, abs_tol=eps)


def zyz(self, params, qubits, up_to_diagonal=False):
    return self._attach(SingleQubitUnitary(params, qubits, up_to_diagonal, self))

QuantumCircuit.zyz = zyz
CompositeGate.zyz = zyz
