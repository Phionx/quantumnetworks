"""
Driven Double-Mode Linear System with Beam-Splitter Coupling
"""
from typing import Dict
import numpy as np

from quantumnetworks.systems.base import SystemSolver


class DoubleModeSystem(SystemSolver):
    def __init__(self, params: Dict[str, float], A_in=None, B_in=None) -> None:
        """
        Arguments:
            A_in (function): takes in time t and returns np.ndarray
        """
        super().__init__(params)
        self._A = None
        self._B = None
        self.A_in = A_in if A_in else self.default_A_in
        self.B_in = B_in if B_in else self.default_B_in

    def _param_validation(self):
        if "omega_a" not in self.params:
            self.params["omega_a"] = 1  # GHz
        if "omega_b" not in self.params:
            self.params["omega_b"] = 2  # GHz
        if "kappa_a" not in self.params:
            self.params["kappa_a"] = 0.001  # GHz
        if "kappa_b" not in self.params:
            self.params["kappa_b"] = 0.005  # GHz
        if "g_ab" not in self.params:
            self.params["g_ab"] = 0.002  # GHz

    # Known System Parameters and Load
    # =================================

    @property
    def A(self):
        if self._A is None:
            omega_a = self.params["omega_a"]
            kappa_a = self.params["kappa_a"]
            omega_b = self.params["omega_b"]
            kappa_b = self.params["kappa_b"]
            g_ab = self.params["g_ab"]

            A = np.zeros((4, 4))
            A[0, 0] = -kappa_a / 2
            A[1, 1] = -kappa_a / 2
            A[2, 2] = -kappa_b / 2
            A[3, 3] = -kappa_b / 2

            A[0, 1] = omega_a
            A[1, 0] = -omega_a
            A[2, 3] = omega_b
            A[3, 2] = -omega_b

            A[0, 3] = g_ab
            A[1, 2] = -g_ab
            A[2, 1] = g_ab
            A[3, 0] = -g_ab
            self._A = A
        return self._A

    @property
    def B(self):
        if self._B is None:
            kappa_a = self.params["kappa_a"]
            kappa_b = self.params["kappa_b"]
            B = np.zeros((4, 4))
            B[0, 0] = -np.sqrt(kappa_a)
            B[1, 1] = -np.sqrt(kappa_a)
            B[2, 2] = -np.sqrt(kappa_b)
            B[3, 3] = -np.sqrt(kappa_b)
            self._B = B
        return self._B

    def default_A_in(self, t: float):
        omega_a = self.params["omega_a"]
        return np.exp(1j * (omega_a * t))

    def default_B_in(self, t: float):
        omega_b = self.params["omega_b"]
        return np.exp(1j * (omega_b * t))

    # Eval
    # =================================

    def eval_f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        f = self.A.dot(x) + u
        return f

    def eval_u(self, t: float):
        A_in = self.A_in(t)
        B_in = self.B_in(t)
        in_vec = np.array([np.real(A_in), np.imag(A_in), np.real(B_in), np.imag(B_in)])
        u = self.B.dot(in_vec)
        return u

    def eval_Jf(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A
