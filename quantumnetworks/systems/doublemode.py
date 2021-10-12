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
        self.A_in = A_in if A_in else lambda t: 0
        self.B_in = B_in if B_in else lambda t: 0

    def _param_validation(self):
        if "omega_a" not in self.params:
            self.params["omega_a"] = 2 * np.pi * 1  # 2pi * GHz
        if "omega_b" not in self.params:
            self.params["omega_b"] = 2 * np.pi * 2  # 2pi * GHz
        if "kappa_a" not in self.params:
            self.params["kappa_a"] = 2 * np.pi * 0.001  # 2pi * GHz
        if "kappa_b" not in self.params:
            self.params["kappa_b"] = 2 * np.pi * 0.005  # 2pi * GHz
        if "gamma_a" not in self.params:
            self.params["gamma_a"] = 2 * np.pi * 0.002  # 2pi * GHz
        if "gamma_b" not in self.params:
            self.params["gamma_b"] = 2 * np.pi * 0.002  # 2pi * GHz
        if "kerr_a" not in self.params:
            self.params["kerr_a"] = 2 * np.pi * 0.001  # 2pi * GHz
        if "kerr_b" not in self.params:
            self.params["kerr_b"] = 2 * np.pi * 0.001  # 2pi * GHz
        if "g_ab" not in self.params:
            self.params["g_ab"] = 2 * np.pi * 0.002  # 2pi * GHz

    # Known System Parameters and Load
    # =================================

    @property
    def A(self):
        if self._A is None:
            omega_a = self.params["omega_a"]
            kappa_a = self.params["kappa_a"]
            gamma_a = self.params["gamma_a"]
            omega_b = self.params["omega_b"]
            kappa_b = self.params["kappa_b"]
            gamma_b = self.params["gamma_b"]

            g_ab = self.params["g_ab"]

            A = np.zeros((4, 4))
            A[0, 0] = -kappa_a / 2 - gamma_a / 2
            A[1, 1] = -kappa_a / 2 - gamma_a / 2
            A[2, 2] = -kappa_b / 2 - gamma_b / 2
            A[3, 3] = -kappa_b / 2 - gamma_b / 2

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

    # Nonlinear
    # =================================
    def f_nl(self, x: np.ndarray):
        """
        Nonlinear part of eq of motion
        """
        K_a = self.params["kerr_a"]
        K_b = self.params["kerr_b"]
        Ks = [K_a, K_b]

        non_linearity = np.zeros_like(x)
        for mode in [0, 1]:
            qi = 0 + mode * 2
            pi = 1 + mode * 2
            q = x[qi]
            p = x[pi]
            K = Ks[mode]
            non_linearity[qi] = 2 * K * (q ** 2 + p ** 2) * p
            non_linearity[pi] = -2 * K * (q ** 2 + p ** 2) * q
        return non_linearity

    def Jf_nl(self, x: np.ndarray):
        """
        Jacobian of nonlinear part of eq of motion
        """
        K_a = self.params["kerr_a"]
        K_b = self.params["kerr_b"]
        Ks = [K_a, K_b]

        nonlinear_Jf = np.zeros((x.size, x.size))

        for mode in [0, 1]:
            qi = 0 + mode * 2
            pi = 1 + mode * 2
            q = x[qi]
            p = x[pi]
            K = Ks[mode]
            nonlinear_Jf[qi][qi] = 4 * K * q * p
            nonlinear_Jf[qi][pi] = 2 * K * (q ** 2 + p ** 2) + 4 * K * p ** 2
            nonlinear_Jf[pi][qi] = -2 * K * (q ** 2 + p ** 2) - 4 * K * q ** 2
            nonlinear_Jf[pi][pi] = -4 * K * q * p
        return nonlinear_Jf

    # Eval
    # =================================

    def eval_f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        f = self.A.dot(x) + self.f_nl(x) + u
        return f

    def eval_u(self, t: float):
        A_in = self.A_in(t)
        B_in = self.B_in(t)
        in_vec = np.array([np.real(A_in), np.imag(A_in), np.real(B_in), np.imag(B_in)])
        u = self.B.dot(in_vec)
        return u

    def eval_Jf(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A + self.Jf_nl(x)
