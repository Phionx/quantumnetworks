"""
Driven Single Mode Linear System
"""
from typing import Dict, Any
import numpy as np

from quantumnetworks.systems.base import SystemSolver


class SingleModeSystem(SystemSolver):
    def __init__(self, params: Dict[str, Any], A_in=None) -> None:
        """
        Arguments:
            A_in (function): takes in time t and returns np.ndarray
        """
        super().__init__(params)
        self._A = None
        self._B = None
        self.A_in = A_in if A_in else self.default_A_in

    def _param_validation(self):
        if "omega_a" not in self.params:
            self.params["omega_a"] = 1  # Ghz
        if "kappa_a" not in self.params:
            self.params["kappa_a"] = 0.001  # Ghz

    # Known System Parameters and Load
    # =================================

    @property
    def A(self):
        if self._A is None:
            omega_a = self.params["omega_a"]
            kappa_a = self.params["kappa_a"]

            A = np.array([[-kappa_a / 2, omega_a], [-omega_a, -kappa_a / 2]])
            self._A = A
        return self._A

    @property
    def B(self):
        if self._B is None:
            kappa_a = self.params["kappa_a"]
            B = np.array([[-np.sqrt(kappa_a), 0], [0, -np.sqrt(kappa_a)]])
            self._B = B
        return self._B

    def default_A_in(self, t: float):
        omega_a = self.params["omega_a"]
        return np.exp(1j * (omega_a * t))

    # Eval
    # =================================

    def eval_f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        f = self.A.dot(x) + u
        return f

    def eval_u(self, t: float):
        A_in = self.A_in(t)
        A_in_vec = np.array([np.real(A_in), np.imag(A_in)])
        u = self.B.dot(A_in_vec)
        return u

    def eval_Jf(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A
