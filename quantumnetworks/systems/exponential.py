"""
Exponential system
"""
import numpy as np

from quantumnetworks.systems.base import SystemSolver


class ExpSystem(SystemSolver):
    def _param_validation(self):
        if "lambda" not in self.params:
            self.params["lambda"] = 2

    def eval_f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return -1 * self.params["lambda"] * x + u

    def eval_u(self, t: float):
        return t

    def eval_Jf(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([[-1 * self.params["lambda"]]])
