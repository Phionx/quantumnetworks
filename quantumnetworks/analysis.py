"""
Base Analysis
"""
from typing import Dict, Any
from abc import abstractmethod, ABCMeta
import numpy as np
import matplotlib.pyplot as plt


class SystemSolver(metaclass=ABCMeta):
    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params
        self._param_validation()

    @abstractmethod
    def _param_validation(self):
        pass

    @abstractmethod
    def eval_f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def eval_u(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def eval_Jf(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass

    def eval_Jf_numerical(
        self, x: np.ndarray, u: np.ndarray, dx: float = 0.01
    ) -> np.ndarray:
        f = self.eval_f(x, u)
        J = np.zeros((x.size, x.size))
        for i, _ in enumerate(x):
            delta_x = np.zeros(x.size)
            delta_x[i] = dx
            new_x = x + delta_x
            f_new = self.eval_f(new_x, u)
            delta_f = f_new - f
            J[:, i] = delta_f / dx
        return J

    def forward_euler(self, x_0: np.ndarray, ts: np.ndarray):
        X = 1.0 * np.zeros((x_0.size, ts.size))
        X[:, 0] = x_0
        dt = ts[1] - ts[0]
        for i, t in enumerate(ts[:-1]):
            u = self.eval_u(t)
            f = self.eval_f(X[:, i], u)
            X[:, i + 1] = X[:, i] + dt * f
        return X
