"""
Base Analysis
"""
from typing import Dict, Any
from abc import abstractmethod, ABCMeta
import numpy as np


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

    def eval_f_linear(
        self, x: np.ndarray, u: np.ndarray, x0: np.ndarray, u0: np.ndarray
    ) -> np.ndarray:
        # df/du = 1.0 in our case
        return self.eval_f(x0, u0) + self.eval_Jf(x0, u0).dot(x - x0) + 1.0 * (u - u0)

    def eval_Jf_numerical(
        self, x: np.ndarray, u: np.ndarray, dx: float = 1e-7
    ) -> np.ndarray:
        x = x.astype(float)
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

    def forward_euler(self, x_start: np.ndarray, ts: np.ndarray):
        x_start = x_start.astype(float)
        X = 1.0 * np.zeros((x_start.size, ts.size))
        X[:, 0] = x_start
        dt = ts[1] - ts[0]
        for i, t in enumerate(ts[:-1]):
            u = self.eval_u(t)
            f = self.eval_f(X[:, i], u)
            X[:, i + 1] = X[:, i] + dt * f
        return X

    def forward_euler_linear(
        self, x_start: np.ndarray, ts: np.ndarray, x0: np.ndarray, u0: np.ndarray
    ):
        X = 1.0 * np.zeros((x_start.size, ts.size))
        X[:, 0] = x_start
        dt = ts[1] - ts[0]
        for i, t in enumerate(ts[:-1]):
            u = self.eval_u(t)
            f = self.eval_f_linear(X[:, i], u, x0, u0)
            X[:, i + 1] = X[:, i] + dt * f
        return X

    def copy(self):
        """
        Just copies params into another instance of the system class. 
        Not a full copy (so that stored analysis can be reset).
        """
        cls = self.__class__
        return cls(self.params.copy())
