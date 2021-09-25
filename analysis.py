#%%
from typing import Dict
from abc import abstractmethod, ABCMeta
import numpy as np
import matplotlib.pyplot as plt


class SystemSolver(metaclass=ABCMeta):
    def __init__(self, params: Dict[str, float]) -> None:
        self.params = params
        self._param_validation()

    @abstractmethod
    def _param_validation(self):
        pass

    @abstractmethod
    def eval_f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def eval_u(self, t: float):
        pass

    def forward_euler(self, x_0: np.ndarray, ts: np.ndarray):
        X = 1.0 * np.zeros((x_0.size, ts.size))
        X[:, 0] = x_0
        dt = ts[1] - ts[0]
        for i, t in enumerate(ts[:-1]):
            u = self.eval_u(t)
            f = self.eval_f(X[:, i], u)
            X[:, i + 1] = X[:, i] + dt * f
        return X


class ExampleSystem(SystemSolver):
    def _param_validation(self):
        if "w" not in self.params:
            self.params["w"] = 2

    def eval_f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.params["w"] * x + u

    def eval_u(self, t: float):
        return t


#%%
sys = ExampleSystem(params={"w": 10})
x_0 = np.array([0])
ts = np.linspace(0, 1, 101)
X = sys.forward_euler(x_0, ts)
plt.plot(ts, X[0, :])

# %%
