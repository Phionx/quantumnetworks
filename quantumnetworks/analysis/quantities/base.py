"""
System Quantity and associated Analysis
"""

from abc import abstractmethod, ABCMeta
from quantumnetworks.systems.base import SystemSolver
import numpy as np


class SystemQuantity(metaclass=ABCMeta):
    def __init__(self, system: SystemSolver, x0: np.ndarray, ts: np.ndarray) -> None:
        self.system = system
        self.x0 = x0
        self.ts = ts
        self._xs = None
        self._q = None

    @property
    def xs(self):
        if self._xs is None:
            self._xs = self.system.forward_euler(self.x0, self.ts)
        return self._xs

    @property
    def q(self):
        if self._q is None:
            self._q = self.calculate(self.xs)
        return self._q

    def sensitivity(self, param_name, param_indx=None, dp=1e-2):
        if not np.isscalar(self.system.params[param_name]):
            if param_indx is None:
                raise ValueError("Please provide a param_indx.")
        else:
            if param_indx is not None:
                raise ValueError("Unnecessary param_indx provided.")

        copy_system = self.system.copy()

        # perturb params
        val = None
        if param_indx is None:
            val = copy_system.params[param_name]
            copy_system.params[param_name] = val * (1.0 + dp)
        else:
            val = copy_system.params[param_name][param_indx]
            copy_system.params[param_name][param_indx] = val * (1.0 + dp)

        # new solutions
        xs_new = copy_system.forward_euler(self.x0, self.ts)
        q_new = self.calculate(xs_new)

        # finite difference
        dqdp_i = (q_new - self.q) / (val * dp)

        return dqdp_i

    @abstractmethod
    def calculate(self, xs: np.ndarray) -> np.ndarray:
        """
        Calculate quantity of interest.
        
        E.g. Average value might be np.average(xs, axis=1)
        """
        pass

