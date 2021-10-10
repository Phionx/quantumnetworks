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

    def sensitivity(self, param_name, param_indx=None, dp=1e-4):
        if not np.isscalar(self.system.params[param_name]):
            if param_indx is None:
                raise ValueError("Please provide a param_indx.")
        else:
            if param_indx is not None:
                raise ValueError("Unnecessary param_indx provided.")

        # old solutions
        xs = self.system.forward_euler(self.x0, self.ts)
        q = self.calculate(xs)

        # perturb params
        val = None
        if param_indx is None:
            val = self.system.params[param_name]
            self.system.params[param_name] *= 1 + dp
        else:
            val = self.system.params[param_name][param_indx]
            self.system.params[param_name][param_indx] *= 1 + dp

        # new solutions
        xs_new = self.system.forward_euler(self.x0, self.ts)
        q_new = self.calculate(xs_new)

        # finite difference
        dqdp = (q_new - q) / dp

        # reset params
        if param_indx is None:
            self.system.params[param_name] = val
        else:
            self.system.params[param_name][param_indx] = val

        return dqdp

    @abstractmethod
    def calculate(self, xs: np.ndarrary) -> np.ndarray:
        """
        Calculate quantity of interest.
        
        E.g. Average value might be np.average(xs, axis=0)
        """
        pass

