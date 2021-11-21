"""
Performance Analysis Tools
"""

import time

from quantumnetworks.systems.base import SystemSolver
from tqdm import tqdm
import numpy as np


def time_func(func, *args, **kwargs):
    time_vals = []
    n = kwargs.pop("n", 100)
    for i in tqdm(range(n)):
        start = time.time()
        func(*args, **kwargs)
        time_vals.append(time.time() - start)
    top_times = sorted(time_vals)[: max(n // 10, 4)]
    return sum(top_times) / len(top_times)


class SolverOptimizer:
    """
    Find finite ODE solver parameters for system
    """

    def __init__(self, system, x_0, t_i, t_f) -> None:
        self.system = system
        self.x_0 = x_0
        self.t_i = t_i
        self.t_f = t_f
        if not isinstance(system, SystemSolver):
            raise ValueError(
                "Please provide an instance of a subclass of SystemSolver for system."
            )

    def optimal_dt(self, threshold=0.0001):
        total_t = self.t_f - self.t_i
        dt_gen = lambda n: total_t / n

        dts = []
        dchi_2s = []
        X_prev = None
        num_steps = 1000
        converged = False
        factor = 2
        while not converged:
            ts = np.linspace(self.t_i, self.t_f, num_steps + 1)
            X = self.system.forward_euler(self.x_0, ts)
            if X_prev is not None:
                dts.append(dt_gen(num_steps))
                dchi_2s.append(self.max_diff(X_prev, X[:, 0::factor]))
                if dchi_2s[-1] < threshold:
                    # system has converged
                    converged = True
            X_prev = X
            num_steps *= factor
        return np.array(dchi_2s), np.array(dts)

    def max_diff(self, a, b):
        return np.max(np.abs(a - b))
