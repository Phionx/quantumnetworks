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

    def sweep_dt(self, dts=None, threshold=0.01, metric=None):
        total_t = self.t_f - self.t_i
        dt_gen = lambda n: total_t / n
        num_steps_gen = lambda dt: int(np.ceil(total_t / dt))

        dts_calc = []
        dchi_2s = []
        X_prev = None
        converged = False

        dt_curr = dt_gen(1000) if dts is None else dts[0]
        factor = 2
        i = 0

        metric = self.diff if metric is None else metric
        while not converged:
            num_steps = num_steps_gen(dt_curr)
            print(num_steps)
            ts = np.linspace(self.t_i, self.t_f, num_steps + 1)
            X = self.system.forward_euler(self.x_0, ts)
            if X_prev is not None:
                dts_calc.append(dt_curr)
                diff = metric(X_prev, X, factor)
                dchi_2s.append(diff)
                print(diff)
                if dchi_2s[-1] < threshold or (dts is not None and i >= len(dts) - 1):
                    # system has converged
                    converged = True
            X_prev = X

            # Next Step
            if dts is None:
                dt_curr = dt_gen(num_steps * factor)
            else:
                i += 1
                dt_curr = dts[i]
        return np.array(dchi_2s), np.array(dts_calc)

    def sweep_dt_err(self, X_r, dts, metric=None):
        total_t = self.t_f - self.t_i
        num_steps_gen = lambda dt: int(np.ceil(total_t / dt))

        dchi_2s = []

        metric = self.diff if metric is None else metric

        for dt in tqdm(dts):
            num_steps = num_steps_gen(dt)
            ts = np.linspace(self.t_i, self.t_f, num_steps + 1)
            X = self.system.forward_euler(self.x_0, ts)
            if X.shape[1] < X_r.shape[1]:
                factor = int((X_r.shape[1] - 1) / (X.shape[1] - 1))
                diff = metric(X, X_r, factor)
            else:
                factor = int((X.shape[1] - 1) / (X_r.shape[1] - 1))
                diff = metric(X_r, X, factor)
            dchi_2s.append(diff)
        return np.array(dchi_2s)

    def diff(self, a, b, factor):
        b = b[:, 0::factor]
        return np.linalg.norm(a - b, 2) / np.linalg.norm(a, 2)
