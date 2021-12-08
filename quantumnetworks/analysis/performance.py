"""
Performance Analysis Tools
"""

import time

from quantumnetworks.systems.base import SystemSolver
from tqdm import tqdm
import time
import numpy as np


def time_func(func, *args, n=100, **kwargs):
    """
    Help method to time a function.
    Times the function n times and takes the average of the top 10% times.

    Args:
        *args: arguments needed for function evaluation
        n (int): number of times function is run
        **kwargs: keyword arguments needed for function evaluation

    Returns:
        average_time (float): average time of top 10% fastest times
    """
    time_vals = []
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
        """
        SolverOptimizer initialization.

        Args:
            system (SystemSolver): instance of SystemSolver subclass
            x_0 (np.ndarray): start state vector
            t_i (float): starting time
            t_f (float): ending time
        """
        self.system = system
        self.x_0 = x_0
        self.t_i = t_i
        self.t_f = t_f
        if not isinstance(system, SystemSolver):
            raise ValueError(
                "Please provide an instance of a subclass of SystemSolver for system."
            )

    def sweep_dt(self, dts=None, threshold=0.01, metric=None):
        """
        Sweep dt and measure change in solutions.

        Args:
            dts (optional[list]): optional list of dts to check
            threshold (float): a change in solutions below this threshold will result in convergence
            metric (optiona[function]): if provided, this will override the change metric

        Return:
            dchi_2s (np.ndarray): change metric
            dts_calc (np.ndarray): dts calculated
        """
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

    def sweep_dt_err(
        self,
        X_r,
        dts,
        *args,
        solver_method="forward_euler",
        time_iters=10,
        metric=None,
        **kwargs
    ):
        """
        Sweep dt and measure error with respect to a reference solution.

        Args:
            X_r (np.ndarray): reference solution
            dts (list): list of dt to sweep over
            *args: optional arguments to solver_method
            solver_method (str):
                solver_method to use to solve for system dynamics at different dt
                e.g. "forward_euler" or "trapezoidal"
            time_iters (int): number of iterations for average runtime calculation
            metric (optional): optional metric to override default
            **kwargs: optional keyword arguments to solver_method

        Returns:
            dchi_2s (np.ndarray): error values
        """
        total_t = self.t_f - self.t_i
        num_steps_gen = lambda dt: int(np.ceil(total_t / dt))

        dchi_2s = []
        runtimes = []

        metric = self.diff if metric is None else metric

        for dt in tqdm(dts):
            num_steps = num_steps_gen(dt)
            ts = np.linspace(self.t_i, self.t_f, num_steps + 1)

            t0 = time.time()
            for j in range(time_iters):
                X = getattr(self.system, solver_method)(self.x_0, ts, *args, **kwargs)
            t1 = time.time()
            runtimes.append((t1 - t0) / time_iters)

            if X.shape[1] < X_r.shape[1]:
                factor = int((X_r.shape[1] - 1) / (X.shape[1] - 1))
                diff = metric(X, X_r, factor)
            else:
                factor = int((X.shape[1] - 1) / (X_r.shape[1] - 1))
                diff = metric(X_r, X, factor)
            dchi_2s.append(diff)

        return np.array(dchi_2s), np.array(runtimes)

    def diff(self, a, b, factor):
        """
        Difference metric.

        Args:
            a (np.array): smaller array
            b (np.array): larger array by some factor
            factor (int): size(b)/size(a)

        Returns:
            ||a-b||_2/||a||
        """
        b = b[:, 0::factor]
        return np.linalg.norm(a - b, 2) / np.linalg.norm(a, 2)
