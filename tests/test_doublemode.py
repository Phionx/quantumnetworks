"""
Unittests

Run using: 
python -m unittest tests/test_doublemode.py
"""
import os
import sys
import unittest

sys.path.insert(0, ".." + os.sep)

from quantumnetworks import DoubleModeSystem
from scipy.integrate import odeint
import numpy as np


class DoubleModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {
            "omega_a": 2 * np.pi * 1,
            "omega_b": 2 * np.pi * 2,
            "kappa_a": 2 * np.pi * 0.001,
            "kappa_b": 2 * np.pi * 0.005,
            "kerr_a": 2 * np.pi * 0.01,
            "kerr_b": 2 * np.pi * 0.01,
            "gamma_a": 2 * np.pi * 0.002,
            "gamma_b": 2 * np.pi * 0.002,
            "g_ab": 2 * np.pi * 0.002,
        }
        self.drive_a = lambda t: np.exp(1.0j * (self.params["omega_a"] * t))
        self.drive_b = lambda t: np.exp(1.0j * (self.params["omega_b"] * t))

    def test_forward_euler_no_A_in(self):
        x_0 = np.array([1, 0, 0, 1])
        ts = np.linspace(0, 1, 100001)

        # solve using SystemSolver
        system = DoubleModeSystem(params=self.params, A_in=None, B_in=None)
        X = system.forward_euler(x_0, ts)

        # solve using scipy.integrate.odeint
        func = lambda y, t: system.eval_f(y, system.eval_u(t))
        sol = odeint(func, x_0, ts)
        self.assertTrue(np.allclose(X.T, sol, atol=0.002))

    def test_forward_euler_default_A_in(self):
        x_0 = np.array([1, 0, 0, 1])
        ts = np.linspace(0, 1, 100001)

        # solve using SystemSolver
        system = DoubleModeSystem(
            params=self.params, A_in=self.drive_a, B_in=self.drive_b
        )
        X = system.forward_euler(x_0, ts)

        # solve using scipy.integrate.odeint
        func = lambda y, t: system.eval_f(y, system.eval_u(t))
        sol = odeint(func, x_0, ts)
        self.assertTrue(np.allclose(X.T, sol, atol=0.002))

    def test_analytic_vs_numerical_Jf(self):
        x_0 = np.array([1, 0, 0, 1])

        # test analytic vs numerical Jf
        system = DoubleModeSystem(params=self.params)
        u = system.eval_u(0)
        Jf_analytic = system.eval_Jf(x_0, u)
        Jf_numeric = system.eval_Jf_numerical(x_0, u)
        self.assertTrue(np.allclose(Jf_analytic, Jf_numeric))

    def test_linearization(self):
        sys = DoubleModeSystem(params=self.params, A_in=None, B_in=None)
        x_0 = np.array([1, 0, 0, 1])
        n = 100000
        ts = np.linspace(0, 1, n + 1)
        X = sys.forward_euler(x_0, ts)
        X_linear = sys.forward_euler_linear(x_0, ts, x_0, 0)

        # take beginning of sequences
        X_linear_i = X_linear[:, : n // 10]
        X_i = X[:, : n // 10]

        # filter to prevent divide by 0 errors
        X_linear_i = X_linear_i[X_i != 0]
        X_i = X_i[X_i != 0]

        max_perc_diff = np.max(np.abs((X_i - X_linear_i) / X_i))
        self.assertTrue(max_perc_diff < 0.03)  # within 3%


#%%
if __name__ == "__main__":
    unittest.main()
