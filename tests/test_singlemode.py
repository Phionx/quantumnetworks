"""
Unittests

Run using: 
python -m unittest tests/test_singlemode.py
"""
import os
import sys
import unittest

sys.path.insert(0, ".." + os.sep)

from quantumnetworks import SingleModeSystem
from scipy.integrate import odeint
import numpy as np


class SingleModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {
            "omega_a": 2 * np.pi * 1,
            "kappa_a": 2 * np.pi * 0.001,
            "gamma_a": 2 * np.pi * 0.002,
            "kerr_a": 2 * np.pi * 0.01,
        }

    def test_forward_euler_no_A_in(self):
        A_in = None
        x_0 = np.array([1, 0])
        ts = np.linspace(0, 1, 100001)

        # solve using SystemSolver
        system = SingleModeSystem(params=self.params, A_in=A_in)
        X = system.forward_euler(x_0, ts)

        # solve using scipy.integrate.odeint
        func = lambda y, t: system.eval_f(y, system.eval_u(t))
        sol = odeint(func, x_0, ts)
        self.assertTrue(np.allclose(X.T, sol, atol=0.002))

    def test_forward_euler_with_drive(self):
        A_in = lambda t: np.exp(1.0j * (self.params["omega_a"] * t))
        x_0 = np.array([1, 0])
        ts = np.linspace(0, 1, 100001)

        # solve using SystemSolver
        system = SingleModeSystem(params=self.params, A_in=A_in)
        X = system.forward_euler(x_0, ts)

        # solve using scipy.integrate.odeint
        func = lambda y, t: system.eval_f(y, system.eval_u(t))
        sol = odeint(func, x_0, ts)
        self.assertTrue(np.allclose(X.T, sol, atol=0.002))

    def test_analytic_vs_numerical_Jf(self):
        A_in = lambda t: np.exp(1.0j * (self.params["omega_a"] * t))
        x_0 = np.array([1, 0])

        # test analytic vs numerical Jf
        system = SingleModeSystem(params=self.params, A_in=A_in)
        u = system.eval_u(0)
        Jf_analytic = system.eval_Jf(x_0, u)
        Jf_numeric = system.eval_Jf_numerical(x_0, u)
        self.assertTrue(np.allclose(Jf_analytic, Jf_numeric))

    def test_linearization(self):
        A_in = None
        sys = SingleModeSystem(params=self.params, A_in=A_in)
        x_0 = np.array([1, 0])
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
