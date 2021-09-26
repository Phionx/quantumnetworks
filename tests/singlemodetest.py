"""
Unittests

Run using: python -m unittest tests/singlemodetest.py
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
        self.params = {"omega_a": 1, "kappa_a": 0.001}

    def test_forward_euler_no_A_in(self):
        A_in = lambda t: 0
        x_0 = np.array([1, 0])
        ts = np.linspace(0, 10, 100001)

        # solve using SystemSolver
        sys = SingleModeSystem(params=self.params, A_in=A_in)
        X = sys.forward_euler(x_0, ts)

        # solve using scipy.integrate.odeint
        func = lambda y, t: sys.eval_f(y, sys.eval_u(t))
        sol = odeint(func, x_0, ts)
        self.assertTrue(np.allclose(X.T, sol, atol=0.002))

    def test_forward_euler_default_A_in(self):
        A_in = None
        x_0 = np.array([1, 0])
        ts = np.linspace(0, 10, 100001)

        # solve using SystemSolver
        sys = SingleModeSystem(params=self.params, A_in=A_in)
        X = sys.forward_euler(x_0, ts)

        # solve using scipy.integrate.odeint
        func = lambda y, t: sys.eval_f(y, sys.eval_u(t))
        sol = odeint(func, x_0, ts)
        self.assertTrue(np.allclose(X.T, sol, atol=0.002))

    def test_analytic_vs_numerical_Jf(self):
        A_in = None
        x_0 = np.array([1, 0])

        # test analytic vs numerical Jf
        sys = SingleModeSystem(params=self.params, A_in=A_in)
        u = sys.eval_u(0)
        Jf_analytic = sys.eval_Jf(x_0, u)
        Jf_numeric = sys.eval_Jf_numerical(x_0, u)
        self.assertTrue(np.allclose(Jf_analytic, Jf_numeric))


#%%
if __name__ == "__main__":
    unittest.main()
