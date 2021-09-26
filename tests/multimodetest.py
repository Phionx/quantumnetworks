"""
Unittests

Run using: python -m unittest tests/multimodetest.py
"""
import os
import sys
import unittest

sys.path.insert(0, ".." + os.sep)

from quantumnetworks import MultiModeSystem
from scipy.integrate import odeint
import numpy as np


class MultiModeTest(unittest.TestCase):
    def test_forward_euler_default_A_in(self):
        omegas = [1, 2]
        kappas = [0.001, 0.005]
        couplings = [[0, 1, 0.002]]
        system = MultiModeSystem(
            params={"omegas": omegas, "kappas": kappas, "couplings": couplings}
        )
        x_0 = np.array([1, 0, 0, 1])
        ts = np.linspace(0, 10, 100001)

        X = system.forward_euler(x_0, ts)

        # solve using scipy.integrate.odeint
        func = lambda y, t: system.eval_f(y, system.eval_u(t))
        sol = odeint(func, x_0, ts)
        self.assertTrue(np.allclose(X.T, sol, atol=0.002))

    def test_analytic_vs_numerical_Jf(self):
        omegas = [1, 2]
        kappas = [0.001, 0.005]
        couplings = [[0, 1, 0.002]]
        system = MultiModeSystem(
            params={"omegas": omegas, "kappas": kappas, "couplings": couplings}
        )
        x_0 = np.array([1, 0, 0, 1])

        u = system.eval_u(0)
        Jf_analytic = system.eval_Jf(x_0, u)
        Jf_numeric = system.eval_Jf_numerical(x_0, u)
        self.assertTrue(np.allclose(Jf_analytic, Jf_numeric))


#%%
if __name__ == "__main__":
    unittest.main()
