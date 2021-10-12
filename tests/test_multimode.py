"""
Unittests

Run using: 
python -m unittest tests/test_multimode.py
"""
import os
import sys
import unittest

sys.path.insert(0, ".." + os.sep)

from quantumnetworks import SingleModeSystem, DoubleModeSystem, MultiModeSystem
from scipy.integrate import odeint
import numpy as np


class MultiModeTest(unittest.TestCase):
    def test_forward_euler_default_A_in(self):
        omegas = [2 * np.pi * 1, 2 * np.pi * 2]
        kappas = [2 * np.pi * 0.001, 2 * np.pi * 0.005]
        couplings = [[0, 1, 2 * np.pi * 0.002]]
        gammas = [2 * np.pi * 0.002, 2 * np.pi * 0.002]
        kerrs = [2 * np.pi * 0.01, 2 * np.pi * 0.01]
        system = MultiModeSystem(
            params={
                "omegas": omegas,
                "kappas": kappas,
                "gammas": gammas,
                "kerrs": kerrs,
                "couplings": couplings,
            }
        )
        x_0 = np.array([1, 0, 0, 1])
        ts = np.linspace(0, 1, 100001)

        X = system.forward_euler(x_0, ts)

        # solve using scipy.integrate.odeint
        func = lambda y, t: system.eval_f(y, system.eval_u(t))
        sol = odeint(func, x_0, ts)
        self.assertTrue(np.allclose(X.T, sol, atol=0.002))

    def test_analytic_vs_numerical_Jf(self):
        omegas = [2 * np.pi * 1, 2 * np.pi * 2]
        kappas = [2 * np.pi * 0.001, 2 * np.pi * 0.005]
        gammas = [2 * np.pi * 0.002, 2 * np.pi * 0.002]
        kerrs = [2 * np.pi * 0.01, 2 * np.pi * 0.01]
        couplings = [[0, 1, 2 * np.pi * 0.002]]
        system = MultiModeSystem(
            params={
                "omegas": omegas,
                "kappas": kappas,
                "gammas": gammas,
                "kerrs": kerrs,
                "couplings": couplings,
            }
        )
        x_0 = np.array([1, 0, 0, 1])

        u = system.eval_u(0)
        Jf_analytic = system.eval_Jf(x_0, u)
        Jf_numeric = system.eval_Jf_numerical(x_0, u)
        self.assertTrue(np.allclose(Jf_analytic, Jf_numeric))

    def test_against_double_mode(self):
        omegas = [2 * np.pi * 1, 2 * np.pi * 2]
        kappas = [2 * np.pi * 0.001, 2 * np.pi * 0.005]
        gammas = [2 * np.pi * 0.002, 2 * np.pi * 0.002]
        kerrs = [2 * np.pi * 0.01, 2 * np.pi * 0.01]
        couplings = [[0, 1, 2 * np.pi * 0.002]]

        system = MultiModeSystem(
            params={
                "omegas": omegas,
                "kappas": kappas,
                "gammas": gammas,
                "kerrs": kerrs,
                "couplings": couplings,
            }
        )
        A_in = lambda t: 0
        B_in = lambda t: 0
        system_double = DoubleModeSystem(
            params={
                "omega_a": 2 * np.pi * 1,
                "omega_b": 2 * np.pi * 2,
                "kappa_a": 2 * np.pi * 0.001,
                "kappa_b": 2 * np.pi * 0.005,
                "kerr_a": 2 * np.pi * 0.01,
                "kerr_b": 2 * np.pi * 0.01,
                "gamma_a": 2 * np.pi * 0.002,
                "gamma_b": 2 * np.pi * 0.002,
                "g_ab": 2 * np.pi * 0.002,
            },
            A_in=A_in,
            B_in=B_in,
        )

        self.assertTrue(np.array_equal(system.A, system_double.A))
        self.assertTrue(np.array_equal(system.B, system_double.B))

    def test_against_single_mode(self):
        omegas = [2 * np.pi * 1]
        kappas = [2 * np.pi * 0.001]
        gammas = [2 * np.pi * 0.002]
        kerrs = [2 * np.pi * 0.01]
        couplings = []

        system = MultiModeSystem(
            params={
                "omegas": omegas,
                "kappas": kappas,
                "gammas": gammas,
                "kerrs": kerrs,
                "couplings": couplings,
            }
        )
        A_in = lambda t: 0
        system_double = SingleModeSystem(
            params={
                "omega_a": 2 * np.pi * 1,
                "kappa_a": 2 * np.pi * 0.001,
                "gamma_a": 2 * np.pi * 0.002,
                "kerr_a": 2 * np.pi * 0.01,
            },
            A_in=A_in,
        )

        self.assertTrue(np.array_equal(system.A, system_double.A))
        self.assertTrue(np.array_equal(system.B, system_double.B))

    def test_linearization(self):
        omegas = [2 * np.pi * 1, 2 * np.pi * 2, 2 * np.pi * 1]
        kappas = [2 * np.pi * 0.001, 2 * np.pi * 0.005, 2 * np.pi * 0.001]
        gammas = [2 * np.pi * 0.002, 2 * np.pi * 0.002, 2 * np.pi * 0.002]
        kerrs = [2 * np.pi * 0.001, 2 * np.pi * 0.001, 2 * np.pi * 0.001]
        couplings = [[0, 1, 2 * np.pi * 0.002], [1, 2, 2 * np.pi * 0.002]]
        sys = MultiModeSystem(
            params={
                "omegas": omegas,
                "kappas": kappas,
                "gammas": gammas,
                "kerrs": kerrs,
                "couplings": couplings,
            }
        )

        x_0 = np.array([1, 0, 0, 1, 1, 0])
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
        self.assertTrue(max_perc_diff < 0.01)  # within 1%


#%%
if __name__ == "__main__":
    unittest.main()
