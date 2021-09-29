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
        omegas = [1, 2]
        kappas = [0.001, 0.005]
        couplings = [[0, 1, 0.002]]
        gammas = [0.002, 0.002]
        system = MultiModeSystem(
            params={
                "omegas": omegas,
                "kappas": kappas,
                "gammas": gammas,
                "couplings": couplings,
            }
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
        gammas = [0.002, 0.002]
        couplings = [[0, 1, 0.002]]
        system = MultiModeSystem(
            params={
                "omegas": omegas,
                "kappas": kappas,
                "gammas": gammas,
                "couplings": couplings,
            }
        )
        x_0 = np.array([1, 0, 0, 1])

        u = system.eval_u(0)
        Jf_analytic = system.eval_Jf(x_0, u)
        Jf_numeric = system.eval_Jf_numerical(x_0, u)
        self.assertTrue(np.allclose(Jf_analytic, Jf_numeric))

    def test_against_double_mode(self):
        omegas = [1, 2]
        kappas = [0.001, 0.005]
        gammas = [0.002, 0.002]
        couplings = [[0, 1, 0.002]]

        system = MultiModeSystem(
            params={
                "omegas": omegas,
                "kappas": kappas,
                "gammas": gammas,
                "couplings": couplings,
            }
        )
        A_in = lambda t: 0
        B_in = lambda t: 0
        system_double = DoubleModeSystem(
            params={
                "omega_a": 1,
                "omega_b": 2,
                "kappa_a": 0.001,
                "kappa_b": 0.005,
                "gamma_a": 0.002,
                "gamma_b": 0.002,
                "g_ab": 0.002,
            },
            A_in=A_in,
            B_in=B_in,
        )

        self.assertTrue(np.array_equal(system.A, system_double.A))
        self.assertTrue(np.array_equal(system.B, system_double.B))

    def test_against_single_mode(self):
        omegas = [1]
        kappas = [0.001]
        gammas = [0.002]
        couplings = []

        system = MultiModeSystem(
            params={
                "omegas": omegas,
                "kappas": kappas,
                "gammas": gammas,
                "couplings": couplings,
            }
        )
        A_in = lambda t: 0
        system_double = SingleModeSystem(
            params={"omega_a": 1, "kappa_a": 0.001, "gamma_a": 0.002}, A_in=A_in,
        )

        self.assertTrue(np.array_equal(system.A, system_double.A))
        self.assertTrue(np.array_equal(system.B, system_double.B))


#%%
if __name__ == "__main__":
    unittest.main()
