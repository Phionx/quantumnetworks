"""
Driven Signle Mode System
"""
from typing import Dict, Any, List
import numpy as np
import os

from quantumnetworks.analysis import SystemSolver


class MultiModeSystem(SystemSolver):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self._A = None
        self._B = None
        self.load_data(str(self.params["dir"]))

    def _param_validation(self):
        if "dir" not in self.params:
            raise Exception(
                "Please provide a `dir` param in which to find system paramters."
            )
        if "drives" not in self.params:
            self.params["drives"] = {}

    # Load Data
    # =================================
    def load_data(self, dir: str) -> None:
        # omegas
        omegas_raw_data = self.load_file(dir + os.sep + "omegas.txt")
        omegas = np.zeros(omegas_raw_data.shape[0])
        for row in omegas_raw_data:
            omegas[int(row[0])] = row[1]
        self.params["omegas"] = omegas
        num_modes = omegas.size
        self.params["num_modes"] = num_modes

        # kappas
        kappas_raw_data = self.load_file(dir + os.sep + "kappas.txt")
        kappas = np.zeros(num_modes)
        for row in kappas_raw_data:
            i = int(row[0])
            kappas[i] = row[1]
        self.params["kappas"] = kappas
        self.params["num_drives"] = np.count_nonzero(kappas)

        # coupling
        couplings_raw_data = self.load_file(dir + os.sep + "couplings.txt")
        couplings = np.zeros((num_modes, num_modes))
        for row in couplings_raw_data:
            i = int(row[0])
            j = int(row[1])
            couplings[i, j] = row[2]
            couplings[j, i] = row[2]
        self.params["couplings"] = couplings

    def load_file(self, filename):
        a = np.loadtxt(filename, delimiter=",")
        if len(a.shape) == 1:
            return a.reshape((-1, a.size))
        return a

    # Known System Parameters and Load
    # =================================
    @property
    def A(self):
        if self._A is None:
            num_modes = self.params["num_modes"]
            omegas = self.params["omegas"]
            kappas = self.params["kappas"]
            couplings = self.params["couplings"]
            A = np.zeros((num_modes * 2, num_modes * 2))

            # omegas
            for i, omega in enumerate(omegas):
                A[2 * i, 2 * i + 1] = omega
                A[2 * i + 1, 2 * i] = -omega

            # kappas
            for i, kappa in enumerate(kappas):
                A[2 * i, 2 * i] = kappa / 2
                A[2 * i + 1, 2 * i + 1] = kappa / 2

            # couplings
            for i in range(couplings.shape[0]):
                for j in range(couplings.shape[1]):
                    if i != j:
                        g_ij = couplings[i, j]
                        A[2 * i, 2 * j + 1] = g_ij
                        A[2 * i + 1, 2 * j] = -g_ij
                        A[2 * j, 2 * i + 1] = g_ij
                        A[2 * j + 1, 2 * i] = -g_ij

            self._A = A
        return self._A

    @property
    def B(self):
        if self._B is None:
            kappas = self.params["kappas"]
            num_modes = self.params["num_modes"]
            num_drives = self.params["num_drives"]

            B = np.zeros((num_modes * 2, num_drives * 2))
            for i, kappa in enumerate(kappas):
                if kappa != 0:
                    B[2 * i, 2 * i] = np.sqrt(kappa)
                    B[2 * i + 1, 2 * i + 1] = np.sqrt(kappa)

            self._B = B
        return self._B

    def default_drive(self, _t):
        return 0

    # Eval
    # =================================

    def eval_f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        f = self.A.dot(x) + u
        return f

    def eval_u(self, t: float):
        kappas = self.params["kappas"]
        drives = self.params["drives"]

        drive_vec: List[float] = []
        for i, kappa in enumerate(kappas):
            if kappa != 0:
                if i in drives:
                    drive = drives[i]
                else:
                    drive = self.default_drive
                drive_vec += [np.real(drive(t)), np.imag(drive(t))]
        drive_vec = np.array(drive_vec)
        u = self.B.dot(drive_vec)
        return u

    def eval_Jf(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A
