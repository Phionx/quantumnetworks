"""
Driven Multi-Mode Mode Linear System with Beam-Splitter Couplings
"""
from typing import Dict, Any, List
import numpy as np
import os

from quantumnetworks.systems.base import SystemSolver


class MultiModeSystem(SystemSolver):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self._A = None
        self._B = None
        if self.params["dir"] is not None:
            self.load_data(str(self.params["dir"]))

    def _param_validation(self):
        self.params["dir"] = self.params.get("dir")
        self.params["drives"] = self.params.get("drives", {})
        if self.params["dir"] is not None:
            return

        # If data isn't provided via txt, then we should expect
        # direct inputs of omeags, kappas, couplings
        self.params["omegas"] = np.array(self.params.get("omegas"))
        if self.params["omegas"] is not None:
            self.params["num_modes"] = self.params["omegas"].size
        else:
            raise Exception("Please provide an `omegas` param")

        self.params["kappas"] = np.array(self.params.get("kappas"))
        if self.params["kappas"] is not None:
            self.params["num_drives"] = np.count_nonzero(self.params["kappas"])
        else:
            raise Exception("Please provide a `kappas` param")

        self.params["gammas"] = np.array(self.params.get("gammas"))
        if self.params["gammas"] is None:
            raise Exception("Please provide a `gammas` param")

        couplings_raw = self.params.get("couplings")
        if couplings_raw is not None:
            self.params["couplings"] = self.parse_couplings(np.array(couplings_raw))
        else:
            raise Exception("Please provide a `couplings` param")

    # Load Data
    # =================================
    def load_data(self, dir: str) -> None:
        # omegas
        omegas_raw_data = self.load_file(dir + os.sep + "omegas.txt")
        num_modes = omegas_raw_data.shape[0]
        self.params["num_modes"] = num_modes
        self.params["omegas"] = self.load_raw_dict_to_list(omegas_raw_data, num_modes)

        # kappas
        kappas_raw_data = self.load_file(dir + os.sep + "kappas.txt")
        self.params["kappas"] = self.load_raw_dict_to_list(kappas_raw_data, num_modes)
        self.params["num_drives"] = np.count_nonzero(self.params["kappas"])

        # gammas
        gammas_raw_data = self.load_file(dir + os.sep + "gammas.txt")
        self.params["gammas"] = self.load_raw_dict_to_list(gammas_raw_data, num_modes)

        # coupling
        couplings_raw_data = self.load_file(dir + os.sep + "couplings.txt")
        self.params["couplings"] = self.parse_couplings(couplings_raw_data)
        couplings = np.zeros((num_modes, num_modes))
        for row in couplings_raw_data:
            i = int(row[0])
            j = int(row[1])
            couplings[i, j] = row[2]
            couplings[j, i] = row[2]
        self.params["couplings"] = couplings

    def parse_couplings(self, couplings_raw_data):
        num_modes = self.params["num_modes"]
        couplings = np.zeros((num_modes, num_modes))
        for row in couplings_raw_data:
            i = int(row[0])
            j = int(row[1])
            couplings[i, j] = row[2]
            couplings[j, i] = row[2]
        return couplings

    def load_file(self, filename):
        a = np.loadtxt(filename, delimiter=",")
        if len(a.shape) == 1:
            return a.reshape((-1, a.size))
        return a

    def load_raw_dict_to_list(self, raw_data, length):
        data = np.zeros(length)
        for row in raw_data:
            i = int(row[0])
            data[i] = row[1]
        return data

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
                A[2 * i, 2 * i] = -kappa / 2
                A[2 * i + 1, 2 * i + 1] = -kappa / 2

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
                    B[2 * i, 2 * i] = -np.sqrt(kappa)
                    B[2 * i + 1, 2 * i + 1] = -np.sqrt(kappa)

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
