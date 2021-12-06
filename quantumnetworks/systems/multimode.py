"""
Driven Multi-Mode Mode Linear System with Beam-Splitter Couplings
"""
from typing import Dict, Any, List
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt

from quantumnetworks.systems.base import SystemSolver
from quantumnetworks.utils.visualization import draw_graph
from IPython.display import HTML, display
from matplotlib import animation


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
        # direct inputs of omegas, kappas, couplings (all in 2pi * GHz)
        omegas = self.params.get("omegas")
        if omegas is not None:
            self.params["omegas"] = np.array(omegas)
            self.params["num_modes"] = self.params["omegas"].size
        else:
            raise Exception("Please provide an `omegas` param")

        kappas = self.params.get("kappas")
        if kappas is not None:
            self.params["kappas"] = np.array(kappas)
            self.params["num_drives"] = np.count_nonzero(self.params["kappas"])
        else:
            raise Exception("Please provide a `kappas` param")

        gammas = self.params.get("gammas")
        if gammas is None:
            raise Exception("Please provide a `gammas` param")
        self.params["gammas"] = np.array(gammas)

        kerrs = self.params.get("kerrs")
        if kerrs is None:
            raise Exception("Please provide a `kerrs` param")
        self.params["kerrs"] = np.array(kerrs)

        couplings_raw = self.params.get("couplings")
        if couplings_raw is not None:
            self.params["couplings"] = np.array(couplings_raw)
            self.params["couplings_matrix"] = self.parse_couplings(
                self.params["couplings"]
            )
        else:
            raise Exception("Please provide a `couplings` param")

    # Load Data
    # =================================
    def load_data(self, folder: str) -> None:
        # omegas (in 2pi * GHz)
        omegas_raw_data = self.load_file(folder + os.sep + "omegas.txt")
        num_modes = omegas_raw_data.shape[0]
        self.params["num_modes"] = num_modes
        self.params["omegas"] = self.load_raw_dict_to_list(omegas_raw_data, num_modes)

        # kappas (in 2pi * GHz)
        kappas_raw_data = self.load_file(folder + os.sep + "kappas.txt")
        self.params["kappas"] = self.load_raw_dict_to_list(kappas_raw_data, num_modes)
        self.params["num_drives"] = np.count_nonzero(self.params["kappas"])

        # gammas (in 2pi * GHz)
        gammas_raw_data = self.load_file(folder + os.sep + "gammas.txt")
        self.params["gammas"] = self.load_raw_dict_to_list(gammas_raw_data, num_modes)

        # kerrs (in 2pi * GHz)
        kerrs_raw_data = self.load_file(folder + os.sep + "kerrs.txt")
        self.params["kerrs"] = self.load_raw_dict_to_list(kerrs_raw_data, num_modes)

        # coupling (in 2pi * GHz)
        couplings_raw_data = self.load_file(folder + os.sep + "couplings.txt")
        self.params["couplings_matrix"] = self.parse_couplings(couplings_raw_data)

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
            gammas = self.params["gammas"]
            couplings = self.params["couplings_matrix"]
            A = np.zeros((num_modes * 2, num_modes * 2))

            # omegas
            for i, omega in enumerate(omegas):
                A[2 * i, 2 * i + 1] += omega
                A[2 * i + 1, 2 * i] += -omega

            # kappas
            for i, kappa in enumerate(kappas):
                A[2 * i, 2 * i] += -kappa / 2
                A[2 * i + 1, 2 * i + 1] += -kappa / 2

            # gammas
            for i, gamma in enumerate(gammas):
                A[2 * i, 2 * i] += -gamma / 2
                A[2 * i + 1, 2 * i + 1] += -gamma / 2

            # couplings
            for i in range(couplings.shape[0]):
                for j in range(couplings.shape[1]):
                    if i != j:
                        g_ij = couplings[i, j]
                        A[2 * i, 2 * j + 1] += g_ij
                        A[2 * i + 1, 2 * j] += -g_ij

            self._A = A
        return self._A

    @property
    def B(self):
        if self._B is None:
            kappas = self.params["kappas"]
            num_modes = self.params["num_modes"]
            num_drives = self.params["num_drives"]

            B = np.zeros((num_modes * 2, num_drives * 2))
            j = 0
            for i, kappa in enumerate(kappas):
                if kappa != 0:
                    B[2 * i, 2 * j] = -np.sqrt(kappa)
                    B[2 * i + 1, 2 * j + 1] = -np.sqrt(kappa)
                    j += 1
            self._B = B
        return self._B

    def default_drive(self, _t):
        return 0

    # Nonlinear
    # =================================
    def f_nl(self, x: np.ndarray):
        """
        Nonlinear part of eq of motion
        """
        Ks = self.params["kerrs"]

        non_linearity = np.zeros_like(x)
        for mode in range(self.params["num_modes"]):
            qi = 0 + mode * 2
            pi = 1 + mode * 2
            q = x[qi]
            p = x[pi]
            K = Ks[mode]
            non_linearity[qi] = 2 * K * (q ** 2 + p ** 2) * p
            non_linearity[pi] = -2 * K * (q ** 2 + p ** 2) * q
        return non_linearity

    def Jf_nl(self, x: np.ndarray):
        """
        Jacobian of nonlinear part of eq of motion
        """
        Ks = self.params["kerrs"]

        nonlinear_Jf = np.zeros((x.size, x.size))

        for mode in range(self.params["num_modes"]):
            qi = 0 + mode * 2
            pi = 1 + mode * 2
            q = x[qi]
            p = x[pi]
            K = Ks[mode]
            nonlinear_Jf[qi][qi] = 4 * K * q * p
            nonlinear_Jf[qi][pi] = 2 * K * (q ** 2 + p ** 2) + 4 * K * p ** 2
            nonlinear_Jf[pi][qi] = -2 * K * (q ** 2 + p ** 2) - 4 * K * q ** 2
            nonlinear_Jf[pi][pi] = -4 * K * q * p
        return nonlinear_Jf

    # Eval
    # =================================

    def eval_f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        f = self.A.dot(x) + self.f_nl(x) + u
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
        drive_array = np.array(drive_vec)
        u = self.B.dot(drive_array)
        return u

    def eval_Jf(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.A + self.Jf_nl(x)

    # Plotting
    # =================================
    def draw_network(self, **kwargs):
        G = nx.Graph()
        for i in range(self.params["num_modes"]):
            G.add_node(i)

        couplings = self.params["couplings_matrix"]

        for i in range(self.params["num_modes"]):
            for j in range(0, i + 1):
                if couplings[i, j] > 0:
                    G.add_edge(i, j, weight=couplings[i, j])

        return draw_graph(G, **kwargs)

    def animate_networkx(
        self,
        xs,
        ts,
        ax=None,
        pos=None,
        num_frames=200,
        animation_time=5,
        save_animation=None,
        **kwargs,
    ):

        # https://stackoverflow.com/questions/43646550/how-to-use-an-update-function-to-animate-a-networkx-graph-in-matplotlib-2-0-0
        if len(xs) % 2 != 0:
            raise ValueError("Please enter state data with an even number of rows.")

        num_modes = len(xs) // 2
        num_points = len(ts)

        if ax is None:
            fig, ax = plt.subplots(1, figsize=(4, 4), squeeze=False)
            ax = ax[0][0]
        fig = ax.get_figure()

        # color scale
        max_amp = 0
        for i in range(num_modes):
            q = xs[2 * i, :]
            p = xs[2 * i + 1, :]
            max_amp_i = np.max(q ** 2 + p ** 2)
            max_amp = max_amp_i if max_amp_i > max_amp else max_amp

        if pos is None:
            _, _, pos = self.draw_network(ax=ax)

        def animate(j):
            indx = j * num_points // num_frames
            ax.clear()
            node_color = []
            for i in range(num_modes):
                q = xs[2 * i, indx]
                p = xs[2 * i + 1, indx]
                alpha = (q ** 2 + p ** 2) / max_amp
                node_color.append((1, 0, 0, alpha))
            self.draw_network(ax=ax, node_color=node_color, pos=pos, **kwargs)
            ax.set_title(f"t = {ts[indx]:.2f} ns")
            # ax.margins(0.05)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            fig.tight_layout()

        interval = animation_time * 1000 // num_frames
        anim = animation.FuncAnimation(
            fig, animate, frames=num_frames, interval=interval, repeat=True
        )
        if save_animation is not None:
            animation_title = (
                save_animation if isinstance(save_animation, str) else "animation.gif"
            )
            anim.save(animation_title, writer="pillow", fps=60)
        html = HTML(anim.to_jshtml())
        display(html)
        plt.close()
        return fig, ax

