"""
Error Analysis
"""
import numpy as np

from quantumnetworks.systems.base import SystemSolver
from typing import Any, Dict
from abc import abstractmethod, ABCMeta

from quantumnetworks.systems.multimode import MultiModeSystem
from quantumnetworks.utils.visualization import plot_full_evolution
from tqdm import tqdm


class SystemError(metaclass=ABCMeta):
    def __init__(self, system: SystemSolver, params_error: Dict[str, Any]) -> None:
        """
        Initialize SystemError tool.

        Args:
            system (SystemSolver): instance of SystemSolver subclass
            params_error (dict):
                key (str): parameter name
                val (Any): any parameter uncertainty value
        """
        self.system = system
        self.params_error = params_error
        self.solves = None
        self.reset_solves()

    def reset_solves(self):
        """
        Reset self.solves.
        """
        self.solves = {"original": None, "with_error": [], "std": None}

    @abstractmethod
    def calculate_error(self, method: str, *args, parse_output=lambda X: X, **kwargs):
        """
        Method to sample system solver with a parameters sampled form a distributino 
        determined by parameter uncertainty. Stores runs in self.solves.

        This is system specific.
        
        Args:
            method (str): solver method of SystemSolver e.g. "trapezoidal" 
            *args: arguments provided to solver method, e.g. ts, x0
            parse_output (function pointer): 
                how to parse output after solving, 
                e.g. with dynamic trapezoidal parse_output = lambda X: X[0]
            **kwargs: keyword arguments provided to solver method 
        
        """
        pass

    def run(
        self, method: str, *args, parse_output=lambda X: X, num_samples=11, **kwargs
    ):
        """
        Wrapper method on self.calculate_error to run sample-based error analysis. 
        Stores results in self.solves.

        General idea:
            Let's say we measure parameter a ±  δa, b ±  δb, and c ±  δc. 
            Then, to find the error in f(a,b,c), we can sample the parameter values 
            of a* in the gaussian distribution centered around a with standard deviation of 
            δa (and similarly for b* and c*) and calculate f(a*,b*,c*) multiple times. 
            Then, we can take the standard deviation of that set of f(a*,b*,c*) values to find δf. 
        
        Args:
            method (str): solver method of SystemSolver e.g. "trapezoidal" 
            *args: arguments provided to solver method, e.g. ts, x0
            parse_output (function pointer): 
                how to parse output after solving, 
                e.g. with dynamic trapezoidal parse_output = lambda X: X[0]
            **kwargs: keyword arguments provided to solver method 
        """
        self.solves["original"] = parse_output(
            getattr(self.system, method)(*args, **kwargs)
        )
        self.calculate_error(
            method, *args, parse_output=parse_output, num_samples=num_samples, **kwargs
        )
        solves_with_error = np.array(self.solves["with_error"])
        self.solves["std"] = np.std(solves_with_error, axis=0)

    def plot(self, ts, **kwargs):
        """
        Plot state evolution along with error bars.
        Wrapper on plot_full_evolution.

        Args:
            ts (np.ndarray): timesteps
        
        Returns:
            fig: matplotlib figure
            ax: matplotlib axis
        """
        fig, ax = plot_full_evolution(
            self.solves["original"],
            ts,
            xs_min=(self.solves["original"] - self.solves["std"]),
            xs_max=(self.solves["original"] + self.solves["std"]),
            **kwargs
        )
        ax.legend()
        fig.tight_layout()
        return fig, ax


class MultiModeError(SystemError):
    def calculate_error(
        self, method: str, *args, num_samples=11, parse_output=lambda X: X, **kwargs
    ):
        """
        Overriden.
        """
        # =====================
        def nonnegative(a):
            a[a < 0] = 0
            return a

        def couplings_list2dict(cps):
            # TODO: remove when couplings are represented by dict
            return {(min(row[0], row[1]), max(row[0], row[1])): row[2] for row in cps}

        def couplings_dict2list(cps):
            return np.array([[key[0], key[1], val] for key, val in cps.items()])

        # =====================
        # error
        omegas_error = np.array(self.params_error["omegas"])
        kappas_error = np.array(self.params_error["kappas"])
        gammas_error = np.array(self.params_error["gammas"])
        kerrs_error = np.array(self.params_error["kerrs"])
        couplings_error = couplings_list2dict(np.array(self.params_error["couplings"]))

        params_original = self.system.params.copy()
        couplings_original_dict = couplings_list2dict(params_original["couplings"])

        params = self.system.params.copy()

        num_modes = params_original["num_modes"]
        num_variables = 4 * num_modes + len(couplings_original_dict)
        scalings = np.random.normal(size=(num_samples, num_variables))

        self.solves["with_error"] = []

        for scaling_vec in tqdm(scalings):
            params["omegas"] = nonnegative(
                params_original["omegas"] + omegas_error * scaling_vec[:num_modes]
            )
            params["kappas"] = nonnegative(
                params_original["kappas"]
                + kappas_error * scaling_vec[num_modes : num_modes * 2]
            )
            params["gammas"] = nonnegative(
                params_original["gammas"]
                + gammas_error * scaling_vec[num_modes * 2 : num_modes * 3]
            )
            params["kerrs"] = nonnegative(
                params_original["gammas"]
                + kerrs_error * scaling_vec[num_modes * 3 : num_modes * 4]
            )
            start = num_modes * 4
            params["couplings"] = nonnegative(
                couplings_dict2list(
                    {
                        key: val + scaling_vec[start + j] * couplings_error[key]
                        for j, (key, val) in enumerate(couplings_original_dict.items())
                    }
                )
            )

            system = MultiModeSystem(params)
            self.solves["with_error"].append(
                parse_output(getattr(system, method)(*args, **kwargs))
            )
