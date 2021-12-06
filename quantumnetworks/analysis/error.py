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
        self.system = system
        self.params_error = params_error
        self.solves = None
        self.reset_solves()

    def reset_solves(self):
        self.solves = {"original": None, "with_error": [], "std": None}

    @abstractmethod
    def calculate_error(self, method: str, *args, parse_output=lambda X: X, **kwargs):
        pass

    def run(
        self, method: str, *args, parse_output=lambda X: X, num_samples=None, **kwargs
    ):
        self.solves["original"] = parse_output(
            getattr(self.system, method)(*args, **kwargs)
        )
        self.calculate_error(
            method, *args, parse_output=parse_output, num_samples=num_samples, **kwargs
        )
        solves_with_error = np.array(self.solves["with_error"])
        self.solves["std"] = np.std(solves_with_error, axis=0)

    def plot(self, ts, **kwargs):
        fig, ax = plot_full_evolution(
            self.solves["original"],
            ts,
            xs_min=self.solves["original"] - self.solves["std"],
            xs_max=self.solves["original"] + self.solves["std"],
            **kwargs
        )
        ax.legend()
        fig.tight_layout()
        return fig, ax


class MultiModeError(SystemError):
    def calculate_error(
        self, method: str, *args, num_samples=None, parse_output=lambda X: X, **kwargs
    ):
        params = self.system.params.copy()
        omegas_original = np.array(params["omegas"])
        omegas_error = np.array(self.params_error["omegas"])

        num_variables = len(omegas_original)
        scalings = np.random.normal(size=num_samples * num_variables).reshape(
            num_samples, num_variables
        )

        self.solves["with_error"] = []

        for scaling_vec in tqdm(scalings):
            # omegas
            omegas_copy = omegas_original.copy()
            omegas_copy += omegas_error * scaling_vec
            params["omegas"] = omegas_copy
            system = MultiModeSystem(params)
            self.solves["with_error"].append(
                parse_output(getattr(system, method)(*args, **kwargs))
            )
