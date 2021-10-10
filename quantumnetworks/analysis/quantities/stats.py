"""
Statistical Quantities
"""

import numpy as np

from quantumnetworks.analysis.quantities.base import SystemQuantity


class Average(SystemQuantity):
    def calculate(self, xs: np.ndarray) -> np.ndarray:
        return np.average(xs, axis=1)


class Std(SystemQuantity):
    def calculate(self, xs: np.ndarray) -> np.ndarray:
        return np.std(xs, axis=1)
