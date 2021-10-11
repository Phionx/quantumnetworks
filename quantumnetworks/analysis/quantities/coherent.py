"""
Coherent state Quantities of Interest
"""

import numpy as np
from quantumnetworks.analysis.quantities.base import SystemQuantity

from scipy.optimize import curve_fit


def decaying_sinusoid(t, a, decay, w, phi0):
    return a * np.exp(-1.0 * decay * t / 2.0) * np.cos(w * t + phi0)


class Decay(SystemQuantity):
    def calculate(self, xs: np.ndarray) -> np.ndarray:
        decay = []
        for x in xs:
            popt, _ = curve_fit(decaying_sinusoid, self.ts, x, p0=(1, 1, 1, 0))
            decay.append(popt[1])
        decay_np = np.array(decay)
        return decay_np

