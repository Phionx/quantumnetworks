"""
Visualization Tools
"""
import matplotlib.pyplot as plt


def plot_evolution(x, ts, fig=None, ax=None, **kwargs):
    fig = fig if fig is not None else plt.figure(figsize=(4, 3), dpi=200)
    ax = ax if ax is not None else fig.subplots()
    ax.plot(ts, x, **kwargs)
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.set_title(f"Time Evolution of System")
    return fig, ax
