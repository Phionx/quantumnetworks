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


def plot_full_evolution(xs, ts, labels=None, fig=None, ax=None, **kwargs):
    fig = fig if fig is not None else plt.figure(figsize=(4, 3), dpi=200)
    ax = ax if ax is not None else fig.subplots()
    for i, x in enumerate(xs):
        if labels:
            label = labels[i]
        plot_evolution(x, ts, fig=fig, ax=ax, label=label, **kwargs)

    if labels:
        ax.legend()
    return fig, ax
