"""
Visualization Tools
"""
import matplotlib.pyplot as plt


def plot_evolution(X, ts, fig=None, ax=None):
    fig = fig if fig is not None else plt.figure(figsize=(4, 3), dpi=200)
    ax = ax if ax is not None else fig.subplots()
    ax.plot(ts, X[0, :])
    ax.set_xlabel("Time")
    ax.set_ylabel("X")
    ax.set_title("Time Evolution of X")
