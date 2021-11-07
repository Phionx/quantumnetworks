"""
Visualization Tools
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display


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


def animate_evolution(
    xs,
    ts,
    labels=None,
    num_frames=200,
    animation_time=5,
    xlabel="q",
    ylabel="p",
    **kwargs,
):
    if len(xs) % 2 != 0:
        raise ValueError("Please enter state data with an even number of rows.")

    num_modes = len(xs) // 2
    num_points = len(ts)

    fig, axs = plt.subplots(1, num_modes, figsize=(4 * num_modes, 4), squeeze=False)
    axs = axs[0]

    lines = []
    #  prepare axis
    for i in range(num_modes):
        q = xs[2 * i, :]
        p = xs[2 * i + 1, :]

        min_q = min(np.min(q) * 1.1, np.min(q) * 0.9)
        max_q = max(np.max(q) * 1.1, np.max(q) * 0.9)
        min_p = min(np.min(p) * 1.1, np.min(p) * 0.9)
        max_p = max(np.max(p) * 1.1, np.max(p) * 0.9)

        axs[i].set_aspect("equal", adjustable="box")
        axs[i].set_xlim([min_q, max_q])
        axs[i].set_ylim([min_p, max_p])

        lines.append(axs[i].plot([], [], "o", markersize=10, **kwargs)[0])
        if labels:
            xlabel = labels[2 * i]
            ylabel = labels[2 * i + 1]
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)

    def init_func():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(indx):
        i = indx * num_points // num_frames
        t_val = ts[i]

        for lnum, line in enumerate(lines):
            q_val = xs[2 * lnum, i]
            p_val = xs[2 * lnum + 1, i]
            line.set_data([q_val], [p_val])
            axs[lnum].set_title(f"t = {t_val:.2f}")
        return lines

    interval = animation_time * 1000 // num_frames
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init_func,
        frames=num_frames,
        interval=interval,
        blit=True,
    )
    fig.tight_layout()
    html = HTML(anim.to_jshtml())
    display(html)
    plt.close()
    return fig, axs
