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
        label = str(i)
        if labels:
            label = labels[i]
        plot_evolution(x, ts, fig=fig, ax=ax, label=label, **kwargs)

    if labels:
        ax.legend()
    return fig, ax


def plot_evolution_phase_space(
    X, fig=None, ax=None, use_arrows=True, arrow_width=0.001, **kwargs
):
    fig = fig if fig is not None else plt.figure(figsize=(4, 4), dpi=200)
    ax = ax if ax is not None else fig.subplots()
    q = X[0, :]
    p = X[1, :]
    ls = kwargs.pop("ls", "--")
    lw = kwargs.pop("lw", 0.5)
    ax.plot(q, p, ls=ls, lw=lw, **kwargs)
    if use_arrows:
        insert_arrows(
            q,
            p,
            ax,
            fc="blue",
            color="blue",
            ec="blue",
            num_arrows=50,
            length_includes_head=True,
            width=arrow_width,
        )
        insert_arrows(
            q,
            p,
            ax,
            fc="red",
            color="red",
            ec="red",
            num_arrows=1,
            length_includes_head=True,
            width=arrow_width,
        )  # start arrow
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_title(f"Phase Space Evolution")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    ax.grid()
    return fig, ax


def plot_full_evolution_phase_space(xs, **kwargs):
    num_modes = len(xs) // 2
    fig, axs = plt.subplots(
        1, num_modes, figsize=(4 * num_modes, 4), dpi=200, squeeze=False
    )
    axs = axs[0]
    for i in range(num_modes):
        plot_evolution_phase_space(xs[2 * i : 2 * i + 2], fig=fig, ax=axs[i], **kwargs)
        axs[i].set_xlabel(f"$q_{i+1}$")
        axs[i].set_ylabel(f"$p_{i+1}$")
    return fig, axs


def insert_arrows(x, y, ax, num_arrows=10, **kwargs):
    N = len(x)
    for i in range(0, N, N // num_arrows + 1):
        x_val = x[i]
        y_val = y[i]
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        ax.arrow(x_val, y_val, dx, dy, **kwargs)
    return ax


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
