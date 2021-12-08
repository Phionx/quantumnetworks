"""
Visualization Tools
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib import animation
from IPython.display import HTML, display
import networkx as nx


def plot_evolution(
    x: np.ndarray, ts: np.ndarray, x_min=None, x_max=None, fig=None, ax=None, **kwargs
):
    """
    Plot evolution of single dimension of a state vector, optionally add error bars.

    Args:
        x (np.ndarray): signle dimension of a state vector over time
        ts (np.ndarray): timesteps
        x_min (optional[np.ndarray]): error bar minimum on x
        x_max (optional[np.ndarray]): error bar maximum on x
        fig (optional): matplotlib figure
        ax (optional): matplotlib axis
        **kwargs: optional keyword arguments used for plotting

    Returns:
        fig: matplotlib figure
        ax: matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(4, 3), dpi=200)
    fig = ax.get_figure()

    if x_min is not None and x_max is not None:
        ax.fill_between(ts, x_min, x_max, alpha=0.1)
    ax.plot(ts, x, **kwargs)
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.set_title(f"Time Evolution of System")
    return fig, ax


def plot_full_evolution(
    xs: np.ndarray,
    ts: np.ndarray,
    xs_min=None,
    xs_max=None,
    labels=None,
    fig=None,
    ax=None,
    **kwargs,
):
    """
    Plot evolution of multiple dimensions of a state vector, optionally add error bars.

    Args:
        xs (np.ndarray): multiple dimensions of a state vector over time
        ts (np.ndarray): timesteps
        xs_min (optional[np.ndarray]): error bars minimum on xs
        xs_max (optional[np.ndarray]): error bars maximum on xs
        labels (optional[list]): list of labels for each dimension
        fig (optional): matplotlib figure
        ax (optional): matplotlib axis
        **kwargs: optional keyword arguments used for plotting

    Returns:
        fig: matplotlib figure
        ax: matplotlib axis
    """

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(4, 3), dpi=200)
    fig = ax.get_figure()

    for i, x in enumerate(xs):
        x_min = xs_min[i] if xs_min is not None else None
        x_max = xs_max[i] if xs_max is not None else None
        label = str(i)
        if labels:
            label = labels[i]
        plot_evolution(
            x, ts, x_min=x_min, x_max=x_max, fig=fig, ax=ax, label=label, **kwargs
        )

    if labels:
        ax.legend()
    return fig, ax


def plot_evolution_phase_space(X, fig=None, ax=None, use_arrows=True, **kwargs):
    """
    Plot evolution of single quantum network mode in phase space (q/p representation).

    Args:
        X (np.ndarray): q/p coordinates of a single quantum network mode over time
        fig (optional): matplotlib figure
        ax (optional): matplotlib axis
        use_arrows (bool): if true, use arrows on trace representing evolution
        **kwargs: other optional keyword arguments used for plotting
    
    Returns:
        fig: matplotlib figure
        ax: matplotlib axis
    """

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(4, 4), dpi=200)
    fig = ax.get_figure()

    q = X[0, :]
    p = X[1, :]
    ls = kwargs.pop("ls", "--")
    lw = kwargs.pop("lw", 0.5)
    arrow_width = kwargs.pop("arrow_width", 0.001)
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
    return fig, ax


def plot_full_evolution_phase_space(xs, **kwargs):
    """
    Plot evolution of multiple quantum network modes in phase space (q/p representation).

    Args:
        xs (np.ndarray): q/p coordinates of a multiple quantum network modes over time
        **kwargs: 
            optional keyword arguments used for plotting with self.plot_evolution_phase_space
    
    Returns:
        fig: matplotlib figure
        ax: matplotlib axis
    """
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
    """
    Helper functon to add arrows on a plot of y vs. x.

    Args:
        x (np.ndarray): x coordinates
        y (np.ndarray): x coordinates
        num_arrows (int): number of arrows desired
        **kwargs: keyword arguments sent to ax.arrow
    
    Returns:
        ax: matplotlib axis
    """
    N = len(x)
    for i in range(0, N, N // num_arrows + 1):
        x_val = x[i]
        y_val = y[i]
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        ax.arrow(x_val, y_val, dx, dy, **kwargs)
    return ax


def animate_evolution_old(
    xs,
    ts,
    labels=None,
    num_frames=200,
    animation_time=5,
    xlabel="q",
    ylabel="p",
    is_traced=True,
    save_animation=False,
    **kwargs,
):
    """
    Animate phase space evolution of multiple quantum network modes.
    Both generate and display this animation.

    Args:
        xs (np.ndarray): multiple dimensions of a state vector over time
        ts (np.ndarray): timesteps
        labels (optional[list]): list of labels for each q/p dimension
        num_frames (int): number of frames in animation
        animation_time (float): length of animation in seconds
        xlabel (str): default xlabel
        xlabel (str): default ylabel
        is_traced (bool): whether or not to draw trace of path in q/p space
        save_animation (optional[str]): 
                if save_animation is not None, then the animation will be saved 
                to a filename represented by the save_animation string
        **kwargs: optional keyword arguments used for animation
        
    Returns:
        fig: matplotlib figure
        axs: matplotlib axes
    """
    if len(xs) % 2 != 0:
        raise ValueError("Please enter state data with an even number of rows.")

    num_modes = len(xs) // 2
    num_points = len(ts)

    fig, axs = plt.subplots(
        1, num_modes, figsize=(4 * num_modes, 4), dpi=100, squeeze=False
    )
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
        min_tot = min(min_q, min_p)
        max_tot = max(max_q, max_p)
        axs[i].set_xlim([min_tot, max_tot])
        axs[i].set_ylim([min_tot, max_tot])

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
            if is_traced:
                plot_evolution_phase_space(
                    xs[2 * lnum : 2 * lnum + 2, : i + 1], ax=axs[lnum], use_arrows=False
                )
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
    if save_animation:
        animation_title = (
            save_animation if isinstance(save_animation, str) else "animation.gif"
        )
        anim.save(animation_title, writer="pillow", fps=60)
    html = HTML(anim.to_jshtml())
    display(html)
    plt.close()
    return fig, axs


def animate_evolution(
    xs,
    ts,
    labels=None,
    num_frames=200,
    animation_time=5,
    xlabel="q",
    ylabel="p",
    is_traced=True,
    save_animation=None,
    **kwargs,
):
    """
    Animate phase space evolution of multiple quantum network modes.
    Both generate and display this animation.

    Args:
        xs (np.ndarray): multiple dimensions of a state vector over time
        ts (np.ndarray): timesteps
        labels (optional[list]): list of labels for each q/p dimension
        num_frames (int): number of frames in animation
        animation_time (float): length of animation in seconds
        xlabel (str): default xlabel
        xlabel (str): default ylabel
        is_traced (bool): whether or not to draw trace of path in q/p space
        save_animation (optional[str]): 
                if save_animation is not None, then the animation will be saved 
                to a filename represented by the save_animation string
        **kwargs: optional keyword arguments used for animation
        
    Returns:
        fig: matplotlib figure
        axs: matplotlib axes
    """
    if len(xs) % 2 != 0:
        raise ValueError("Please enter state data with an even number of rows.")

    num_modes = len(xs) // 2
    num_points = len(ts)

    fig, axs = plt.subplots(
        1, num_modes, figsize=(4 * num_modes, 4), dpi=100, squeeze=False
    )
    axs = axs[0]

    min_tot = []
    max_tot = []
    #  prepare axis
    for i in range(num_modes):
        q = xs[2 * i, :]
        p = xs[2 * i + 1, :]

        min_q = min(np.min(q) * 1.1, np.min(q) * 0.9)
        max_q = max(np.max(q) * 1.1, np.max(q) * 0.9)
        min_p = min(np.min(p) * 1.1, np.min(p) * 0.9)
        max_p = max(np.max(p) * 1.1, np.max(p) * 0.9)

        axs[i].set_aspect("equal", adjustable="box")
        min_tot.append(min(min_q, min_p))
        max_tot.append(max(max_q, max_p))
        axs[i].set_xlim([min_tot[-1], max_tot[-1]])
        axs[i].set_ylim([min_tot[-1], max_tot[-1]])

        if labels:
            xlabel = labels[2 * i]
            ylabel = labels[2 * i + 1]
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)

    def animate(indx):
        i = indx * num_points // num_frames
        t_val = ts[i]

        for ax_num, ax in enumerate(axs):
            ax.clear()
            ax.grid()

            ax.set_xlim([min_tot[ax_num], max_tot[ax_num]])
            ax.set_ylim([min_tot[ax_num], max_tot[ax_num]])

            q_val = xs[2 * ax_num, i]
            p_val = xs[2 * ax_num + 1, i]
            ax.plot([q_val], [p_val], "o", markersize=10, **kwargs)
            ax.set_title(f"t = {t_val:.2f}")
            if is_traced:
                plot_evolution_phase_space(
                    xs[2 * ax_num : 2 * ax_num + 2, : i + 1],
                    ax=axs[ax_num],
                    use_arrows=False,
                    color="blue",
                )
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
    return fig, axs


def draw_graph(
    graph: nx.Graph,
    ax=None,
    pos=None,
    with_node_labels=True,
    with_edge_labels=True,
    **kwargs,
):
    """
    Plots 2D graphs in IPython/Jupyter.

    Args:
        graph (rx.PyGraph): graph to be plotted
        ax (optional): matplotlib axis
        pos (optional[dict]): a dictionary with nodes as keys and positions as values.
        with_node_labels (bool): if true, then plot with node labels
        with_edge_labels (bool): if true, then plot with edge (coupling) labels
        **kwargs: other key word arguments used to modify the network graph visualization

    Returns:
        fig: matplotlib figure
        ax: matplotlib axis
        pos: a dictionary with nodes as keys and positions as values.
    """
    # Figure
    dpi = kwargs.pop("dpi", 200)

    if ax is None:
        fig = plt.figure(dpi=dpi)
        ax = fig.subplots()
    fig = ax.get_figure()

    # Graph

    font_size = kwargs.pop("font_size", 20)
    node_size = kwargs.pop("node_size", 2000)
    node_color = kwargs.pop("node_color", "pink")
    edge_color = kwargs.pop("edge_color", "black")

    edge_labels = dict(
        [((n1, n2), np.around(d["weight"], 3)) for n1, n2, d in graph.edges(data=True)]
    )
    pos = nx.spring_layout(graph) if pos is None else pos

    nx.draw_networkx(
        graph,
        pos,
        ax=ax,
        edge_color=edge_color,
        node_size=node_size,
        node_color=node_color,
        font_size=font_size,
        with_labels=with_node_labels,
        labels={node: node for node in graph.nodes()} if with_node_labels else None,
        **kwargs,
    )

    if with_edge_labels:
        nx.draw_networkx_edge_labels(
            graph, ax=ax, pos=pos, edge_labels=edge_labels, font_size=font_size
        )
    ax.collections[0].set_edgecolor("#000000")
    fig.tight_layout()
    return (fig, ax, pos)

