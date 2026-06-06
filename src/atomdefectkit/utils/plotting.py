"""Shared plotting helpers for workflow outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt

DEFAULT_TITLE_FONTSIZE = 12


def plot_xy_curves(
    curves,
    xlabel,
    ylabel,
    title,
    save_path=None,
    figsize=(6, 4),
    show_legend=False,
    grid_alpha=0.3,
):
    """Plot one or more x-y curves with a shared style.

    Args:
        curves: Iterable of dictionaries with ``x`` and ``y`` arrays plus optional
            ``label``, ``marker``, ``linewidth``, and ``linestyle`` entries.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Figure title.
        save_path: Optional file path for ``Figure.savefig``.
        figsize: Matplotlib figure size.
        show_legend: Whether to draw a legend.
        grid_alpha: Alpha value used for the grid.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Created figure and axes.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for curve in curves:
        ax.plot(
            curve["x"],
            curve["y"],
            marker=curve.get("marker", "o"),
            label=curve.get("label"),
            linewidth=curve.get("linewidth"),
            linestyle=curve.get("linestyle", "-"),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=DEFAULT_TITLE_FONTSIZE)
    ax.grid(True, alpha=grid_alpha)
    if show_legend:
        ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    return fig, ax
