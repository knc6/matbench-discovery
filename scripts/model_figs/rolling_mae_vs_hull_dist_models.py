"""Plot rolling MAE as a function of hull distance for all models."""


# %%
from typing import Final

import numpy as np
import plotly.graph_objects as go
from pymatviz.utils import save_fig

from matbench_discovery import PDF_FIGS, SITE_FIGS
from matbench_discovery.plots import rolling_mae_vs_hull_dist
from matbench_discovery.preds import (
    df_each_pred,
    df_metrics,
    df_preds,
    each_true_col,
    models,
)

__author__ = "Rhys Goodall, Janosh Riebesell"
__date__ = "2022-06-18"

df_err, df_std = None, None  # variables to cache rolling MAE and std


# %%
backend: Final = "plotly"

fig, df_err, df_std = rolling_mae_vs_hull_dist(
    e_above_hull_true=df_preds[each_true_col],
    e_above_hull_errors=df_each_pred[models],
    backend=backend,
    with_sem=False,
    df_rolling_err=df_err,
    df_err_std=df_std,
    show_dummy_mae=False,
)

if backend == "matplotlib":
    # increase line width in legend
    legend = fig.legend(frameon=False, loc="lower right")
    fig.figure.set_size_inches(10, 9)
    for handle in legend.get_lines():
        handle._linewidth *= 6
    for line in fig.lines:
        line._linewidth *= 2
else:
    for trace in fig.data:
        model = trace.name.split(" MAE=")[0]
        if model in df_metrics.T.sort_values("MAE").index[8:]:
            trace.visible = "legendonly"  # show only top models by default

    # increase line width
    fig.update_traces(line=dict(width=3))
    fig.layout.legend.update(
        bgcolor="rgba(0,0,0,0)", title="", x=1.01, y=0, yanchor="bottom"
    )
    # increase legend handle size and reverse order
    fig.layout.margin.update(l=5, r=5, t=5, b=55)

    # plot marginal histogram of true hull distances
    counts, bins = np.histogram(
        df_preds[each_true_col], bins=400, range=fig.layout.xaxis.range
    )
    marginal_trace = go.Scatter(
        x=bins, y=counts, name="Density", fill="tozeroy", showlegend=False, yaxis="y2"
    )
    marginal_trace.marker.color = "rgba(0, 150, 200, 1)"
    # add marginal trace to existing figure
    fig.add_trace(marginal_trace)

    # update layout to include marginal plot
    fig.layout.update(
        yaxis1=dict(domain=[0, 0.75]),  # main yaxis
        yaxis2=dict(  # marginal yaxis
            domain=[0.8, 1], tickformat="s", tickvals=[*range(0, 100_000, 2000)]
        ),
    )
    fig.show()

img_name = "rolling-mae-vs-hull-dist-models"


# %%
save_fig(fig, f"{SITE_FIGS}/{img_name}.svelte")
save_fig(fig, f"{PDF_FIGS}/{img_name}.pdf", width=600, height=400)
