# -*- coding: utf-8 -*-

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Python functions which run only within a Jupyter or Colab notebook."""

import random
import string
import sys
from typing import List, Optional, Tuple

from vertexai.preview.evaluation import _base as eval_base
from vertexai.preview.evaluation import constants

# pylint: disable=g-import-not-at-top
try:
    import pandas as pd
except ImportError:
    pandas = None

_MARKDOWN_H2 = "##"
_MARKDOWN_H3 = "###"
_DEFAULT_COLUMNS_TO_DISPLAY = [
    constants.Dataset.MODEL_RESPONSE_COLUMN,
    constants.Dataset.BASELINE_MODEL_RESPONSE_COLUMN,
    constants.Dataset.PROMPT_COLUMN,
    constants.MetricResult.ROW_COUNT_KEY,
]
_DEFAULT_RADAR_RANGE = (0, 5)


def _get_ipython_shell_name() -> str:
    if "IPython" in sys.modules:
        # pylint: disable=g-import-not-at-top, g-importing-member
        from IPython import get_ipython

        return get_ipython().__class__.__name__
    return ""


def is_ipython_available() -> bool:
    return _get_ipython_shell_name()


def _filter_df(
    df: pd.DataFrame, substrings: Optional[List[str]] = None
) -> pd.DataFrame:
    """Filters a DataFrame to include only columns containing the given substrings."""
    if substrings is None:
        return df

    return df.copy().filter(
        [
            column_name
            for column_name in df.columns
            if any(substring in column_name for substring in substrings)
        ]
    )


def display_eval_result(
    eval_result: "eval_base.EvalResult",
    title: Optional[str] = None,
    metrics: Optional[List[str]] = None,
) -> None:
    """Displays evaluation results in a notebook using IPython.display.

    Args:
        eval_result: An object containing evaluation results with
          `summary_metrics` and `metrics_table` attributes.
        title: A string title to display above the results.
        metrics: A list of metric name substrings to filter displayed columns. If
          provided, only metrics whose names contain any of these strings will be
          displayed.
    """
    if not is_ipython_available():
        return
    # pylint: disable=g-import-not-at-top, g-importing-member
    from IPython.display import display
    from IPython.display import Markdown

    summary_metrics, metrics_table = (
        eval_result.summary_metrics,
        eval_result.metrics_table,
    )

    summary_metrics_df = pd.DataFrame.from_dict(summary_metrics, orient="index").T

    if metrics:
        columns_to_keep = metrics + _DEFAULT_COLUMNS_TO_DISPLAY
        summary_metrics_df = _filter_df(summary_metrics_df, columns_to_keep)
        metrics_table = _filter_df(metrics_table, columns_to_keep)

    # Display the title in Markdown.
    if title:
        display(Markdown(f"{_MARKDOWN_H2} {title}"))

    # Display the summary metrics.
    display(Markdown(f"{_MARKDOWN_H3} Summary Metrics"))
    display(summary_metrics_df)

    # Display the metrics table.
    display(Markdown(f"{_MARKDOWN_H3} Row-based Metrics"))
    display(metrics_table)


def display_explanations(
    eval_result: "eval_base.EvalResult",
    num: int = 1,
    metrics: Optional[List[str]] = None,
) -> None:
    """Displays the explanations in a notebook using IPython.display.

    Args:
        eval_result: An object containing evaluation results. It is expected to
          have attributes `summary_metrics` and `metrics_table`.
        num: The number of row samples to display. Defaults to 1. If the number of
          rows is less than `num`, all rows will be displayed.
        metrics: A list of metric name substrings to filter displayed columns. If
          provided, only metrics whose names contain any of these strings will be
          displayed.
    """
    if not is_ipython_available():
        return
    # pylint: disable=g-import-not-at-top, g-importing-member
    from IPython.display import display
    from IPython.display import HTML

    style = "white-space: pre-wrap; width: 1500px; overflow-x: auto;"
    metrics_table = eval_result.metrics_table

    if num < 1:
        raise ValueError("Num must be greater than 0.")
    num = min(num, len(metrics_table))

    df = metrics_table.sample(n=num)

    if metrics:
        columns_to_keep = metrics + _DEFAULT_COLUMNS_TO_DISPLAY
        df = _filter_df(df, columns_to_keep)

    for _, row in df.iterrows():
        for col in df.columns:
            display(HTML(f"<div style='{style}'><h4>{col}:</h4>{row[col]}</div>"))
        display(HTML("<hr>"))


def display_radar_plot(
    eval_results_with_title: List[Tuple[str, "eval_base.EvalResult"]],
    metrics: List[str],
    radar_range: Tuple[float, float] = _DEFAULT_RADAR_RANGE,
) -> None:
    """Plots a radar plot comparing evaluation results.

    Args:
        eval_results_with_title: List of (title, eval_result) tuples.
        metrics: A list of metrics whose mean values will be plotted.
        radar_range: Range of the radar plot axes.
    """
    # pylint: disable=g-import-not-at-top
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            '`plotly` is not installed. Please install using "!pip install plotly"'
        ) from exc

    fig = go.Figure()
    for title, eval_result in eval_results_with_title:
        summary_metrics = eval_result.summary_metrics
        if metrics:
            summary_metrics = {
                key.replace("/mean", ""): summary_metrics[key]
                for key in summary_metrics
                if any(selected_metric + "/mean" in key for selected_metric in metrics)
            }
        fig.add_trace(
            go.Scatterpolar(
                r=list(summary_metrics.values()),
                theta=list(summary_metrics.keys()),
                fill="toself",
                name=title,
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=radar_range)),
        showlegend=True,
    )
    fig.show()


def display_bar_plot(
    eval_results_with_title: List[Tuple[str, "eval_base.EvalResult"]],
    metrics: List[str],
) -> None:
    """Plots a bar plot comparing evaluation results.

    Args:
        eval_results_with_title: List of (title, eval_result) tuples.
        metrics: A list of metrics whose mean values will be plotted.
    """

    # pylint: disable=g-import-not-at-top
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            '`plotly` is not installed. Please install using "!pip install plotly"'
        ) from exc

    data = []

    for title, eval_result in eval_results_with_title:
        summary_metrics = eval_result.summary_metrics
        mean_summary_metrics = [f"{metric}/mean" for metric in metrics]
        updated_summary_metrics = []
        if metrics:
            for k, v in summary_metrics.items():
                if k in mean_summary_metrics:
                    updated_summary_metrics.append((k, v))
            summary_metrics = dict(updated_summary_metrics)

        data.append(
            go.Bar(
                x=list(summary_metrics.keys()),
                y=list(summary_metrics.values()),
                name=title,
            )
        )

    fig = go.Figure(data=data)

    fig.update_layout(barmode="group", showlegend=True)
    fig.show()


def generate_uuid(length: int = 8) -> str:
    """Generates a uuid of a specified length (default=8)."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))
