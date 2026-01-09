from pathlib import Path

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

from .plutils import get_offsets, implode_like  # noqa: F401


PLUGIN_PATH = Path(__file__).parent


def implode_like_expr(target_expr: IntoExpr, layout_expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="implode_like_expr",
        args=(target_expr, layout_expr),
        is_elementwise=False,
        changes_length=True,
    )
