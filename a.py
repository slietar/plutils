import polars as pl
from polars import col as c

from expression_lib import implode_like


pl.Config.set_verbose(True)


df = pl.DataFrame(dict(
    a=[
        [3, 4],
        [5, 6, 7],
        [8],
    ]
), schema=pl.Schema({
    "a": pl.List(pl.Float32),
}))

print(
    df.select(
        c.a,
        b=implode_like(
            c.a.list.explode() * 2,
            c.a,
        ),
    )
)
