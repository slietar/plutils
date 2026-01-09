import contextlib
import random
import time
from typing import Optional

import polars as pl
from polars import col as c

import plutils


pl.Config.set_verbose(True)


@contextlib.contextmanager
def measure(name: Optional[str] = None):
    start_time = time.time_ns()

    try:
        yield
    finally:
        end_time = time.time_ns()
        print(f"{name or 'Execution time'}: {(end_time - start_time) * 1e-9:.3f} s")



df = pl.DataFrame(dict(
    a=[
        # [],
        [3, 4],
        [5, 6, 7],
        [8],
        [],
    ],
), schema=pl.Schema({
    "a": pl.List(pl.UInt32),
}))

print(plutils.get_offsets(df["a"]))

# df = pl.concat([df] * 10_000)
# df.vstack(df, in_place=True)
# df = df.select(
#     c.a.append([])
# )

b = c.a.list.eval(
    pl.element() * 2,
).alias("b")

res = df.select(
    # implode_like(c.a.list.explode(empty_as_null=False), c.a),
    plutils.implode_like_expr(
        pl.struct(
            c.a.list.explode(empty_as_null=False),
            b.list.explode(empty_as_null=False),
        ),
        # c.a,
        pl.when(c.a.list.len() > 0).then(c.a),
    )
    # c.a.list.explode()
)

print(res)



# Benchmarks

# df1 = pl.DataFrame(dict(
#     a=[
#         [random.randint(1, 10_000) for _ in range(random.randint(1, 100))] for _ in range(1_000_000)
#     ],
# ))


# with measure("Method 1"):
#     res1 = (
#         df1.select(
#             c.a,
#             b=implode_like(
#                 c.a.list.explode() * 2,
#                 c.a,
#             ),
#         )
#     )

# with measure("Method 2"):
#     res2 = (
#         df1.select(
#             c.a,
#             b=(c.a.list.explode() * 2).implode().over(pl.row_index())
#         )
#     )

# # print(res1)
# # print(res2)

# assert res1.equals(res2)


# def zip(a: pl.Expr, b: pl.Expr) -> pl.Expr:
#     return implode_like(
#         pl.struct(
#             a.list.explode(),
#             b.list.explode(),
#         ),
#         a,
#     )
