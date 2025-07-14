from __future__ import annotations


def _make_arg_signature(ndim: int, var_symbol: str) -> str:
    return ",".join(f"{var_symbol}{dim + 1}" for dim in range(ndim))
