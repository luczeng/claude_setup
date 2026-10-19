"""
Differential check + benchmark for "before vs after" simplifications.

Import it from a throwaway check script (never from the user's codebase):

    import sys
    sys.path.insert(0, "<this skill's directory>/scripts")
    from equiv import compare, bench, edge_arrays, case

    rng = np.random.default_rng(0)
    cases = [case(x, 5, label=f"{name}, w=5") for name, x in edge_arrays(rng)]
    cases += [case(rng.standard_normal(10_000) + 1e6, 101, label="long + DC offset")]

    ok = compare(old_smooth, new_smooth, cases, rtol=1e-10)
    # outputs with near-zero entries next to large ones (spectra): tolerance relative to the peak
    ok = compare(old_psd, new_psd, cases, rtol=1e-9, peak_atol=1e-12)

    bench(old_smooth, new_smooth, case(rng.standard_normal(100_000), 101))
    bench(None, new_smooth, case(rng.standard_normal(10_000_000), 101))  # new only, full size

What compare() checks, per case:
  * both versions see independent deep copies of the inputs
  * outputs match: type (list vs tuple vs ndarray vs scalar; np.float64 counts
    as float), shape, dtype, and values (exact for ints/bools/strings,
    rtol/atol/peak_atol for floats, NaN positions equal)
  * if one version raises, the other must raise the same exception type
  * inputs are left in the same state afterwards (catches in-place mutation
    that one version does and the other doesn't)
"""

from __future__ import annotations

import copy
import math
import numbers
import timeit
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    label: str = ""


def case(*args, label: str = "", **kwargs) -> Case:
    """Build a test case: case(x, 5, mode="same", label="short signal")."""
    return Case(args=args, kwargs=kwargs, label=label)


def edge_arrays(rng: np.random.Generator | None = None, n: int = 257,
                include_int: bool = True) -> list[tuple[str, np.ndarray]]:
    """Common 1-D edge cases. Returns [(label, array), ...].

    Add cases specific to the code under test on top of these (ties for
    argmax, values at window boundaries, duplicates for searches, ...).
    """
    rng = rng or np.random.default_rng(0)
    x = rng.standard_normal(n)
    out = [
        ("empty", np.array([], dtype=float)),
        ("length 1", np.array([0.7])),
        ("length 2", np.array([1.0, -1.0])),
        ("random normal", x),
        ("constant (all ties)", np.full(n, 3.0)),
        ("sorted increasing", np.sort(x)),
        ("sorted decreasing", np.sort(x)[::-1].copy()),
        ("all negative", -np.abs(x) - 1.0),
        ("contains NaN", np.where(np.arange(n) % 37 == 5, np.nan, x)),
        ("contains inf", np.where(np.arange(n) == n // 2, np.inf, x)),
        ("large offset 1e9", x + 1e9),
        ("tiny scale 1e-12", x * 1e-12),
    ]
    if include_int:
        out += [
            ("small ints", rng.integers(-5, 6, n)),
            ("large int64", rng.integers(2**40, 2**41, n, dtype=np.int64)),
        ]
    return out


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
@dataclass
class _Diff:
    problems: list[str] = field(default_factory=list)
    max_abs: float = 0.0
    max_rel: float = 0.0

    def add(self, msg: str) -> None:
        self.problems.append(msg)


def _is_array_like_number(x: Any) -> bool:
    return isinstance(x, (np.ndarray, np.generic, numbers.Number)) and not isinstance(x, bool)


# np.float64 / np.complex128 subclass Python float / complex and behave the same,
# so mixing them is not a behaviour change. np.int64 vs int IS (overflow), as is
# np.float32 vs float.
_EQUIVALENT_TYPES = ({float, np.float64}, {complex, np.complex128})


def _types_differ(a: Any, b: Any) -> bool:
    ta, tb = type(a), type(b)
    return ta is not tb and not any(ta in s and tb in s for s in _EQUIVALENT_TYPES)


def _peak(x: Any) -> float:
    """Largest finite |value| anywhere in a (nested) output; 0 if none."""
    if isinstance(x, dict):
        return max((_peak(v) for v in x.values()), default=0.0)
    if isinstance(x, (list, tuple)) and not all(_is_array_like_number(v) for v in x):
        return max((_peak(v) for v in x), default=0.0)
    try:
        arr = np.asarray(x)
        if arr.dtype.kind not in "biufc" or arr.size == 0:
            return 0.0
        vals = np.abs(arr.astype(np.complex128 if arr.dtype.kind == "c" else np.float64))
        vals = vals[np.isfinite(vals)]
        return float(vals.max()) if vals.size else 0.0
    except (TypeError, ValueError):
        return 0.0


def _compare_numeric(a, b, path: str, rtol: float, atol: float, strict_types: bool, d: _Diff) -> None:
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if strict_types and _types_differ(a, b):
        # e.g. list vs ndarray, int vs np.int64, float vs np.float32
        d.add(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
    if a_arr.shape != b_arr.shape:
        d.add(f"{path}: shape {a_arr.shape} vs {b_arr.shape}")
        return
    if strict_types and isinstance(a, (np.ndarray, np.generic)) and a_arr.dtype != b_arr.dtype:
        d.add(f"{path}: dtype {a_arr.dtype} vs {b_arr.dtype}")

    kinds = {a_arr.dtype.kind, b_arr.dtype.kind}
    if kinds <= set("biu"):  # bool / int / uint: exact
        if not np.array_equal(a_arr, b_arr):
            neq = np.atleast_1d(a_arr != b_arr)
            i = int(np.flatnonzero(neq)[0])
            av, bv = np.atleast_1d(a_arr)[i], np.atleast_1d(b_arr)[i]
            d.add(f"{path}: {int(neq.sum())} integer/bool element(s) differ (first at [{i}]: {av!r} vs {bv!r})")
        return
    if not kinds <= set("biufc"):
        if not np.array_equal(a_arr, b_arr):
            d.add(f"{path}: non-numeric arrays differ")
        return

    a_f = a_arr.astype(np.complex128 if "c" in kinds else np.float64)
    b_f = b_arr.astype(a_f.dtype)
    nan_a, nan_b = np.isnan(a_f), np.isnan(b_f)
    if not np.array_equal(nan_a, nan_b):
        d.add(f"{path}: NaN positions differ ({int(nan_a.sum())} vs {int(nan_b.sum())} NaNs)")
        return
    inf_a, inf_b = np.isinf(a_f), np.isinf(b_f)
    if not np.array_equal(inf_a, inf_b) or not np.array_equal(a_f[inf_a], b_f[inf_b]):
        d.add(f"{path}: inf positions/signs differ")
        return
    finite = ~(nan_a | inf_a)
    if not finite.any():
        return
    af, bf = a_f[finite], b_f[finite]
    abs_err = np.abs(af - bf)
    denom = np.maximum(np.abs(af), np.finfo(float).tiny)
    d.max_abs = max(d.max_abs, float(abs_err.max()))
    d.max_rel = max(d.max_rel, float((abs_err / denom).max()))
    bad = abs_err > atol + rtol * np.abs(af)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        d.add(f"{path}: {int(bad.sum())} value(s) outside rtol={rtol:g}, atol={atol:g} "
              f"(first: {af[i]!r} vs {bf[i]!r})")


def _compare(a, b, path: str, rtol: float, atol: float, strict_types: bool, d: _Diff) -> None:
    if _is_array_like_number(a) and _is_array_like_number(b):
        _compare_numeric(a, b, path, rtol, atol, strict_types, d)
        return
    if isinstance(a, bool) or isinstance(b, bool):
        if type(a) is not type(b) and strict_types:
            d.add(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
        if a != b:
            d.add(f"{path}: {a!r} vs {b!r}")
        return
    if _types_differ(a, b):
        if strict_types:
            d.add(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
            return
        # loose mode: compare list/tuple/ndarray by content
        if isinstance(a, (list, tuple, np.ndarray)) and isinstance(b, (list, tuple, np.ndarray)):
            _compare_numeric(np.asarray(a), np.asarray(b), path, rtol, atol, False, d)
            return
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            d.add(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, f"{path}[{i}]", rtol, atol, strict_types, d)
            if len(d.problems) > 5:
                return
        return
    if isinstance(a, dict):
        if set(a) != set(b):
            d.add(f"{path}: dict keys differ: {sorted(map(str, set(a) ^ set(b)))}")
            return
        for k in a:
            _compare(a[k], b[k], f"{path}[{k!r}]", rtol, atol, strict_types, d)
        return
    if isinstance(a, np.ndarray):  # non-numeric ndarray
        _compare_numeric(a, b, path, rtol, atol, strict_types, d)
        return
    try:
        equal = a == b
        if isinstance(equal, np.ndarray):
            equal = equal.all()
    except Exception as e:  # pragma: no cover
        d.add(f"{path}: could not compare ({e})")
        return
    if not equal:
        d.add(f"{path}: {a!r} vs {b!r}")


def _run(fn: Callable, c: Case):
    args, kwargs = copy.deepcopy(c.args), copy.deepcopy(c.kwargs)
    try:
        return fn(*args, **kwargs), None, (args, kwargs)
    except Exception as e:  # noqa: BLE001 - we want every exception
        return None, e, (args, kwargs)


def compare(ref: Callable, cand: Callable, cases: Iterable[Case | tuple], *,
            rtol: float = 1e-10, atol: float = 0.0, peak_atol: float = 0.0,
            strict_types: bool = True, check_mutation: bool = True, verbose: bool = True) -> bool:
    """Run ref and cand on every case and report differences.

    Returns True if every case matches. Plain tuples are accepted as
    positional-argument cases.

    peak_atol: per-case absolute tolerance relative to the largest |value| in
        that case's reference output (e.g. 1e-12). Use it when some outputs sit
        near zero next to large ones (spectra, residuals), where relative error
        is meaningless. It is added to `atol`.
    strict_types: set False only when the user has agreed that e.g.
        list -> ndarray is an acceptable change.
    """
    cases = [c if isinstance(c, Case) else Case(args=tuple(c)) for c in cases]
    n_fail = 0
    worst_abs = worst_rel = 0.0
    lines = []
    for i, c in enumerate(cases):
        label = c.label or f"case {i}"
        out_a, err_a, inputs_a = _run(ref, c)
        out_b, err_b, inputs_b = _run(cand, c)
        case_atol = atol + (peak_atol * _peak(out_a) if peak_atol and err_a is None else 0.0)
        d = _Diff()
        if err_a or err_b:
            if err_a and err_b:
                if type(err_a) is not type(err_b):
                    d.add(f"both raise, but {type(err_a).__name__} vs {type(err_b).__name__}")
            else:
                who, e = ("reference", err_a) if err_a else ("candidate", err_b)
                d.add(f"only {who} raises: {type(e).__name__}: {e}")
        else:
            _compare(out_a, out_b, "output", rtol, case_atol, strict_types, d)
        if check_mutation:
            (args_a, kw_a), (args_b, kw_b) = inputs_a, inputs_b
            for j, (x, y) in enumerate(zip(args_a, args_b)):
                _compare(x, y, f"arg {j} after call (in-place mutation?)", rtol, atol, strict_types, d)
            for k in kw_a:
                _compare(kw_a[k], kw_b[k], f"kwarg {k!r} after call (in-place mutation?)",
                         rtol, atol, strict_types, d)
        worst_abs, worst_rel = max(worst_abs, d.max_abs), max(worst_rel, d.max_rel)
        status = "ok  " if not d.problems else "FAIL"
        note = " (both raise " + type(err_a).__name__ + ")" if (err_a and err_b and not d.problems) else ""
        lines.append(f"  [{status}] {label}{note}  max_abs={d.max_abs:.2e} max_rel={d.max_rel:.2e}")
        for p in d.problems[:5]:
            lines.append(f"         - {p}")
        n_fail += bool(d.problems)

    if verbose:
        tol = f"rtol={rtol:g}, atol={atol:g}" + (f", peak_atol={peak_atol:g}" if peak_atol else "")
        print(f"equivalence: {len(cases) - n_fail}/{len(cases)} cases match "
              f"({tol}); worst abs err {worst_abs:.2e}, worst rel err {worst_rel:.2e}")
        print("\n".join(lines))
    return n_fail == 0


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def _time_per_call(fn: Callable, c: Case, repeat: int, min_time: float, max_time: float) -> float:
    def call():
        fn(*c.args, **c.kwargs)

    t0 = timeit.default_timer()
    call()  # warm-up, and tells us if one call is already very slow
    first = timeit.default_timer() - t0
    if first > max_time:
        return first
    number = max(1, int(min_time / max(first, 1e-9)))
    times = timeit.repeat(call, number=number, repeat=repeat)
    return min(times) / number


def bench(ref: Callable | None, cand: Callable, c: Case | tuple, *, repeat: int = 5,
          min_time: float = 0.2, max_time: float = 20.0, verbose: bool = True) -> tuple[float | None, float]:
    """Time ref vs cand on one (realistically sized) case. Returns (t_ref, t_cand) in seconds/call.

    Pass ref=None to time only the candidate, e.g. at full size when the
    original is too slow to run there (benchmark both at a smaller size and
    extrapolate the original from its complexity, saying so in the report).
    The inputs are NOT copied between calls; if a function mutates its input,
    benchmark a wrapper that copies first (and say so in the report).
    If a single call takes longer than max_time, it is timed once.
    """
    c = c if isinstance(c, Case) else Case(args=tuple(c))
    t_ref = _time_per_call(ref, c, repeat, min_time, max_time) if ref is not None else None
    t_new = _time_per_call(cand, c, repeat, min_time, max_time)
    if verbose:
        if t_ref is None:
            print(f"benchmark {c.label or ''}: candidate {_fmt(t_new)}")
        else:
            speedup = t_ref / t_new if t_new > 0 else math.inf
            print(f"benchmark {c.label or ''}: reference {_fmt(t_ref)}, candidate {_fmt(t_new)}, "
                  f"speedup x{speedup:,.1f}")
    return t_ref, t_new


def _fmt(t: float) -> str:
    for unit, scale in (("s", 1), ("ms", 1e-3), ("µs", 1e-6), ("ns", 1e-9)):
        if t >= scale:
            return f"{t / scale:.3g} {unit}"
    return f"{t / 1e-9:.3g} ns"
