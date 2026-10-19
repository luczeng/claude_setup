# Rewrite patterns: Python / NumPy

A catalogue of common before → after simplifications. Each entry lists the **pitfall** that makes the naive drop-in not equivalent. Check those pitfalls with the differential check rather than trusting the table.

## Contents
1. Signal processing & windows
2. Search, sort & matching
3. Aggregation & grouping
4. Linear algebra & math identities
5. Python structure & readability
6. When *not* to vectorize

---

## 1. Signal processing & windows

### Moving sum or average with centred window `[i-h, i+h]`, edge windows shrink
- **Shrinking edges:** `full = np.convolve(x, np.ones(2*h+1))`, then `sums = full[h:h+n]`. Divide by `counts = np.minimum(i+h, n-1) - np.maximum(i-h, 0) + 1`, where `i = np.arange(n)`.
- **Interior only:** `np.convolve(x, np.ones(w)/w, mode="valid")`.
- **Exact per-window:** `np.lib.stride_tricks.sliding_window_view(np.pad(x, h), 2*h+1).sum(-1)`. This is O(n·w) work but vectorized.
- **O(n) for large windows:** `c = np.concatenate(([0.], np.cumsum(x)))`, then `c[hi] - c[lo]`.

Pitfalls:
- `np.convolve(mode="same")` returns `max(len(x), len(k))` samples when the signal is shorter than the kernel. Slice `mode="full"` output instead.
- `np.convolve` raises `ValueError` on empty input.
- `width // 2` on each side gives `width + 1` points for even widths. Preserve the original's quirk.
- The cumsum route loses precision on long signals with a large DC offset. Subtract a constant first and add it back. It also lets one NaN or inf poison every later output (`inf − inf = nan`).
- Direct `np.convolve` keeps NaN/inf local to the windows that contain them. `scipy.signal.fftconvolve` spreads them everywhere and adds ~1e-16·max|x| noise.

### FIR filter `y[n] = Σₖ b[k]·x[n−k]`
- → `np.convolve(x, b)[:len(x)]` or `scipy.signal.lfilter(b, 1.0, x)`.
- Pitfall: the original's handling of `n − k < 0` (zero-padding vs skipping vs wrap-around).

### IIR / linear recurrence `y[n] = x[n] + a·y[n−1]`
- → `scipy.signal.lfilter([1.0], [1.0, -a], x)`.
- Pitfalls: initial condition (`zi`). Sign convention of the `a` coefficients (denominator is `1 − a z⁻¹`).

### Hand-written DFT (double loop with cos/sin)
- → `np.fft.fft` / `np.fft.rfft` (real input, bins 0…n//2).
- Frequencies: `np.arange(n//2 + 1) * fs / n` or `np.fft.rfftfreq(n, d=1/fs)`.
- Pitfalls:
  - Sign convention: `e^{−2πikn/N}` is NumPy's forward transform.
  - Normalisation: NumPy's forward FFT is unnormalised.
  - One-sided doubling covers all bins except DC and, for even n, Nyquist: `psd[1:(n+1)//2] *= 2`.
  - FFT differs from direct sums by ~1e-13 relative and is usually *more* accurate. Use an `atol` scaled to the spectrum peak, because near-zero bins (e.g. DC after mean removal) have meaningless relative error.

### Periodogram / PSD
- One-sided periodogram with boxcar window → `scipy.signal.periodogram(x, fs, detrend="constant" or False, scaling="density")`. Welch averaging → `scipy.signal.welch`.
- Pitfall: scipy's defaults for `detrend`, `window`, `scaling` and `return_onesided` must match the original. It also returns arrays, not lists.

### Cross-correlation loops
- → `np.correlate(a, v, mode)` or `scipy.signal.correlate(..., method="fft")`. Lags: `scipy.signal.correlation_lags`.
- Pitfall: lag sign convention and which argument is conjugated or reversed.

### Local maxima
- `x[i] > x[i-1] and x[i] >= x[i+1]` → `np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])) + 1`.
- Pitfall: `scipy.signal.find_peaks` is **not** a drop-in.
  - It returns the middle of a flat plateau.
  - It requires a strict fall after the plateau.
  - For `[0, 1, 1, 2, 0]` the loop above reports index 1 and 3, but `find_peaks` reports only 3.

### Zero crossings, running extrema, block reductions
- Sign changes → `np.flatnonzero(np.diff(np.signbit(x)))`. Decide how exact zeros should count: `signbit(0.0)` is False but `signbit(-0.0)` is True.
- Running max → `np.maximum.accumulate(x)`. Sliding max → `scipy.ndimage.maximum_filter1d`; check its edge `mode`.
- Block mean (downsample by k) → `x[: len(x)//k*k].reshape(-1, k).mean(1)`. Decide what happens to the remainder.

---

## 2. Search, sort & matching

### Nearest value in a 1-D reference set (for each query)
- O(n·m) scan → sort once, then `pos = np.searchsorted(r_sorted, q)`. Compare neighbours `pos-1` and `pos` (clipped). Map back through `order = np.argsort(ref, kind="stable")`. Total cost O((n+m) log m).
- Pitfalls:
  - The reference may be unsorted.
  - **Tie rules**: a strict `<` scan returns the *lowest original index* among equally close candidates. That includes duplicates in `ref` and queries exactly midway between two values. With a stable sort, the lowest original index of a run of equal values is at `np.searchsorted(r_sorted, value, "left")`.
  - Empty reference.
  - Inclusive vs exclusive distance threshold.
  - NaN in the reference: sorting moves it to the end, while a scan may get stuck on it.
- n-D nearest neighbour → `scipy.spatial.cKDTree(ref).query(q)`. Tie behaviour differs, so check.

### Membership, counting, uniqueness
- `if v in some_list` inside a loop → build a `set` once, or use `np.isin(a, b)`.
- Counting → `collections.Counter`, `np.unique(a, return_counts=True)`, or `np.bincount` for small non-negative ints.
- De-duplicate preserving first occurrence → `dict.fromkeys(seq)`, or `np.unique(a, return_index=True)` then sort the indices.

### Top-k and ranking
- Full sort for top-k → `idx = np.argpartition(x, -k)[-k:]`, then `idx[np.argsort(x[idx])[::-1]]`.
- Pitfall: order among ties differs from a stable full sort.

### All pairs satisfying a condition
- Two-sum, pairs within distance d → sort + two pointers, `searchsorted` for range bounds, or a dict of complements.
- Broadcasting `a[:, None] - b[None, :]` is O(n·m) **memory**. At 200k × 50k float64 that is 80 GB. Use it only for small inputs.

### First index satisfying a condition
- → `np.argmax(mask)`.
- Pitfall: if no element matches, `argmax` returns 0. Check `mask.any()` first, or use `next((i for i, v in enumerate(seq) if cond(v)), -1)`.

---

## 3. Aggregation & grouping

### Group sums, scatter-adds, histograms
- Group sums → `np.bincount(labels, weights=values, minlength=k)`.
- Scatter-add with repeated indices → `np.add.at(out, idx, vals)`. Plain `out[idx] += vals` silently counts duplicates **once**.
- Histogram → `np.histogram`. Pitfall: its **last bin is closed** `[a, b]` while the others are half-open. A manual floor-binning loop puts `x == hi` out of range.
- Fixed-width bins → `np.bincount(((x - lo) // width).astype(int), minlength=nbins)`. Clip or mask out-of-range values first.

### Cumulative and pairwise operations
- Prefix sums → `np.cumsum`. Differences → `np.diff`. Prefix products → `np.cumprod`. Watch overflow and underflow.

### Mean and variance
- → `np.mean`, `np.var(x, ddof=…)`.
- Pitfalls:
  - Match `ddof`: a population loop is `ddof=0`, a sample loop is `ddof=1`.
  - Never use `E[x²] − E[x]²` on offset data.
  - Python `statistics.variance` is `ddof=1`.

### Max and argmax with NaN
- Python `max([nan, 1.0])` returns nan but `max([1.0, nan])` returns 1.0: the result depends on order. `np.max` returns nan, and `np.nanmax` ignores NaN. Preserve whichever behaviour the callers saw.

---

## 4. Linear algebra & math identities

### Linear algebra
- `np.linalg.inv(A) @ b` → `np.linalg.solve(A, b)`. This is faster and more accurate. For symmetric positive-definite A → `scipy.linalg.cho_factor` / `cho_solve`.
- Triple-loop matrix product → `A @ B`.
- General contractions → `np.einsum("ij,jk->ik", ...)` with `optimize=True` for 3+ operands. Prefer `@` or `tensordot` when they read better.
- Squared distances → `‖a‖² + ‖b‖² − 2 a·b`. Fast, but cancels for nearby points: clip at 0, and prefer `scipy.spatial.distance.cdist` when exactness matters.

### Polynomials
- Polynomial evaluation → Horner's rule.
- `np.polyval(p, x)` takes coefficients **highest power first**. `np.polynomial.polynomial.polyval(x, c)` takes them **lowest first**.

### Closed forms
- Geometric sum: Σ_{k=0}^{n−1} rᵏ = (1 − rⁿ)/(1 − r) when r ≠ 1, and n when r = 1.
- Σ_{k=0}^{n−1} k = n(n−1)/2.

### Numerically stable forms
- `log(sum(exp(x)))` → `scipy.special.logsumexp(x)`.
- `log(1+x)` → `np.log1p(x)`.
- `exp(x) − 1` → `np.expm1(x)`.
- `sqrt(x² + y²)` → `np.hypot(x, y)`.
- These are *more* accurate than the original, so differences at tiny or huge magnitudes are improvements. Report them as such.

### Integer arithmetic
- Python ints never overflow, but NumPy int64 sums wrap silently: `np.array([2**62, 2**62]).sum()` is negative. Use `dtype=object` or Python `sum` if values can be huge, or check the range.

---

## 5. Python structure & readability

### Loops and accumulation
- `for i in range(len(x)): ... x[i] ...` → iterate directly, or use `enumerate` / `zip`.
- Accumulator + `append` in a loop → comprehension, or vectorize the whole thing.
- Manual best-so-far tracking → `min(items, key=...)` / `max(...)`. Both return the **first** minimum, which matches a strict `<` scan.

### Control flow
- Deeply nested `if`s → guard clauses (`if not cond: continue` / `return`) or one combined condition.
- `if cond: return True` / `else: return False` → `return bool(cond)`, or `return cond` if it's already a bool.

### Cleanup
- Remove dead code: unused variables (`tmp = None`), unreachable branches, flags that are always the same value, and commented-out blocks. Check git history first if unsure.
- Repeated magic numbers → named constants.

### Functions and caching
- Boolean flag arguments that switch behaviour → keep them, because the signature is public, but give them a docstring. Suggest a clearer keyword name in the report instead of renaming silently.
- Pure recursive functions with repeated sub-calls → `functools.lru_cache`, or rewrite as an iterative DP.

### Dictionaries
- Dict boilerplate → `dict.get(k, default)`, `collections.defaultdict`, `setdefault`.

---

## 6. When *not* to vectorize

- **True sequential dependency** (each step needs the previous output and isn't a linear filter): leave the loop, or mention `numba.njit` if the project uses numba.
- **Tiny n with Python scalars**: NumPy call overhead (~1 µs per call) can make the vector version slower and less readable.
- **Memory blow-up**: an O(n·m) or O(n·w) intermediate at realistic sizes. Prefer a sort/search algorithm or a chunked loop.
- **Readability collapse**: a five-index `einsum` or chained fancy indexing that nobody can review. A clear loop with a docstring can be the simplest correct code.
