---
name: algo-simplify
description: Simplify Python and NumPy code and algorithms while proving the new version still behaves like the old one. Use this whenever the user asks to simplify, clean up, refactor, tidy, untangle, speed up, vectorize, or reduce the complexity of Python/NumPy code, asks what a convoluted function actually computes, or wonders whether an algorithm has a simpler equivalent form, closed form, or library call (np.convolve, np.fft, searchsorted, cumsum, scipy.signal) — even if they never say "simplify". Covers readability refactors, big-O improvements, loop-to-NumPy vectorization, math identities, and re-expressing an algorithm in its simplest equivalent form, always backed by an equivalence check against the original before finishing.
---

# Algorithm & code simplification (Python / NumPy)

The goal is code that is shorter, clearer and usually faster, **computing the same thing as before**, with evidence the user can trust. An unverified simplification just moves work onto the user: they now have to re-check it themselves. So the safety net comes before the clever rewrite.

## 1. Pin down the contract before touching anything

Read the code and its call sites (grep for the function name). Note, briefly:

- **Inputs**: types, shapes, dtypes and value ranges that are actually used. Do callers pass lists or arrays? Ints or floats? Do they pass sorted data?
- **Outputs**: list vs tuple vs ndarray vs scalar, shape, dtype.
- **Side effects**: does it mutate its arguments, print, write files, touch globals, or consume random state?
- **Edge behaviour**: empty input, length 1, NaN/inf, ties, window larger than the data, invalid arguments.

Don't guess these: probe the original with a few lines of throwaway code before writing anything new. Check:
- the output type and dtype for list, int, float32 and float64 inputs (NumPy 2 promotion keeps float32 as float32 in places you wouldn't expect)
- what happens on empty input
- what happens on an invalid argument

It takes a minute and regularly turns up behaviour nobody knew about.

Accidental behaviour is still part of the contract. Returning a list, keeping NaNs, or picking the first of several ties may all be relied on by callers. Preserve it by default. If you think it should change, raise it as a flagged suggestion rather than changing it silently.

Simplify what was asked (a function, a file) rather than wandering through the repo.

## 2. Build the safety net first

Before editing, make sure you can tell old from new apart.

- **Existing tests.** If tests cover the code, run them now. They must pass before and after.
- **Differential check.** Use this when there are no tests, or when they don't exercise edge cases. Copy the original function verbatim into a scratch check script as the reference. Then compare it with the new version on many inputs.

The bundled helper `${CLAUDE_SKILL_DIR}/scripts/equiv.py` does the bookkeeping. `${CLAUDE_SKILL_DIR}` is this skill's base directory. It deep-copies inputs and compares nested outputs, types, dtypes and NaN positions. It also checks that exceptions match and that inputs are left in the same state, and it includes a benchmark:

```python
import sys; sys.path.insert(0, "${CLAUDE_SKILL_DIR}/scripts")
from equiv import compare, bench, edge_arrays, case

rng = np.random.default_rng(0)
cases  = [case(x, 7, label=name) for name, x in edge_arrays(rng)]      # generic edge cases
cases += [case(rng.standard_normal(5000) + 1e6, 101, label="long, DC offset")]  # realistic
assert compare(original_fn, new_fn, cases, rtol=1e-10)
```

Keep the reference copy and check script out of the user's code, for example in a temp dir. Delete them at the end.

**Choosing inputs.** Use `edge_arrays()` plus cases specific to this code. Examples: ties for argmax or nearest-match, values exactly on a threshold or window boundary, duplicates for searches, unsorted data if callers might pass it, integer inputs if callers use ints, and at least one realistically sized input.

**Choosing tolerance.** Reordering float operations (a vectorized sum, an FFT) changes the last few bits. For float64 rearrangements, start at `rtol=1e-10`. When some outputs are near zero while others are large (spectra, residuals, mean-removed data), relative error at those near-zero entries is meaningless. Use `peak_atol=1e-12`, which sets an absolute tolerance per case relative to the largest value in that case's output. If you need a looser tolerance than that, treat it as a finding, not a setting: say why in the report and run the accuracy check below. Never loosen a tolerance just to make the check pass.

**When the versions differ beyond tolerance, find out which one is closer to the truth.** Don't just report "they differ"; build a better reference and measure both against it. Options include:
- running the original on float64-cast input when the original used float32
- an input with a known analytic answer, such as a pure tone on an FFT bin or a constant signal
- a slow, obviously-correct formulation

Often the rewrite turns out to be *more* accurate, which is worth showing with numbers. Sometimes it is a real regression that a looser tolerance would have hidden.

## 3. Find simplifications, biggest wins first

Work through four lenses in this order. An earlier lens often makes later ones moot: there's no point polishing a loop you're about to delete.

1. **Conceptual: what does this actually compute?** Name it. Many hand-rolled loops are a known operation in disguise:
   - a convolution or correlation
   - a DFT, periodogram or filter recurrence
   - a cumulative sum or running extremum
   - a histogram or group-by
   - a nearest-neighbour or sorted search
   - a matrix product
   
   Once named, the simplest form is usually one library call or a one-line formula. Naming it is also often the most useful thing you can tell the user.
2. **Algorithmic: is there redundant work?** Examples:
   - nested loops over the same data → sort + `searchsorted`, or two pointers (O(n²) → O(n log n))
   - recomputed window sums → running or cumulative sums
   - membership tests on lists → `set`/`dict`
   - full sort for top-k → `argpartition`
   - recursion without memoisation → `lru_cache` or DP
   - `inv(A) @ b` → `solve(A, b)`
3. **Vectorization & math.** Replace Python-level loops over elements with whole-array operations: broadcasting, boolean masks, `np.where`, ufunc `accumulate`/`reduce`, `np.bincount`/`np.add.at` for scatter-adds, `@`/`einsum` for contractions. Use identities and closed forms such as geometric sums, `logsumexp`, `log1p`, `hypot` and Horner's rule.
4. **Readability.** Clear names, guard clauses instead of deep nesting, and `zip`/`enumerate` instead of index juggling. Remove dead code, unused variables and redundant flags. Keep comments that explain *why*, and add a one-line docstring saying what the function computes if the name doesn't.

`references/patterns.md` catalogues common before → after rewrites with their pitfalls, grouped by:
- signal processing
- search and sort
- aggregation
- linear algebra and math
- Python structure

Read the relevant section when choosing a replacement. Several "obvious" drop-ins differ in subtle ways, such as `scipy.signal.find_peaks`, `np.convolve(mode="same")` and `np.histogram`.

**Know when to stop.** If code is already clear and fast enough, say so and change little: churn costs review time. Prefer the simplest correct version that isn't slower at realistic sizes. Don't trade a readable loop for an opaque `einsum` to save microseconds. Don't add a dependency (scipy, numba) the project doesn't already use without mentioning it. A loop with a true sequential dependency is sometimes best left as a loop; note the `lfilter`/numba options instead.

## 4. Traps to check explicitly

These are the usual ways an "equivalent" rewrite quietly isn't. Add a test case for each one that plausibly applies.

- **Integer overflow.** Python ints are unbounded, but NumPy int64 (or int32) wraps silently. Vectorized sums and products of large ints can overflow where the loop didn't.
- **dtype drift.** Int input may now give float output, float32 may be upcast, or bool arithmetic may change. A list of Python floats may become an array of `np.float64`.
- **NaN / inf.** Python `max`/`sorted` and NumPy treat NaN differently. Comparisons with NaN are always False. A `cumsum` lets a single NaN or inf poison every later output, while a direct window sum keeps it local. An FFT-based convolution spreads it everywhere.
- **Empty and tiny inputs.** `np.max([])` raises, `np.convolve` raises on empty input, and `np.convolve(mode="same")` returns `max(len(a), len(v))` samples when the signal is shorter than the kernel.
- **Float accumulation.** Running sums via `cumsum` drift on long signals with a large offset. `E[x²] − E[x]²` cancels catastrophically. Test long, offset inputs, not just short standard-normal ones.
- **Ties and ordering.** `argmax` returns the first maximum, while a loop using `>=` returns the last. Use `kind="stable"` sorts when index order matters.
- **Return type and aliasing.** Watch for list vs ndarray: `out * 2` repeats a list but doubles an array. Returning a view the caller later mutates is another risk, as is mutating an input the original left alone (or vice versa).
- **Boundaries.** Window edges, convolution modes, inclusive vs exclusive thresholds (`>` vs `>=`), and off-by-one in window width (e.g. `width // 2` on each side gives `width + 1` points for even widths).
- **Memory.** Broadcasting an n×m intermediate can turn a slow loop into an out-of-memory crash at realistic n. Estimate peak array size before choosing broadcasting.

If an exact match is impossible or undesirable, show the user the input where the versions differ, say which you believe is right, and let them decide. This covers cases where the original has a bug, or where it is less accurate than the new method.

## 5. Edit, verify, measure

1. Apply the change in the user's file. Keep the public interface (name, signature, return type, exceptions for valid input) unless the user asked otherwise.
2. Re-run the tests and the differential check. Everything must pass, or the remaining differences must be deliberate and listed.
3. When speed was a goal or the change is algorithmic, benchmark old vs new with `bench()` at a realistic size. If the original is too slow to run at full size, benchmark both at a smaller size and extrapolate the original from its complexity, saying so. Then time the new version alone at full size with `bench(None, new_fn, big_case)`.
4. Delete the scratch files.

## 6. Report

Keep it short and use this shape:

**What it computes.** One or two sentences naming the operation in plain terms. For conceptual rewrites, give the intuition first, then the brief derivation that shows the equivalence. For example: "the inner loop is Σₖ h[k]·x[n−k], so it's a convolution of x with h".

**Changes**

| Change | Lens | Why | Cost before → after |
|---|---|---|---|

**Verification.** Include:
- tests run, and whether they pass
- the differential check: number of cases, which edge cases, max abs/rel error, and the tolerance with a reason
- the benchmark: input size, old vs new time

**Behaviour differences & caveats.** List anything not identical, or deliberately left alone, and why. This includes suggestions you did *not* apply because they would change the interface. Write "None" if there are none.
