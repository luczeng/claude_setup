## How to work
### Think before doing
Don't assume. Don't hide confusion. Surface tradeoffs.
- Think before coding:Before a non-trivial change, state your assumptions and plan in a few lines. If the request is ambiguous, ask instead of guessing.
- If there are several reasonable approaches, propose the simplest and mention the alternative in one line.
- Decide what "done" means up front (usually: specific tests pass), then work until it is verified.
- Check in if a task grows well beyond what I asked for.

### Communicating
- State high level points first: intuitions, core ideas, impact... Details come second.
- Less is better. A well formulated sentence is better than 3 over complicated ones.

### Simplicity first
- Write the minimum code that solves the problem asked. No speculative features, options, parameters or hooks "for later".
- Prefer plain functions. Use a class only when it holds real state shared by several functions; use `@dataclass` for plain data containers.
- No factories, registries, plugin systems, abstract base classes, metaclasses or custom decorators unless I ask.
- Don't abstract before the third repetition; a little duplication beats the wrong abstraction.
- Prefer the standard library and dependencies the project already has. Ask before adding a new one.
- Configuration: function arguments and a few module-level constants. No config frameworks (Hydra etc.) unless the project already uses one.
- Let errors surface. Validate inputs where data enters (loading, user input); don't wrap code in try/except just to log and continue.
- If your solution is long, look for a shorter one before presenting it.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## How to code
### Code structure
- Small functions that do one thing. If you can't name a function clearly, split it.
- Keep I/O at the edges: load → pure transformations → model → save/plot. Transformations take inputs and return outputs; they don't read files or mutate their arguments (no `inplace=True`).
- Split modules by role, e.g. `data.py`, `features.py`, `model.py`, `plots.py`, but don't create modules or folders before there's code to put in them.
- Type hints on function signatures, kept simple. One-line docstrings on public functions; comments explain why, not what.
- Use `pathlib.Path`; no hard-coded absolute paths.

### Testing
- pytest only. Tests mirror modules: `src/pkg/features.py` → `tests/test_features.py`.
- Every new function with logic gets tests: one normal case plus the relevant edge cases (empty input, NaN/missing values, single row, unexpected dtype). Skip trivial glue and plotting code.
- Build tiny DataFrames/arrays inline in the test; don't depend on real data files. Use fixtures only when several tests share the setup.
- Test behaviour (outputs, shapes, dtypes, invariants), not implementation details. Mock only network and external services.
- For bugs: first write a failing test that reproduces it, then fix.
- Run the relevant tests before saying you're done and report the result. If you couldn't run them, say so.
- Never delete, skip or loosen a failing test to make it pass without asking me.

### Tooling
- Use uv for running things: uv run pytest, uv run ruff, etc..

## Reporting back
- When done: a few lines on what changed, why, and how it was verified. No long recaps.

