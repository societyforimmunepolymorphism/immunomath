# Contributing to ImmunoMath

Thank you for considering a contribution. ImmunoMath is maintained by the
Society for Immune Polymorphism (SIP). Because it is intended as a reference
implementation for benchmarking and future clinical validation, we hold
contributions to a high standard of mathematical correctness, testing, and
documentation.

## Development setup

ImmunoMath requires **Python 3.12+**.

```bash
git clone https://github.com/<your-fork>/immunomath.git
cd immunomath
uv sync --extra dev --extra docs     # creates .venv from the committed uv.lock
```

Run tools through uv (`uv run pytest`, `uv run ruff check .`) or activate
`.venv`. If you prefer pip, create a virtual environment first
(`python -m venv .venv && source .venv/bin/activate`) and run
`pip install -e ".[dev,docs]"`.

`uv.lock` is committed. If you change dependencies in `pyproject.toml`, run
`uv lock` and commit the result; CI-style check: `uv lock --check`.

## The fork-and-pull-request workflow

ImmunoMath follows the NMDP Bioinformatics fork method
(<https://github.com/nmdp-bioinformatics/dash/blob/master/README.git.fork.method.md>).
In short:

> Note: the default branch may be `main` rather than `master`.

1. Log into GitHub and browse to
   <https://github.com/societyforimmunepolymorphism/immunomath>; click **Fork**.
2. Clone **your fork** and enter it:

   ```bash
   git clone https://github.com/<your-username>/immunomath.git
   cd immunomath
   ```

3. Add the SIP repository as the `upstream` remote:

   ```bash
   git remote add upstream https://github.com/societyforimmunepolymorphism/immunomath
   ```

4. Keep your fork's default branch up to date with upstream before starting:

   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```

5. Create a feature branch:

   ```bash
   git checkout -b my-feature
   ```

6. Make your changes, pulling in any upstream updates as you go, then commit:

   ```bash
   git pull upstream main
   git commit -m "Describe your change"
   ```

7. Push the feature branch to **your fork**:

   ```bash
   git push origin my-feature
   ```

8. Open a pull request from your fork's feature branch against
   `societyforimmunepolymorphism/immunomath:main`. A maintainer will review it.
9. After merge, delete the remote and local feature branches:

   ```bash
   git branch -d my-feature
   ```

Do not push directly to `main`, and do not force-push shared branches.

## Quality gates

All of the following must pass locally and in CI before a pull request is
merged:

```bash
ruff check .
ruff format --check .
mypy
pytest --doctest-modules src/immunomath tests
mkdocs build --strict
```

## The documentation quality gate

Every new public feature **must** ship with:

1. Unit tests (including the relevant edge cases from the testing checklist).
2. API documentation (NumPy-style docstring with a runnable example).
3. At least one user-facing example or tutorial reference.
4. An explanation of numerical assumptions and edge cases.
5. A `CHANGELOG.md` entry.

We prefer **excellent documentation for a smaller set of correct features** over
broad functionality with incomplete explanations.

## Mathematical-correctness expectations

- Cross-check new metrics against an authoritative reference (SciPy,
  scikit-learn, or a hand-derivation in the docstring/test).
- Add property-based tests (Hypothesis) for invariants, not just example cases.
- Never write a test that merely locks in an implementation's current (possibly
  wrong) output. Tests must encode the *intended* mathematics.
- Never silently normalize, clip, or trim. Edge cases are represented
  explicitly (`+inf`, `KeyError`, residual mass), not hidden.

## Code style

- Full type annotations; `mypy --strict` must pass.
- NumPy-style docstrings, enforced by Ruff's pydocstyle rules.
- Keep the dependency footprint modest and pin new dependencies to known,
  maintained packages.
