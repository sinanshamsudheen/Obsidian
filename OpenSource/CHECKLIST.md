
# Open-Source Contribution Final Checklist

## 0. Sanity check (30 seconds)

- ☐ Is this a **bugfix / small improvement** → PR OK
    
- ☐ Is this a **feature / refactor / behavior change** → Issue exists or approved
    

If you skip this and guess wrong, expect rejection.

---

## 1. Read the repo’s rules (5 minutes)

- ☐ `CONTRIBUTING.md`
    
- ☐ `README.md` (only for workflow expectations)
    
- ☐ Code of Conduct (know what **not** to say in reviews)
    

If `CONTRIBUTING.md` is missing → CI is the law.

---

## 2. Read CI like a contract (non-negotiable)

Open:

- ☐ `.github/workflows/*.yml`
    
- ☐ `.gitlab-ci.yml` / `azure-pipelines.yml` (if present)
    

Extract:

- ☐ Python / Node / Go version
    
- ☐ Exact test command
    
- ☐ Linters enforced
    
- ☐ Formatters enforced
    
- ☐ Coverage thresholds (if any)
    

**Rule:** If CI runs it, you must run it locally.

---

## 3. Detect linters & formatters (don’t assume)

Search for:

- ☐ `pyproject.toml`
    
- ☐ `setup.cfg`
    
- ☐ `tox.ini`
    
- ☐ `.ruff.toml`
    
- ☐ `.flake8`
    
- ☐ `.pre-commit-config.yaml`
    
- ☐ `Makefile`
    

Confirm:

- ☐ Ruff? (`ruff check`, `ruff format`)
    
- ☐ Black / isort?
    
- ☐ Custom rules or ignores?
    
- ☐ Line length enforced?
    

Never guess formatting.

---

## 4. Pre-commit (if present, it’s mandatory)

- ☐ `pip install pre-commit`
    
- ☐ `pre-commit install`
    
- ☐ `pre-commit run --all-files`
    

If you skip this, CI failure is on you.

---

## 5. Tests (run what CI runs)

Identify from CI / config:

- ☐ `pytest`
    
- ☐ `tox`
    
- ☐ `nox`
    
- ☐ `make test`
    

Before pushing:

- ☐ All tests pass locally
    
- ☐ No skipped tests unless explicitly allowed
    

“No time to run tests” = no PR.

---

## 6. Match existing code style (very important)

Before coding:

- ☐ Read 3–5 **recent merged PRs**
    
- ☐ Observe function size & naming
    
- ☐ Observe error handling style
    
- ☐ Observe logging vs exceptions
    
- ☐ Observe docstring tone
    

Consistency > cleverness.

---

## 7. Architecture respect check

- ☐ Did I touch only relevant files?
    
- ☐ Did I avoid unrelated refactors?
    
- ☐ Did I follow existing layering?
    
- ☐ Did I avoid moving code “just because”?
    

If the repo is messy, stay messy **in the same way**.

---

## 8. Scope control (PR survivability)

- ☐ PR does **one thing**
    
- ☐ No drive-by refactors
    
- ☐ No “while I was here” changes
    
- ☐ Diff is easy to review
    

Small PRs get merged. Big ones get ignored.

---

## 9. Commit hygiene

- ☐ Commit messages follow repo pattern
    
- ☐ Commits are clean (no WIP / fix typo spam)
    
- ☐ Squashed if expected
    
- ☐ Signed commits if required
    

People judge you by commits before code.

---

## 10. Final pre-push checklist

Run locally:

- ☐ `pre-commit run --all-files`
    
- ☐ Linters (ruff/flake8/etc.)
    
- ☐ Formatters
    
- ☐ Tests
    

If CI fails after this → something is wrong with CI, not you.

---

## 11. PR description (reviewer-friendly)

- ☐ Clear problem statement
    
- ☐ What changed and why
    
- ☐ Why this approach
    
- ☐ Linked issue (if any)
    
- ☐ Notes for reviewers (edge cases, risks)
    

Your PR text should **reduce reviewer thinking**.

---

## One-line rule to remember

> **CI decides, consistency wins, scope survives.**

