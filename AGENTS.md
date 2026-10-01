# AGENTS.md: standing instructions for the remote agent

You are working on the machine that holds the **real data** (SEM images, `.oas` files, the
metadata CSV). This repository is developed elsewhere by another agent that has **no access
to the data**. That agent sends you work as task files in `docs/tasks/`. Communication is
**one-way**: you receive tasks through `git pull`, and your results reach the other agent
only through the user.

## Rules

1. **Do the task the user names** (`docs/tasks/TNNN-*.md`), exactly as written. If something
   in it doesn't fit the real data, do the sensible thing and state it clearly in your report.
2. **Never commit and never push.** The user receives new tasks with `git pull`, which must
   always succeed. So:
   - never modify, rename or delete files tracked by git (`git ls-files`), including this
     file, `CLAUDE.md`, `docs/SPEC.md`, the task files, notebooks and existing code/tests;
   - create only the new files your task lists.
3. **Data stays out of the repo.** Never copy data files or real values (file names,
   coordinates, CSV rows) into repo files, except into `docs/remote/`, which is gitignored and
   never leaves this machine. In code and tests, use made-up values in the real format.
4. **Reports:** write each report to `docs/remote/TNNN-report.md`. Finish by showing the user
   the report's summary section. Nothing leaves this machine: the user reads it and decides
   what, if anything, to tell the other agent in their own words.
5. **Code style:** follow the "Development approach" and "Key rules" in `CLAUDE.md` (small,
   simple, readable functions; nm units; coordinate frames as in `docs/SPEC.md` §3).
   Ignore `CLAUDE.md` parts about committing, pushing or writing tasks; they are for the other agent.
6. **Before finishing:** run `pip install -e ".[dev]"` and `pytest`. Everything must pass,
   including the contract tests for your code (`tests/test_*_contract.py`). Run those with the
   data folder set as the task describes.
