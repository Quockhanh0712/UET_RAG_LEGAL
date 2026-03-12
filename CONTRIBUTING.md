# Contributing Guide

This repository uses a strict and lightweight Git workflow so both team members can work in parallel without conflicts.

## Branching

- Do not develop directly on `main`.
- Create feature branches from `main`:
  - `feat/a-<topic>` (member A)
  - `feat/b-<topic>` (member B)
  - `fix/<topic>` for bug fixes
  - `docs/<topic>` for documentation-only changes

## Commit Rules

- Use Conventional Commits:
  - `feat:`
  - `fix:`
  - `docs:`
  - `test:`
  - `chore:`
- Keep each commit scoped to one intent.
- Avoid `WIP` commits on shared branches.

## Pull Request Rules

- One PR should target one objective.
- Keep PR size manageable (prefer under 300-500 changed lines).
- Require one review approval from the other teammate before merge.
- Use squash merge to keep `main` history clean.

## Before Opening A PR

Run the minimum checks:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit -q
cd web_app\frontend
npm run build
```

Ensure you are not committing local artifacts or secrets:

```powershell
git status --short
git diff --cached --name-only
```

## What Must Not Be Committed

- `.env` and any secret-bearing file
- `.venv/`, `node_modules/`, frontend build outputs
- heavy data files (`*.parquet`, local vector DB files)
- machine-specific caches and logs

## Branch Protection (GitHub)

Set protection for `main`:

- Require pull request before merging
- Require 1 approval
- Dismiss stale approvals when new commits are pushed
- Require conversation resolution before merging
