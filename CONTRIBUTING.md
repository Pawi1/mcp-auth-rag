# Contributing

## Setup

```bash
make dev     # creates app/.venv, installs deps
make test    # runs the test suite
```

## Before opening a PR

- `make test` passes (277 tests today — add tests for whatever you change,
  especially anything in `app/auth.py`/`app/oauth.py`/`app/main.py`'s
  `/mcp` handler; see [SECURITY.md](SECURITY.md) for why those files get
  extra scrutiny).
- New tools go in `app/server.py` (`whoami`, `rag_search`, `rag_list_documents`
  are the existing pattern to follow: a `Tool()` entry, a dispatch branch,
  `log_tool_call`). Domain logic that isn't "glue between a tool and the rest
  of the app" belongs in its own module (see `rag_*.py`), not inline in
  `server.py`.
- No new runtime dependencies without a good reason — this repo stays small
  enough to read in a sitting, not literally dependency-free.

## Reporting bugs

Functional bug → open an issue.
Security issue → see [SECURITY.md](SECURITY.md), not a public issue.

## Code style

Match what's already there: no type-annotation ceremony, docstrings only
where the *why* isn't obvious from the code, flat `from x import y` module
layout (no package nesting) — matches the original repo this was extracted
from, and keeps the diff between "read main.py" and "understand main.py"
as small as possible.
