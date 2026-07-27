# Contributing

Contributions that improve the implementation, documentation, portability, tests, or developer experience are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Required checks

```bash
python scripts/verify_repository.py
python scripts/verify_notebook_contract.py
pytest -q
ruff check src tests scripts app.py
```

Keep pull requests focused and include tests for behavior changes. Update documentation when changing commands, configuration, dependencies, output schemas, or public APIs. Changes to the paper methodology or published result records should be discussed with the maintainers before implementation.
