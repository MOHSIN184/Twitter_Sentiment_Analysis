# Contributing

Thanks for improving this project.

## Setup

Create a virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Tests

Run the test suite before opening a change:

```bash
pytest -q
```

## Code Style

- Keep code modular, typed where useful, and easy to test.
- Add docstrings to public functions.
- Prefer `pathlib` for filesystem paths.
- Keep API responses stable unless the README and tests are updated together.

## Data and Secrets

- Do not commit the raw dataset.
- Do not commit `.env` files or secrets.
- Keep generated prediction CSVs and log files out of version control.
