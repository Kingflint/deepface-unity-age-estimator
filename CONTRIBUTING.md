# Contributing

Thanks for taking the time to contribute!

## Local setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

## Style

- Format with `ruff format`.
- Lint with `ruff check`.
- Keep public functions typed where it improves readability.
- Don't add new heavy dependencies without discussion.

## Pull requests

- Reference the issue you are addressing in the PR description.
- Add or update tests under [tests/](tests/) for any behaviour change.
- Keep commits focused and descriptive.

## Tests

The DeepFace library and TensorFlow are stubbed out by
[tests/conftest.py](tests/conftest.py) so the suite runs in a few
seconds. If you add a feature that touches the real model, gate the
test with `@pytest.mark.slow`.
