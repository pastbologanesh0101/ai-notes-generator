# Contributing

Thanks for considering a contribution to AI Notes Generator. This project is
intentionally small (a single-file, pure-stdlib pipeline), so the bar for
contributions is: does it make the notes better, or the code clearer, without
adding a dependency or an LLM/network call?

## Running the tests

```bash
pip install pytest    # only needed for the pytest runner, not for the tool itself
pytest tests/ -v
# or, with zero third-party dependencies at all:
python -m unittest discover -s tests -v
```

Both runners execute the same test suite in `tests/test_generate_notes.py`.
Please run one of them before opening a pull request.

## Code style

- Pure Python standard library only — no new third-party dependencies. The
  whole point of this project is that it works with nothing but `python3`.
- Keep `generate_notes.py` typed (`from __future__ import annotations` +
  type hints on public functions) and documented with a short docstring per
  function.
- Favor small, testable functions over one large pipeline function — each
  pipeline stage (segmentation, heading generation, bullet extraction,
  glossary extraction, rendering) should stay independently unit-testable.
- Match the existing formatting (4-space indents, double-quoted strings,
  ~88-100 col lines). There's no enforced formatter/linter yet; just follow
  the surrounding code.

## Submitting changes

1. Fork the repo and create a branch for your change.
2. Add or update tests in `tests/test_generate_notes.py` for any behavior
   change — bug fixes and new features should both come with a test that
   would fail without the fix.
3. Make sure the full test suite passes locally (see above) and that the
   CI workflow (`.github/workflows/tests.yml`) would pass — it runs the
   same commands against multiple Python versions.
4. Open a pull request describing what changed and why. Include a short
   before/after example if the change affects generated output.

## Reporting bugs

Please include a minimal input snippet that reproduces the issue and the
generated output you got vs. what you expected. Since this is a
from-scratch NLP pipeline (no LLM), most "bugs" are edge cases in
tokenization, segmentation, or the heuristics — a concrete example is far
more useful than a description.
