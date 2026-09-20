# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - Initial release

The first working version of AI Notes Generator: a dependency-free,
from-scratch classical-NLP pipeline that turns plain-text lectures,
articles, or textbook chapters into structured Markdown study notes.

Included in the initial commit:

- **Segmentation** — markdown-heading detection, plus a TF-IDF cosine
  similarity topic-shift detector for un-headed text.
- **Heading generation** — synthesizes a short heading for headless
  sections from the section's top TF-IDF keyword, paired with an adjacent
  in-sentence word.
- **Bullet extraction** — a from-scratch TextRank implementation
  (sentence-similarity graph + power iteration) combined with a TF-IDF
  sentence-importance score to pick the 3–6 most salient sentences per
  section.
- **Glossary extraction** — definitional-pattern matching (`"X is defined
  as ..."`, `"X refers to ..."`, `"X is a/an ..."`) plus frequent
  capitalized-term detection.
- **Markdown rendering** of sections, bullets, and a glossary.
- A CLI entry point: `python generate_notes.py input.txt output.md`.
- `tests/test_generate_notes.py` — a unit test suite covering
  segmentation, bullet extraction, glossary extraction, full-pipeline
  Markdown output, and the CLI.
- `examples/sample_input.txt` / `examples/sample_output.md` — a worked
  example (photosynthesis / cellular respiration / enzymes).
- A GitHub Actions workflow running the test suite on Python 3.11 and 3.12.
- MIT license.

## [Unreleased]

- `CONTRIBUTING.md` with test/style/PR guidelines.
- Additional unit tests for glossary `max_terms` capping and CLI error
  paths (wrong argument count, missing input file).
- Friendlier error message for non-UTF-8 input files (previously raised
  an unhandled `UnicodeDecodeError` traceback).
- Troubleshooting/FAQ section in the README.
