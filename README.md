# AI Notes Generator

[![tests](https://github.com/pastbologanesh0101/ai-notes-generator/actions/workflows/tests.yml/badge.svg)](https://github.com/pastbologanesh0101/ai-notes-generator/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](.github/workflows/tests.yml)

Turn plain-text lectures, articles, or textbook chapters into structured,
hierarchical Markdown study notes — headings, extractive bullet points, and
a glossary of key terms.

**No LLM. No API calls. No network access.** Everything is classic NLP
implemented from scratch in pure Python (standard library only): TF-IDF,
cosine similarity, and a TextRank-style graph-ranking algorithm.

This is distinct from a meeting-transcript summarizer: the input here is
general lecture/article/chapter prose, and the output is hierarchical study
notes (sections → bullets → glossary), not a list of decisions or action
items.

## Table of contents

- [How it works](#how-it-works)
- [Usage](#usage)
- [Example](#example)
- [Running the tests](#running-the-tests)
- [Project layout](#project-layout)
- [Troubleshooting / FAQ](#troubleshooting--faq)
- [Limitations](#limitations)
- [License](#license)

## How it works

1. **Segmentation.**
   - If the input already has markdown-style `#`/`##`/... headings, those
     are used directly as section boundaries.
   - Otherwise, the text is split into paragraphs (on blank lines) and
     consecutive paragraphs are compared using **TF-IDF cosine similarity**.
     A similarity score that drops sharply below the running average
     signals a topic shift, and a new section starts there.

2. **Heading generation.** Sections that came from topic-shift detection
   have no heading yet. The most representative keyword in the section is
   found via TF-IDF, then paired with an adjacent word from the same
   sentence (so the phrase is never spliced across sentence boundaries) to
   form a short, readable heading.

3. **Bullet extraction.** Each section's sentences are scored two ways and
   combined:
   - A **TextRank-style** graph algorithm: sentences are nodes, edges are
     weighted by word-overlap similarity, and scores are computed with
     power iteration (a from-scratch PageRank-over-sentences), exactly the
     way the original TextRank paper describes it.
   - A **TF-IDF sentence-importance** score (sum of TF-IDF weights of a
     sentence's content words, length-normalized).

   The top 3–6 sentences (by combined score) become bullet points, shown in
   their original order.

4. **Glossary extraction.** Two signals feed the glossary:
   - **Definitional patterns**: `"X is defined as ..."`, `"X refers to ..."`,
     `"X is a/an ..."` — the subject and definition are captured directly.
   - **Frequent capitalized terms**: multi-occurrence capitalized
     words/phrases that aren't just sentence-initial common words (e.g.
     `Kubernetes`, `Mitochondria`) are added as terms.

5. **Rendering.** Sections and their bullets are rendered as Markdown
   headings and lists, followed by a `## Glossary` section.

## Usage

```bash
python generate_notes.py input.txt output.md
python generate_notes.py input.txt output.md --title "Chapter 4 Notes"
```

Reads plain text from `input.txt` and writes structured Markdown notes to
`output.md`. No dependencies beyond the Python standard library.

The optional `--title "My Title"` flag sets the `# ` heading at the top of
the generated document (default: `Study Notes`).

## Example

**Before** (`examples/sample_input.txt`, excerpt):

```
Photosynthesis is defined as the process by which green plants, algae, and
some bacteria convert light energy into chemical energy stored in glucose.
Chlorophyll, the pigment found in chloroplasts, absorbs sunlight and drives
the reaction. Photosynthesis produces oxygen as a byproduct, which is
released into the atmosphere. ...

Cellular respiration refers to the set of metabolic reactions that convert
the chemical energy stored in glucose into adenosine triphosphate, or ATP.
Mitochondria are the organelles primarily responsible for cellular
respiration in eukaryotic cells. ...

Enzymes are proteins that act as biological catalysts, speeding up chemical
reactions without being consumed themselves. Enzymes are defined as highly
specific molecules, each shaped to bind a particular substrate at its
active site. ...
```

**After** (`examples/sample_output.md`, generated with
`python generate_notes.py examples/sample_input.txt examples/sample_output.md`):

```markdown
# Study Notes

## Photosynthesis produces

- Photosynthesis is defined as the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in glucose.
- Photosynthesis produces oxygen as a byproduct, which is released into the atmosphere.
- Cellular respiration refers to the set of metabolic reactions that convert the chemical energy stored in glucose into adenosine triphosphate, or ATP.
- Mitochondria are the organelles primarily responsible for cellular respiration in eukaryotic cells.
- Cellular respiration consumes the oxygen produced by photosynthesis and releases carbon dioxide as a byproduct, forming a natural cycle between plants and animals.

## Enzymes lower

- Enzymes are proteins that act as biological catalysts, speeding up chemical reactions without being consumed themselves.
- Enzymes are defined as highly specific molecules, each shaped to bind a particular substrate at its active site.
- Enzymes lower the activation energy required for a reaction to proceed, which allows metabolic processes to occur quickly at body temperature.

## Glossary

- **Photosynthesis**: the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in glucose
- **Cellular respiration**: the set of metabolic reactions that convert the chemical energy stored in glucose into adenosine triphosphate, or ATP
- **Enzymes**
- **Cellular**
```

Note that the photosynthesis and cellular-respiration paragraphs were
merged into one section — they share enough vocabulary (glucose, oxygen,
energy) that the topic-shift detector correctly treats them as one
continuous idea, while the enzymes paragraph (little shared vocabulary)
is split into its own section.

## Running the tests

```bash
pip install pytest    # or just use the stdlib unittest runner below
pytest tests/ -v
# or
python -m unittest discover -s tests -v
```

The test suite (`tests/test_generate_notes.py`) covers:

- Section segmentation from existing markdown headings.
- Section segmentation via topic-shift detection on un-headed text.
- Bullet extraction correctly ranking a crafted sentence as most important.
- Bullet-count bounds and short-section passthrough behavior.
- Glossary extraction from `"is defined as"` and `"refers to"` patterns.
- Glossary extraction from frequently-recurring capitalized terms.
- Valid Markdown structure of the full generated output.
- Graceful handling of empty/whitespace-only input.
- End-to-end CLI invocation.

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs the full
suite on every push and pull request against Python 3.11 and 3.12.

## Project layout

```
generate_notes.py           # the entire pipeline + CLI entry point
tests/test_generate_notes.py
examples/sample_input.txt
examples/sample_output.md
.github/workflows/tests.yml
```

## Troubleshooting / FAQ

**"Error reading input.txt: not valid UTF-8 text"** — the tool only reads
plain-text UTF-8 files. This usually means you pointed it at a PDF, Word
doc, or other binary file. Export/copy the text to a `.txt` file first
(e.g. `pdftotext lecture.pdf lecture.txt`).

**My headings look wrong or nonsensical (e.g. two unrelated words stuck
together).** Heading generation only runs on sections that don't already
have a markdown `#` heading. If your source text has natural section
breaks, add `#`/`##` headings yourself — segmentation and heading
generation are skipped entirely for headed sections, which is usually more
reliable than the automatic topic-shift heuristic on short or unusual text.

**Why didn't `<term>` show up in the glossary?** Glossary extraction only
picks up two signals: (1) explicit definitional phrasing like `"X is
defined as ..."` / `"X refers to ..."` / `"X is a/an ..."`, and (2)
capitalized terms that recur at least twice. A term mentioned once, in
lowercase, or phrased differently (e.g. `"X means ..."`) won't be detected
— this is a pattern-based heuristic, not a language model.

## Limitations

This is a from-scratch classical-NLP implementation, not a language model.
It does not understand meaning — it relies on word overlap, frequency
statistics, and simple surface patterns. It works well on reasonably
well-formed prose but can produce awkward headings or miss glossary terms
on very short, very technical, or unusually formatted text.

## License

MIT — see [LICENSE](LICENSE).
