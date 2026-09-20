#!/usr/bin/env python3
"""
AI Notes Generator
===================

Turns plain-text lecture / article / chapter content into structured,
hierarchical Markdown study notes -- headings, extractive bullet points,
and a glossary of key terms.

Pure Python standard library only. No LLM / network calls of any kind.

Pipeline
--------
1. Segmentation: split the input into sections.
   - If the text already contains markdown-style ``#`` headings, those are
     used directly.
   - Otherwise the text is segmented by paragraph-topic-shift detection:
     consecutive paragraphs are compared with TF-IDF cosine similarity and
     a big drop in similarity marks a new section boundary.
2. Heading generation: sections without an explicit heading get one
   synthesized from their most representative TF-IDF keyword / phrase.
3. Bullet extraction: each section's sentences are scored with a
   TextRank-style graph algorithm (sentence-similarity graph + power
   iteration, built from scratch) and the top 3-6 sentences become bullets.
4. Glossary extraction: capitalized terms that recur, plus terms found in
   definitional patterns such as "X is defined as ..." or "X refers to ...",
   become glossary entries.
5. Rendering: everything is assembled into Markdown.

CLI
---
    python generate_notes.py input.txt output.md
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Tokenization helpers
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']*")

STOPWORDS = frozenset(
    """
    a an the and or but if then else when while as of to in on for with
    without by from at is are was were be been being this that these those
    it its it's into over under again further once here there all any both
    each few more most other some such no nor not only own same so than too
    very s t can will just don should now about above below up down out off
    also because between during before after through until against among
    per your you we they he she i my our their his her them him us who
    whom which what where why how do does did doing have has had having
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, stripped of punctuation."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def content_words(text: str) -> List[str]:
    """Tokens with stopwords removed -- used for TF-IDF style comparisons."""
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 1]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> List[str]:
    """Very small, dependency-free sentence splitter."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Protect common abbreviations from being treated as sentence ends.
    placeholder = "\x01"
    protected = re.sub(
        r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|e\.g|i\.e|etc)\.",
        lambda m: m.group(0).replace(".", placeholder),
        text,
    )
    parts = _SENTENCE_SPLIT_RE.split(protected)
    sentences = [p.replace(placeholder, ".").strip() for p in parts if p.strip()]
    return sentences


def split_paragraphs(text: str) -> List[str]:
    """Split raw text into paragraphs on blank lines."""
    raw_paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in raw_paragraphs if p.strip()]


# --------------------------------------------------------------------------
# TF-IDF utilities (implemented from scratch, no external deps)
# --------------------------------------------------------------------------


def term_frequencies(tokens: List[str]) -> Counter:
    return Counter(tokens)


def build_idf(documents: List[List[str]]) -> Dict[str, float]:
    """Inverse document frequency across a list of tokenized documents."""
    n_docs = len(documents)
    df: Counter = Counter()
    for doc in documents:
        for term in set(doc):
            df[term] += 1
    idf = {}
    for term, count in df.items():
        idf[term] = math.log((1 + n_docs) / (1 + count)) + 1.0
    return idf


def tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = term_frequencies(tokens)
    total = sum(tf.values()) or 1
    vec = {}
    for term, count in tf.items():
        freq = count / total
        vec[term] = freq * idf.get(term, math.log(2))
    return vec


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# --------------------------------------------------------------------------
# Section segmentation
# --------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


class Section:
    __slots__ = ("heading", "body", "heading_given")

    def __init__(self, heading: Optional[str], body: str, heading_given: bool):
        self.heading = heading
        self.body = body
        self.heading_given = heading_given

    def __repr__(self):  # pragma: no cover - debugging helper
        return f"Section(heading={self.heading!r}, body_len={len(self.body)})"


def has_markdown_headings(text: str) -> bool:
    return any(_HEADING_RE.match(line) for line in text.splitlines())


def segment_by_headings(text: str) -> List[Section]:
    """Split text using existing markdown ``#`` headings as boundaries."""
    lines = text.splitlines()
    sections: List[Section] = []
    current_heading: Optional[str] = None
    current_body: List[str] = []
    seen_heading = False

    def flush():
        body_text = "\n".join(current_body).strip()
        if current_heading is not None or body_text:
            sections.append(Section(current_heading, body_text, current_heading is not None))

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            # Starting a new section: flush the previous one (even if empty
            # heading with only preamble text before the first heading).
            if current_heading is not None or current_body:
                flush()
            current_heading = match.group(2).strip()
            current_body = []
            seen_heading = True
        else:
            current_body.append(line)
    flush()

    # Drop a leading empty preamble section (text before the first heading
    # that turned out blank).
    sections = [s for s in sections if s.heading or s.body.strip()]
    return sections


def segment_by_topic_shift(
    text: str, similarity_drop_ratio: float = 0.55, min_paragraphs_per_section: int = 1
) -> List[Section]:
    """
    Segment un-headed text into sections by detecting topic shifts between
    consecutive paragraphs using TF-IDF cosine similarity. A similarity
    that drops sharply relative to the recent average marks a new section.
    """
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []
    if len(paragraphs) == 1:
        return [Section(None, paragraphs[0], False)]

    tokenized = [content_words(p) for p in paragraphs]
    idf = build_idf(tokenized)
    vectors = [tfidf_vector(toks, idf) for toks in tokenized]

    similarities = [
        cosine_similarity(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
    ]

    boundaries = set()
    if similarities:
        running_avg = similarities[0]
        for i, sim in enumerate(similarities):
            # A boundary occurs *after* paragraph i (i.e. before i+1) when
            # similarity drops well below the running average of similarity
            # seen so far, and both paragraphs carry real content.
            threshold = running_avg * similarity_drop_ratio
            if sim < threshold and running_avg > 0:
                boundaries.add(i + 1)
            running_avg = (running_avg * (i + 1) + sim) / (i + 2)

    # Build groups of paragraph indices based on boundaries.
    groups: List[List[int]] = []
    current: List[int] = [0]
    for idx in range(1, len(paragraphs)):
        if idx in boundaries:
            groups.append(current)
            current = [idx]
        else:
            current.append(idx)
    groups.append(current)

    sections = [Section(None, "\n\n".join(paragraphs[i] for i in g), False) for g in groups]
    return sections


def segment_text(text: str) -> List[Section]:
    if has_markdown_headings(text):
        return segment_by_headings(text)
    return segment_by_topic_shift(text)


# --------------------------------------------------------------------------
# Heading generation for headless sections
# --------------------------------------------------------------------------


def generate_heading(section_text: str, idf: Optional[Dict[str, float]] = None) -> str:
    """
    Pick a representative heading for a section without one, using the
    highest TF-IDF scoring keyword in the section, optionally paired with
    an adjacent content word to form a short two-word phrase.

    Phrase candidates are built from adjacent-word windows *within a single
    sentence* only, so a heading can never accidentally splice together
    words from two unrelated sentences.
    """
    # Normalize whitespace (hard-wrapped input may contain single newlines
    # mid-sentence) before doing any sentence/word level work.
    normalized_text = re.sub(r"\s+", " ", section_text).strip()

    words = content_words(normalized_text)
    if not words:
        return "Untitled Section"

    local_idf = idf or build_idf([words])
    vec = tfidf_vector(words, local_idf)
    if not vec:
        return "Untitled Section"

    best_term = max(vec, key=lambda t: vec[t])

    candidate_phrases = []
    for sentence in split_sentences(normalized_text):
        tokens = _WORD_RE.findall(sentence)
        lower_tokens = [t.lower() for t in tokens]
        for i, lower_tok in enumerate(lower_tokens):
            if lower_tok != best_term:
                continue
            if (
                i + 1 < len(tokens)
                and lower_tokens[i + 1] not in STOPWORDS
                and len(tokens[i + 1]) > 1
            ):
                candidate_phrases.append(f"{tokens[i]} {tokens[i + 1]}")
            if (
                i - 1 >= 0
                and lower_tokens[i - 1] not in STOPWORDS
                and len(tokens[i - 1]) > 1
            ):
                candidate_phrases.append(f"{tokens[i - 1]} {tokens[i]}")

    if candidate_phrases:
        # Shortest matching phrase tends to be the cleanest heading.
        chosen = min(candidate_phrases, key=len)
        return chosen.title() if chosen.islower() else chosen

    return best_term.title()


# --------------------------------------------------------------------------
# TextRank-style extractive bullet-point scoring
# --------------------------------------------------------------------------


def _sentence_similarity(a_tokens: List[str], b_tokens: List[str]) -> float:
    """Normalized word-overlap similarity between two sentences (as used in
    the original TextRank paper)."""
    set_a, set_b = set(a_tokens), set(b_tokens)
    if not set_a or not set_b:
        return 0.0
    overlap = len(set_a & set_b)
    denom = math.log(len(set_a) + 1) + math.log(len(set_b) + 1)
    if denom == 0:
        return 0.0
    return overlap / denom


def textrank_scores(
    sentences: List[str], damping: float = 0.85, iterations: int = 40, tol: float = 1e-5
) -> List[float]:
    """Score sentences with a from-scratch TextRank (graph + power
    iteration over a word-overlap similarity matrix)."""
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    tokenized = [content_words(s) for s in sentences]
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sim[i][j] = _sentence_similarity(tokenized[i], tokenized[j])

    out_weight_sum = [sum(sim[i]) for i in range(n)]
    scores = [1.0] * n

    for _ in range(iterations):
        new_scores = [0.0] * n
        for i in range(n):
            rank_sum = 0.0
            for j in range(n):
                if i == j or out_weight_sum[j] == 0:
                    continue
                rank_sum += (sim[j][i] / out_weight_sum[j]) * scores[j]
            new_scores[i] = (1 - damping) + damping * rank_sum
        diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if diff < tol:
            break

    return scores


def tfidf_sentence_scores(sentences: List[str]) -> List[float]:
    """Fallback / complementary scoring: sum of TF-IDF weights of a
    sentence's content words, normalized by length."""
    tokenized = [content_words(s) for s in sentences]
    idf = build_idf(tokenized)
    scores = []
    for toks in tokenized:
        vec = tfidf_vector(toks, idf)
        scores.append(sum(vec.values()) / (len(toks) ** 0.5 + 1))
    return scores


def extract_bullets(
    section_text: str, min_bullets: int = 3, max_bullets: int = 6
) -> List[str]:
    """Pick the most important sentences from a section as bullet points,
    preserving their original order of appearance."""
    sentences = split_sentences(section_text)
    if not sentences:
        return []
    if len(sentences) <= min_bullets:
        return sentences

    tr_scores = textrank_scores(sentences)
    tfidf_scores = tfidf_sentence_scores(sentences)

    # Combine both signals (simple average of min-max normalized scores).
    def normalize(values: List[float]) -> List[float]:
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return [0.5 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]

    combined = [
        0.6 * a + 0.4 * b for a, b in zip(normalize(tr_scores), normalize(tfidf_scores))
    ]

    n_bullets = max(min_bullets, min(max_bullets, math.ceil(len(sentences) * 0.5)))
    n_bullets = min(n_bullets, len(sentences))

    ranked_indices = sorted(range(len(sentences)), key=lambda i: combined[i], reverse=True)
    top_indices = sorted(ranked_indices[:n_bullets])
    return [sentences[i] for i in top_indices]


# --------------------------------------------------------------------------
# Glossary / key-term extraction
# --------------------------------------------------------------------------

_DEFINITION_PATTERNS = [
    re.compile(
        r"\b([A-Z][A-Za-z0-9\-]*(?:\s+[A-Za-z0-9\-]+){0,3})\s+is\s+defined\s+as\s+(.+?)(?:[.!?]|$)"
    ),
    re.compile(
        r"\b([A-Z][A-Za-z0-9\-]*(?:\s+[A-Za-z0-9\-]+){0,3})\s+refers\s+to\s+(.+?)(?:[.!?]|$)"
    ),
    re.compile(
        r"\b([A-Z][A-Za-z0-9\-]*(?:\s+[A-Za-z0-9\-]+){0,3})\s+is\s+an?\s+(.+?)(?:[.!?]|$)"
    ),
]

_CAP_TERM_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")

_COMMON_SENTENCE_STARTERS = frozenset(
    {
        "The", "This", "That", "These", "Those", "It", "In", "On", "A", "An",
        "For", "As", "When", "While", "However", "Therefore", "Thus", "So",
        "But", "And", "Or", "If", "Because", "After", "Before", "During",
    }
)


def extract_glossary(text: str, max_terms: int = 12) -> "Dict[str, str]":
    """
    Extract glossary entries:
      * terms found in definitional patterns ("X is defined as ...",
        "X refers to ...", "X is a/an ...") map to their definition text.
      * frequently-recurring capitalized terms (that aren't just sentence
        starters) are added with an empty/context-free definition.
    """
    glossary: "Dict[str, str]" = {}

    # Normalize whitespace so definitional patterns and sentence-relative
    # regexes work correctly even on hard-wrapped input text.
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text).strip()
    text = re.sub(r" {2,}", " ", text)

    for pattern in _DEFINITION_PATTERNS:
        for match in pattern.finditer(text):
            term = match.group(1).strip()
            definition = match.group(2).strip().rstrip(".")
            if term and term not in glossary:
                glossary[term] = definition

    # Frequent capitalized terms (candidate proper nouns / key concepts).
    term_counts: Counter = Counter()
    for sentence in split_sentences(text):
        words = sentence.split()
        for i, m in enumerate(_CAP_TERM_RE.finditer(sentence)):
            term = m.group(1)
            first_word = term.split()[0]
            # Skip if it is simply a sentence-initial common word.
            if sentence.startswith(term) and first_word in _COMMON_SENTENCE_STARTERS:
                continue
            if first_word in _COMMON_SENTENCE_STARTERS and len(term.split()) == 1:
                continue
            term_counts[term] += 1

    for term, count in term_counts.most_common():
        if len(glossary) >= max_terms:
            break
        if count >= 2 and term not in glossary:
            glossary[term] = ""

    return dict(list(glossary.items())[:max_terms])


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def render_markdown(sections: List[Tuple[str, List[str]]], glossary: Dict[str, str], title: str = "Study Notes") -> str:
    lines = [f"# {title}", ""]

    if not sections:
        lines.append("_No content available to summarize._")
        lines.append("")
    else:
        for heading, bullets in sections:
            lines.append(f"## {heading}")
            lines.append("")
            if bullets:
                for bullet in bullets:
                    lines.append(f"- {bullet}")
            else:
                lines.append("_No salient sentences extracted._")
            lines.append("")

    lines.append("## Glossary")
    lines.append("")
    if glossary:
        for term, definition in glossary.items():
            if definition:
                lines.append(f"- **{term}**: {definition}")
            else:
                lines.append(f"- **{term}**")
    else:
        lines.append("_No glossary terms detected._")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Top-level pipeline
# --------------------------------------------------------------------------


def generate_notes(text: str, title: str = "Study Notes") -> str:
    """Run the full pipeline over raw input text and return Markdown notes."""
    text = text or ""
    if not text.strip():
        return render_markdown([], {}, title=title)

    raw_sections = segment_text(text)
    rendered_sections: List[Tuple[str, List[str]]] = []

    all_word_lists = [content_words(s.body) for s in raw_sections]
    idf = build_idf(all_word_lists) if all_word_lists else {}

    for section in raw_sections:
        heading = section.heading or generate_heading(section.body, idf=idf)
        bullets = extract_bullets(section.body)
        rendered_sections.append((heading, bullets))

    glossary = extract_glossary(text)
    return render_markdown(rendered_sections, glossary, title=title)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: List[str]) -> Optional[Tuple[str, str, str]]:
    """Parse CLI args, returning (input_path, output_path, title) or None
    if the arguments are invalid."""
    title = "Study Notes"
    positional: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--title":
            if i + 1 >= len(argv):
                return None
            title = argv[i + 1]
            i += 2
        elif arg.startswith("--title="):
            title = arg.split("=", 1)[1]
            i += 1
        else:
            positional.append(arg)
            i += 1

    if len(positional) != 2 or not title.strip():
        return None
    return positional[0], positional[1], title


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parsed = _parse_args(argv)
    if parsed is None:
        print(
            "Usage: python generate_notes.py <input.txt> <output.md> [--title \"My Title\"]",
            file=sys.stderr,
        )
        return 1

    input_path, output_path, title = parsed
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        print(f"Error reading {input_path}: {exc}", file=sys.stderr)
        return 1
    except UnicodeDecodeError as exc:
        print(
            f"Error reading {input_path}: not valid UTF-8 text ({exc}). "
            "This tool expects a plain-text (.txt) file, not a binary or "
            "differently-encoded document.",
            file=sys.stderr,
        )
        return 1

    notes = generate_notes(text, title=title)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(notes)
    except OSError as exc:
        print(f"Error writing {output_path}: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote structured notes to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
