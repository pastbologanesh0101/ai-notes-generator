"""
Unit tests for the AI Notes Generator pipeline.

Run with:  pytest
       or:  python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import generate_notes as ng  # noqa: E402


class TestSegmentationByHeadings(unittest.TestCase):
    def test_splits_on_existing_markdown_headings(self):
        text = (
            "# Introduction\n"
            "This is the intro paragraph about the topic.\n\n"
            "# Methods\n"
            "This section explains the methods used in the study.\n\n"
            "# Results\n"
            "This section reports the results that were found.\n"
        )
        self.assertTrue(ng.has_markdown_headings(text))
        sections = ng.segment_by_headings(text)
        headings = [s.heading for s in sections]
        self.assertEqual(headings, ["Introduction", "Methods", "Results"])
        self.assertIn("intro paragraph", sections[0].body)
        self.assertIn("methods used", sections[1].body)
        self.assertIn("results that were found", sections[2].body)

    def test_heading_levels_are_stripped_of_hashes(self):
        text = "## Overview\nSome overview text here that matters a lot.\n"
        sections = ng.segment_by_headings(text)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].heading, "Overview")
        self.assertFalse(sections[0].heading.startswith("#"))


class TestSegmentationByTopicShift(unittest.TestCase):
    def test_detects_topic_shift_on_unheaded_text(self):
        # Two clearly distinct topics, each internally coherent, separated
        # by paragraph breaks and with essentially no shared vocabulary.
        photosynthesis = (
            "Photosynthesis is the process plants use to convert sunlight into energy. "
            "Chlorophyll in plant leaves absorbs sunlight for photosynthesis. "
            "The photosynthesis process produces oxygen as a byproduct for plants.\n\n"
            "Plants rely on photosynthesis and chlorophyll to synthesize glucose. "
            "Sunlight, water, and carbon dioxide are inputs to plant photosynthesis."
        )
        stock_market = (
            "\n\nThe stock market allows investors to buy and sell shares of companies. "
            "Stock prices fluctuate in the market based on investor supply and demand. "
            "Investors trade stocks on the market hoping for financial returns.\n\n"
            "Market indices track the overall performance of stocks across the market. "
            "Investors watch stock market indices to gauge financial trends."
        )
        text = photosynthesis + stock_market
        sections = ng.segment_by_topic_shift(text)
        self.assertGreaterEqual(
            len(sections), 2, "expected topic-shift detection to find at least 2 sections"
        )
        # The first section should be dominated by photosynthesis vocabulary
        # and the last by stock-market vocabulary.
        self.assertIn("photosynthesis", sections[0].body.lower())
        self.assertIn("stock", sections[-1].body.lower())

    def test_single_paragraph_is_single_section(self):
        text = "Just one paragraph with a single coherent idea in it."
        sections = ng.segment_by_topic_shift(text)
        self.assertEqual(len(sections), 1)


class TestBulletExtraction(unittest.TestCase):
    def test_picks_genuinely_important_sentence(self):
        # "Mitochondria" and "energy" are the dominant, repeated concepts;
        # the sentence that ties both central concepts together most
        # densely should be scored highest.
        section_text = (
            "Mitochondria are organelles found in eukaryotic cells. "
            "Mitochondria produce energy for the cell through respiration. "
            "The energy produced by mitochondria powers most cellular activity. "
            "Some cells contain many mitochondria depending on their energy needs. "
            "A banana was left on the kitchen counter yesterday afternoon. "
            "The weather today is sunny with a light breeze."
        )
        sentences = ng.split_sentences(section_text)
        scores = ng.textrank_scores(sentences)
        top_sentence = sentences[scores.index(max(scores))]
        self.assertIn("mitochondria", top_sentence.lower())
        self.assertIn("energy", top_sentence.lower())

        bullets = ng.extract_bullets(section_text, min_bullets=3, max_bullets=4)
        self.assertGreaterEqual(len(bullets), 3)
        self.assertLessEqual(len(bullets), 4)
        # The clearly off-topic filler sentences should not dominate,
        # meaning at least one mitochondria/energy sentence is present.
        self.assertTrue(any("mitochondria" in b.lower() for b in bullets))

    def test_bullet_count_within_bounds(self):
        section_text = " ".join(
            f"This is sentence number {i} about network protocols and packets."
            for i in range(1, 12)
        )
        bullets = ng.extract_bullets(section_text, min_bullets=3, max_bullets=6)
        self.assertGreaterEqual(len(bullets), 3)
        self.assertLessEqual(len(bullets), 6)

    def test_short_section_returns_all_sentences(self):
        section_text = "First sentence here. Second sentence here."
        bullets = ng.extract_bullets(section_text, min_bullets=3, max_bullets=6)
        self.assertEqual(len(bullets), 2)


class TestGlossaryExtraction(unittest.TestCase):
    def test_definitional_pattern_is_defined_as(self):
        text = (
            "Photosynthesis is defined as the process by which plants convert "
            "light energy into chemical energy. Plants use this stored energy to grow."
        )
        glossary = ng.extract_glossary(text)
        self.assertIn("Photosynthesis", glossary)
        self.assertIn("process by which plants convert", glossary["Photosynthesis"])

    def test_definitional_pattern_refers_to(self):
        text = (
            "Latency refers to the delay between a request being sent and a "
            "response being received. Low latency is desirable for real-time systems."
        )
        glossary = ng.extract_glossary(text)
        self.assertIn("Latency", glossary)
        self.assertIn("delay between", glossary["Latency"])

    def test_frequent_capitalized_terms_included(self):
        text = (
            "Kubernetes is a container orchestration platform. "
            "Engineers deploy applications with Kubernetes across clusters. "
            "Kubernetes automates scaling and management of containers. "
            "Many teams adopted Kubernetes for production workloads."
        )
        glossary = ng.extract_glossary(text)
        self.assertIn("Kubernetes", glossary)


class TestMarkdownOutput(unittest.TestCase):
    def test_output_has_valid_markdown_structure(self):
        text = (
            "# Background\n"
            "Neural networks are inspired by the human brain. They consist of "
            "layers of interconnected nodes. Training adjusts the weights of "
            "these connections. Backpropagation is defined as the algorithm "
            "used to compute gradients for training.\n\n"
            "# Applications\n"
            "Neural networks are used in image recognition tasks. They also "
            "power natural language processing systems. Many industries rely "
            "on neural networks for automation.\n"
        )
        markdown = ng.generate_notes(text, title="Deep Learning Notes")

        self.assertTrue(markdown.startswith("# Deep Learning Notes"))
        self.assertIn("## Background", markdown)
        self.assertIn("## Applications", markdown)
        self.assertIn("## Glossary", markdown)
        bullet_lines = [line for line in markdown.splitlines() if line.startswith("- ")]
        self.assertGreater(len(bullet_lines), 0)
        # Glossary section should come after all content sections.
        self.assertLess(markdown.index("## Background"), markdown.index("## Glossary"))
        self.assertLess(markdown.index("## Applications"), markdown.index("## Glossary"))

    def test_empty_input_handled_gracefully(self):
        markdown = ng.generate_notes("", title="Empty Notes")
        self.assertIn("# Empty Notes", markdown)
        self.assertIn("No content available", markdown)
        self.assertIn("## Glossary", markdown)

        markdown_whitespace = ng.generate_notes("   \n\n   ", title="Whitespace Notes")
        self.assertIn("No content available", markdown_whitespace)


class TestGlossaryMaxTerms(unittest.TestCase):
    def test_glossary_respects_max_terms_cap(self):
        # Fifteen distinct, individually-repeated capitalized terms should
        # still yield no more than `max_terms` glossary entries.
        sentences = []
        for i in range(15):
            term = f"Protocol{i}"
            sentences.append(f"{term} is a networking standard. {term} is widely deployed.")
        text = " ".join(sentences)
        glossary = ng.extract_glossary(text, max_terms=5)
        self.assertLessEqual(len(glossary), 5)


class TestCLI(unittest.TestCase):
    def test_cli_wrong_argument_count_prints_usage_and_fails(self):
        result = ng.main(["only_one_arg.txt"])
        self.assertEqual(result, 1)

    def test_cli_missing_input_file_reports_error(self):
        result = ng.main(["/nonexistent/path/does-not-exist.txt", "/tmp/whatever-output.md"])
        self.assertEqual(result, 1)

    def test_cli_writes_output_file(self):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            output_path = os.path.join(tmpdir, "output.md")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(
                    "# Topic\nThis is a short section with a couple of sentences. "
                    "It should still produce output correctly.\n"
                )

            script_path = os.path.join(os.path.dirname(__file__), "..", "generate_notes.py")
            result = subprocess.run(
                [sys.executable, script_path, input_path, output_path],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.path.exists(output_path))
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("## Topic", content)


if __name__ == "__main__":
    unittest.main()
