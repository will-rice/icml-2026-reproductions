"""Unittest verification suite for Chain-of-Thought Reasoning In The Wild Is Not Always Faithful reproduction."""

import json
from pathlib import Path
import unittest

from cot_unfaithfulness.iphr_eval import evaluate_iphr_unfaithfulness, load_iphr_benchmark_pairs
from cot_unfaithfulness.patterns import analyze_unfaithfulness_patterns
from cot_unfaithfulness.hard_math import evaluate_hard_math_shortcuts
from cot_unfaithfulness.restoration import analyze_restoration_errors
from generate_evidence import generate_evidence


class TestCoTUnfaithfulness(unittest.TestCase):

    def test_iphr_evaluation(self):
        res = evaluate_iphr_unfaithfulness()
        self.assertTrue(res["claim1_non_adversarial_unfaithfulness_verified"])
        self.assertTrue(res["claim2_iphr_rate_range_verified"])
        self.assertGreaterEqual(res["min_unfaithfulness_rate_pct"], 0.0)
        self.assertLessEqual(res["max_unfaithfulness_rate_pct"], 13.5)
        pairs = load_iphr_benchmark_pairs()
        self.assertGreaterEqual(len(pairs), 3)

    def test_patterns_analysis(self):
        res = analyze_unfaithfulness_patterns()
        self.assertTrue(res["claim3_patterns_verified"])
        self.assertIn("Argument Switching", res["patterns"])
        self.assertIn("Biased Fact Inconsistency", res["patterns"])
        self.assertIn("Answer Flipping", res["patterns"])

    def test_hard_math_shortcuts(self):
        res = evaluate_hard_math_shortcuts()
        self.assertTrue(res["claim4_hard_math_shortcuts_verified"])
        self.assertTrue(res["thinking_exhibit_shortcuts"])
        self.assertTrue(res["non_thinking_exhibit_shortcuts"])

    def test_restoration_errors(self):
        res = analyze_restoration_errors()
        self.assertTrue(res["claim5_restoration_errors_verified"])
        self.assertGreater(res["gsm8k_error_rate_pct"], 0.0)

    def test_generate_evidence_output(self):
        evidence = generate_evidence()
        self.assertEqual(evidence["paper_id"], "NUyt4uxzx0")
        self.assertEqual(evidence["reproducibility_status"], "verified")
        self.assertEqual(len(evidence["claims"]), 5)
        for claim in evidence["claims"]:
            self.assertEqual(claim["status"], "verified")


if __name__ == "__main__":
    unittest.main()
