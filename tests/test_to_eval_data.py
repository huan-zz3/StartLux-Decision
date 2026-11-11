"""finetune/to_eval_data.py: normalization of fine-tuning targets into the single gold labels startlux_decide eval
takes, skipping untargeted questions, dropping empty records, and rejecting ambiguous ties or malformed input with a
line number.

Standard library only: the converter is loaded straight from its file and never imports torch, a model or the network.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr

from startlux_decide.evaluate import gold_label as eval_gold_label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "finetune", "to_eval_data.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("startlux_to_eval_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


to_eval = _load_script()


def _choice():
    return {"type": "choice", "instructions": "Which team?",
            "criteria": {"billing": "Payments", "shipping": "Delivery", "account": "Login"}}


def _noul():
    return {"type": "noul", "instructions": "Does the customer need an answer today?"}


def _score():
    return {"type": "score", "instructions": "How badly is the customer affected?",
            "criteria": ["cosmetic", "annoying", "blocks the customer"]}


class ConvertTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _convert(self, rows=None, raw=None):
        path = os.path.join(self.dir.name, "in.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            if raw is not None:
                f.write(raw)
            else:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out = os.path.join(self.dir.name, "out.jsonl")
        result = to_eval.convert(path, out)
        with open(out, encoding="utf-8") as f:
            written = [json.loads(line) for line in f]
        return result, written

    def test_gold_labels_for_choice_noul_and_score(self):
        rec = {"state": "ticket 4411", "questions": {"team": _choice(), "urgent": _noul(), "severity": _score()},
               "targets": {"team": {"billing": 1.0}, "urgent": {"false": 1.0}, "severity": {"1": 1.0}}}
        result, written = self._convert([rec])
        self.assertEqual(result, (1, 0, 0))
        self.assertEqual(written[0]["gold"], {"team": "billing", "urgent": "false", "severity": 1})
        self.assertIs(type(written[0]["gold"]["severity"]), int)
        for key, spec in rec["questions"].items():
            self.assertEqual(eval_gold_label(spec, written[0]["gold"][key]), str(written[0]["gold"][key]))
        with self.assertRaises(ValueError):        # why the conversion to int matters
            eval_gold_label(_score(), "1")

    def test_bool_type_is_treated_as_noul(self):
        rec = {"state": "ticket", "questions": {"ok": {"type": "bool", "instructions": "Fine?"}},
               "targets": {"ok": {"true": 1.0}}}
        _, written = self._convert([rec])
        self.assertEqual(written[0]["gold"], {"ok": "true"})

    def test_questions_without_targets_are_skipped(self):
        rec = {"state": "ticket", "questions": {"team": _choice(), "urgent": _noul(), "severity": _score()},
               "targets": {"team": {"shipping": 1.0}}}
        result, written = self._convert([rec])
        self.assertEqual(result, (1, 2, 0))
        self.assertEqual(list(written[0]["questions"]), ["team"])
        self.assertEqual(written[0]["gold"], {"team": "shipping"})

    def test_record_with_every_question_skipped_is_dropped(self):
        rows = [{"state": "no targets", "questions": {"team": _choice(), "urgent": _noul()}},
                {"state": "ok", "questions": {"team": _choice()}, "targets": {"team": {"account": 1.0}}}]
        result, written = self._convert(rows)
        self.assertEqual(result, (1, 2, 1))
        self.assertEqual([r["state"] for r in written], ["ok"])

    def test_tied_probabilities_raise_with_line_and_question(self):
        rec = {"state": "ticket", "questions": {"team": _choice()},
               "targets": {"team": {"billing": 0.6, "account": 0.6 - 1e-10}}}
        with self.assertRaises(ValueError) as caught:
            self._convert([rec])
        message = str(caught.exception)
        self.assertIn("in.jsonl:1:", message)
        self.assertIn("'team'", message)
        self.assertIn("'billing'", message)
        self.assertIn("'account'", message)

    def test_images_are_copied_through(self):
        images = ["data:image/png;base64,iVBORw0KGgo="]
        rec = {"state": "photo <image>", "questions": {"damaged": _noul()},
               "targets": {"damaged": {"true": 1.0}}, "images": images}
        _, written = self._convert([rec])
        self.assertEqual(written[0]["images"], images)

    def test_blank_line_and_bad_json_report_the_line(self):
        with self.assertRaises(ValueError) as caught:
            self._convert(raw="\n")
        self.assertIn("in.jsonl:1:", str(caught.exception))
        self.assertIn("invalid JSON", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self._convert(raw='{"state": "x", "questions": {}}\nnot json\n')
        self.assertIn("in.jsonl:2:", str(caught.exception))

    def test_missing_state_and_questions_are_errors(self):
        with self.assertRaises(ValueError) as caught:
            self._convert([{"questions": {}}])
        self.assertIn("in.jsonl:1:", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self._convert([{"state": "x"}])
        self.assertIn('"questions"', str(caught.exception))

    def test_main_writes_the_summary_to_stderr(self):
        path = os.path.join(self.dir.name, "in.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"state": "x", "questions": {"team": _choice()},
                                "targets": {"team": {"billing": 1.0}}}) + "\n")
        out = os.path.join(self.dir.name, "out.jsonl")
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = to_eval.main([path, "-o", out])
        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), f"wrote 1 requests (0 questions skipped, 0 records dropped) to {out}\n")


if __name__ == "__main__":
    unittest.main()
