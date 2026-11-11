"""Regression tests for ``finetune/check_data.py`` against the local decision sample.

``data/`` is gitignored, so the whole suite skips unless the sample files exist on this machine.  The fixtures are
read-only: nothing is downloaded and no optional dependency (transformers/torch) is needed beyond the standard
library plus Pillow, which the checker itself requires.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(ROOT, "finetune", "check_data.py")
SAMPLE = os.path.join(ROOT, "data", "decision_sample", "out")
HAS_SAMPLE = os.path.exists(os.path.join(SAMPLE, "decision_sample_300.jsonl"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses resolves cls.__module__ through sys.modules
    spec.loader.exec_module(module)
    return module


def _load_checker():
    try:
        return _load("startlux_check_data_sample", CHECK)
    except SystemExit:                       # no Pillow: the checker refuses to start, by design
        return None


check = _load_checker()
NEEDS_PIL = unittest.skipUnless(check is not None, "needs Pillow (check_data.py refuses to run without it)")


def _write_jsonl(directory, rows, name="data.jsonl"):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row if isinstance(row, str) else json.dumps(row, ensure_ascii=False))
            f.write("\n")
    return path


def _record():
    return {"state": "ticket 4411",
            "questions": {"q": {"type": "choice", "instructions": "Which team?",
                                "criteria": {"billing": "Payments", "shipping": "Delivery"}}},
            "targets": {"q": {"billing": 1, "shipping": 3}}}


def _rule_sets(result):
    """line -> every E rule on that record, including its per-question findings."""
    out = {}
    for rec in result["records"]:
        rules = {f["rule"] for f in rec["findings"] if f["kind"] == "E"}
        for qres in rec["questions"]:
            rules |= {f["rule"] for f in qres["findings"] if f["kind"] == "E"}
        out[rec["line"]] = rules
    return out


@NEEDS_PIL
@unittest.skipUnless(HAS_SAMPLE, "requires local data/decision_sample")
class DecisionSampleTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="startlux_check_sample_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = check.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_decision_sample_300_counts_and_findings(self):
        path = os.path.join(SAMPLE, "decision_sample_300.jsonl")
        context = {"signatures": {}, "option_lines": {}, "target_keys": set()}
        result = check.check_file(path, context=context)
        self.assertEqual(result["counts"], {"records": 300, "questions": 1079, "trainable": 1078,
                                            "blocked": 1, "blocked_records": 0})
        blocking = [f for f in result["findings"] if f["kind"] == "E"]
        self.assertEqual([(f["rule"], f["line"], f["question"]) for f in blocking], [("E08", 205, "service")])
        self.assertIn("28", blocking[0]["message"])
        observations = check.finalize_observations(context)
        self.assertEqual(len([f for f in observations if f["rule"] == "W01"]), 76)
        self.assertTrue(any(f["rule"] == "W03" and "domain" in f["message"] for f in result["findings"]))

    def test_decision_sample_extended_40_counts(self):
        path = os.path.join(SAMPLE, "decision_sample_extended_40.jsonl")
        result = check.check_file(path)
        self.assertEqual(result["counts"], {"records": 40, "questions": 200, "trainable": 200,
                                            "blocked": 0, "blocked_records": 0})
        self.assertEqual([f for f in result["findings"] if f["kind"] == "E"], [])

    def test_invalid_cases_rule_sets(self):
        path = os.path.join(SAMPLE, "invalid_cases.jsonl")
        result = check.check_file(path)
        rules = _rule_sets(result)
        rejected = {
            1: {"E02"}, 2: {"E12"}, 3: {"E06"}, 5: {"E07"}, 6: {"E08", "E13"}, 7: {"E08"}, 8: {"E11"},
            # line 9's target only carries an unknown key, so both E13 and zero usable mass (E14) fire.
            9: {"E13", "E14"}, 11: {"E15"}, 13: {"E12"}, 16: {"E12"}, 17: {"E12", "E14"},
            19: {"E13", "E14"},
        }
        for line, expected in rejected.items():
            with self.subTest(line=line):
                self.assertEqual(rules[line], expected)
        accepted = {
            4: "type 缺省为 choice",
            10: "target 概率会被归一化，不需要预先和为 1",
            12: "数字字符串会按 float 转换后接受",
            14: "targets 里多出的问题键会被训练器忽略（Task B 之后报 W03）",
            15: "state 只在 dev 集限制为字符串/对象，train 接受任意 JSON",
            18: "noul 的 criteria 会被渲染器丢弃（W03）",
            20: "只要求概率非负且总质量非零，>1 的总质量会被归一化",
        }
        for line, reason in accepted.items():
            with self.subTest(line=line, reason=reason):
                self.assertEqual(rules[line], set())
        ghost = [f for f in result["findings"]
                 if f["rule"] == "W03" and f["line"] == 14 and ("targets" in f["message"]
                                                               or "不属于任何问题" in f["message"])]
        self.assertTrue(ghost)
        # Line 3's questions is a list (E06): its targets keys are not "unknown", so no extra-key W03 there.
        extra_key_lines = {f["line"] for f in result["findings"]
                           if f["rule"] == "W03" and "不属于任何问题" in f["message"]}
        self.assertEqual(extra_key_lines, {14, 16})

    def test_dev_split_against_train(self):
        with open(os.path.join(SAMPLE, "decision_sample_300.jsonl"), encoding="utf-8") as f:
            lines = f.readlines()
        train = _write_jsonl(self.dir, lines[:240], name="train.jsonl")
        dev = _write_jsonl(self.dir, lines[240:], name="dev.jsonl")
        code, out, err = self._main([dev, "--role", "dev", "--against", train])
        self.assertEqual(code, 0, out + err)
        w04 = [line for line in out.splitlines() if line.startswith("[W04]")]
        self.assertEqual(len(w04), 2)
        self.assertTrue(any("choice" in line and "<200" in line for line in w04))
        self.assertTrue(any("noul" in line and "<200" in line for line in w04))
        self.assertNotIn("[E", out + err)

    def test_dev_tie_is_e16(self):
        rec = dict(_record(), targets={"q": {"billing": 0.5, "shipping": 0.5}})
        path = _write_jsonl(self.dir, [rec], name="tie.jsonl")
        code, out, err = self._main([path, "--role", "dev"])
        self.assertEqual(code, 1)
        self.assertIn("[E16]", out + err)

    def test_leak_against_is_e17(self):
        rec = _record()
        main = _write_jsonl(self.dir, [rec], name="main.jsonl")
        other = _write_jsonl(self.dir, [rec], name="other.jsonl")
        code, out, err = self._main([main, "--against", other])
        self.assertEqual(code, 1)
        self.assertIn("[E17]", out + err)

    def test_malformed_row_does_not_crash_and_writes_report(self):
        path = os.path.join(self.dir, "broken.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json\n")
            f.write(json.dumps(_record(), ensure_ascii=False) + "\n")
        report_path = os.path.join(self.dir, "report.json")
        code, out, err = self._main([path, "--report", report_path])
        self.assertEqual(code, 1, out + err)
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        self.assertIn("E01", [f["rule"] for f in report["findings"]])
        self.assertEqual(report["files"][0]["questions"], 1)
        combined = out + err
        self.assertNotIn("Traceback", combined)
        self.assertNotIn("检查器的问题", combined)

    def test_unknown_target_key_is_w03_only(self):
        rec = dict(_record(), targets={"q": {"billing": 1.0, "shipping": 3.0},
                                       "_ghost_question": {"anything": 1.0}})
        path = _write_jsonl(self.dir, [rec], name="ghost.jsonl")
        code, out, err = self._main([path])
        self.assertEqual(code, 0, out + err)
        self.assertNotIn("[E", out + err)
        w03 = [line for line in out.splitlines() if line.startswith("[W03]") and "_ghost_question" in line]
        self.assertTrue(w03)


if __name__ == "__main__":
    unittest.main()
