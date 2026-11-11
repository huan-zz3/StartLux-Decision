"""Conformance suite for ``finetune/check_data.py``: the provider-facing data checker must agree with the trainer.

The checker is loaded by path (like ``tests/test_finetune.py`` loads the trainer) and everything that needs Pillow or
the package is skipped when those optional pieces are missing.  The suite proves *strictness*, not equality: a fixture
the trainer raises on or skips must be rejected by the checker, and anything the checker accepts must train with the
same option count, target vector, type and effective length.
"""
from __future__ import annotations

import ast
import base64
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

from startlux_decide import images as images_mod
from startlux_decide import prompt as prompt_mod
from startlux_decide.backends.fake import FakeTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(ROOT, "finetune", "check_data.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses resolves cls.__module__ through sys.modules
    spec.loader.exec_module(module)
    return module


def _load_checker():
    try:
        return _load("startlux_check_data", CHECK)
    except SystemExit:                       # no Pillow: the whole checker refuses to start, by design
        return None


check = _load_checker()
trainer = _load("startlux_finetune_lora_for_check", os.path.join(ROOT, "finetune", "finetune_lora.py"))
to_eval = _load("startlux_to_eval_data_for_check", os.path.join(ROOT, "finetune", "to_eval_data.py"))

NEEDS_PIL = unittest.skipUnless(check is not None, "needs Pillow (check_data.py refuses to run without it)")


def _png_uri(width=8, height=8):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 30, 200)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _write_jsonl(directory, rows, name="data.jsonl"):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row if isinstance(row, str) else json.dumps(row, ensure_ascii=False))
            f.write("\n")
    return path


def _record():
    return {"state": "ticket 4411", "questions": {"q": {"type": "choice", "instructions": "Which team?",
                                                        "criteria": {"billing": "Payments", "shipping": "Delivery"}}},
            "targets": {"q": {"billing": 1, "shipping": 3}}}


@NEEDS_PIL
class ConstantsTest(unittest.TestCase):
    def test_constants_match_the_trainer(self):
        self.assertEqual(check.MAX_OPTIONS, prompt_mod.MAX_OPTIONS)
        self.assertEqual(check.MAX_OPTIONS, 26)
        self.assertEqual(check.MAX_LEVELS, prompt_mod.MAX_LEVELS)
        self.assertEqual(check.MAX_LEVELS, 10)
        self.assertEqual(check.MAX_IMAGES, images_mod.MAX_IMAGES)
        self.assertEqual(check.MAX_IMAGES, 16)
        self.assertEqual(check.MIN_PIXELS, images_mod.MIN_PIXELS)
        self.assertEqual(check.MAX_PIXELS, images_mod.MAX_PIXELS)
        self.assertEqual(check.FACTOR, images_mod.FACTOR)
        self.assertEqual(check.FACTOR, 32)
        self.assertEqual(check.DEFAULT_MAX_LENGTH, 8192)
        self.assertEqual(check._parser().get_default("max_length"), 8192)
        self.assertEqual(check.SYSTEM, prompt_mod.SYSTEM)
        self.assertEqual(check.NOUL_INSTRUCTIONS, prompt_mod.NOUL_INSTRUCTIONS)


@NEEDS_PIL
class RenderingEqualityTest(unittest.TestCase):
    CASES = [
        ("choice-dict", "ticket text",
         {"type": "choice", "instructions": "Which team?",
          "criteria": {"billing": "Payments", "shipping": "Delivery"}}),
        ("choice-list", "ticket text",
         {"type": "choice", "instructions": "Pick one", "criteria": ["billing", "shipping"]}),
        ("choice-no-type", "ticket text", {"instructions": "Default type", "options": ["a", "b"]}),
        ("noul-blank", "ticket text", {"type": "noul", "instructions": "   "}),
        ("score-object", "ticket text",
         {"type": "score", "instructions": "How bad?", "criteria": {"2": "bad", "0": "fine", "1": "ok"}}),
        ("long-list", [{"id": i, "note": f"n{i}"} for i in range(10)],
         {"type": "choice", "instructions": "Which item?", "criteria": {"x": None, "y": None}}),
        ("bare-hide", "ticket text",
         {"type": "choice", "instructions": "Pick", "criteria": {"A": "first", "B": "second"}}),
    ]

    def test_messages_are_byte_equal(self):
        for name, state, spec in self.CASES:
            with self.subTest(case=name):
                # 原始 state 对象直接进两侧的 from_systemone：这样 state_text/_index 的注入真的被比较到。
                expected = prompt_mod.messages(prompt_mod.from_systemone(state, spec))
                actual = check.messages(check.from_systemone(state, spec))
                self.assertEqual(actual, expected)

    def test_state_text_matches_the_renderer(self):
        cases = [
            [{"id": i, "note": f"n{i}"} for i in range(9)],
            {"outer": {"inner": [i for i in range(9)]}},
            None, {}, [], "plain string", 42,
        ]
        for value in cases:
            with self.subTest(value=value):
                self.assertEqual(check.state_text(value), prompt_mod.state_text(value))

    def test_image_state_renders_the_same(self):
        uri = _png_uri()
        state = images_mod.place_images(prompt_mod.state_text("Photo <image> attached"), 1)
        spec = {"type": "noul", "instructions": "Is it damaged?"}
        expected = prompt_mod.messages(prompt_mod.from_systemone(state, spec))
        actual = check.messages(check.from_systemone(state, spec))
        self.assertEqual(actual, expected)
        self.assertIn(images_mod.SLOT, actual[1]["content"])
        self.assertTrue(uri)                 # the URI itself is not part of the text; it only needs to decode

    def test_bool_alias_renders_as_noul(self):
        placed = "ticket text"
        spec = {"type": "bool", "instructions": "Yes?"}
        expected = prompt_mod.messages(prompt_mod.from_systemone(placed, spec))
        actual = check.messages(check.from_systemone(placed, spec))
        self.assertEqual(actual, expected)
        self.assertEqual(check.from_systemone(placed, spec)["type"], "noul")


@NEEDS_PIL
class VerdictContainmentTest(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()
        self.dir = tempfile.mkdtemp(prefix="startlux_check_verdicts_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _check(self, rec, **kwargs):
        path = _write_jsonl(self.dir, [rec], name="one.jsonl")
        return path, check.check_file(path, tokenizer=self.tok, **kwargs)

    def _blocking(self, result):
        return [f for f in result["findings"] if f["kind"] == "E"]

    def test_trainer_rejects_are_rejected(self):
        cases = {
            "bad-type": _record_with({"q": {"type": "wat", "instructions": "x"}}),
            "choice-criteria-int": _record_with({"q": {"type": "choice", "instructions": "x", "criteria": 5}}),
            "choice-one-option": _record_with({"q": {"type": "choice", "instructions": "x", "criteria": ["only"]}}),
            "choice-27-options": _record_with({"q": {"type": "choice", "instructions": "x",
                                                     "criteria": {f"o{i}": None for i in range(27)}}}),
            "score-one-level": _record_with({"q": {"type": "score", "instructions": "x", "criteria": ["a"]}}),
            "score-11-levels": _record_with({"q": {"type": "score", "instructions": "x",
                                                   "criteria": [str(i) for i in range(11)]}}),
            "blank-instructions": _record_with({"q": {"type": "choice", "instructions": "  ",
                                                      "criteria": ["a", "b"]}}),
            "newline-option-id": _record_with({"q": {"type": "choice", "instructions": "x",
                                                     "criteria": {"a\nb": None, "c": None}}}),
            "unknown-target-key": _record_with({}, targets={"q": {"nope": 1.0}}),
            "target-not-number": _record_with({}, targets={"q": {"billing": "x", "shipping": 1.0}}),
            "zero-mass": _record_with({}, targets={"q": {"billing": 0, "shipping": 0}}),
            "target-not-object": _record_with({}, targets={"q": 3}),
            "targets-not-object": _record_with({}, targets=[1, 2]),
            "images-not-list": _record_with({}, images="abc"),
            "too-many-images": _record_with({}, images=[_png_uri()] * 17),
            "bad-base64": _record_with({}, images=["not base64!"]),
            "mark-mismatch": _record_with({"q": {"type": "noul", "instructions": "x"}}, state="a <image>",
                                          images=[_png_uri(), _png_uri()], targets={"q": {"true": 1.0}}),
            "aspect-200": _record_with({"q": {"type": "noul", "instructions": "x"}}, state="a <image>",
                                       images=[_png_uri(1, 300)], targets={"q": {"true": 1.0}}),
            "missing-questions": {"state": "s", "targets": {}},
            "question-not-object": {"state": "s", "questions": {"q": 3}, "targets": {"q": {"a": 1.0}}},
        }
        for name, rec in cases.items():
            with self.subTest(case=name):
                _path, result = self._check(rec)
                self.assertTrue(self._blocking(result), f"{name} was accepted but the trainer rejects it")

    def test_contract_rejections_are_rejected(self):
        cases = {
            "blank-state": _record_with({}, state="   "),
            "null-state": _record_with({}, state=None),
            "no-target": _record_with({"q": {"type": "noul", "instructions": "x"}}, targets={}),
            "negative-target": _record_with({}, targets={"q": {"billing": 1.0, "shipping": -0.5}}),
        }
        for name, rec in cases.items():
            with self.subTest(case=name):
                _path, result = self._check(rec)
                self.assertTrue(self._blocking(result), f"{name} was accepted")

    def test_dev_tie_is_rejected(self):
        rec = _record_with({}, targets={"q": {"billing": 0.5, "shipping": 0.5}})
        _path, result = self._check(rec, role="dev")
        self.assertIn("E16", [f["rule"] for f in self._blocking(result)])

    def test_accepted_fixtures_train_identically(self):
        cases = {
            "choice": _record(),
            "noul": _record_with({"q": {"type": "noul", "instructions": "Yes?"}}, targets={"q": {"true": 1.0, }}),
            "bool": _record_with({"q": {"type": "bool", "instructions": "Yes?"}}, targets={"q": {"false": 1.0}}),
            "absent-type": _record_with({"q": {"instructions": "Pick", "options": ["a", "b"]}},
                                        targets={"q": {"a": 1.0}}),
            "choice-list": _record_with({"q": {"type": "choice", "instructions": "Pick", "criteria": ["a", "b"]}},
                                        targets={"q": {"a": 2.0, "b": 1.0}}),
            "score-object": _record_with({"q": {"type": "score", "instructions": "How bad?",
                                                "criteria": {"2": "bad", "0": "fine", "1": "ok"}}},
                                         targets={"q": {"1": 1.0}}),
            "noul-blank-instructions": _record_with({"q": {"type": "noul", "instructions": "  "}},
                                                    targets={"q": {"true": 1.0}}),
            "dict-state": _record_with({}, state={"a": 1, "b": [1, 2, 3]}),
        }
        for name, rec in cases.items():
            with self.subTest(case=name):
                path, result = self._check(rec)
                self.assertFalse(self._blocking(result), f"{name} rejected: {self._blocking(result)}")
                items, over, long = trainer.load_items(path, self.tok, 8192)
                self.assertEqual((over, long), (0, 0))
                self.assertEqual(len(items), 1, f"{name}: trainer did not accept the fixture")
                item = items[0]
                qres = result["records"][0]["questions"][0]
                self.assertTrue(qres["trainable"])
                self.assertEqual(len(qres["options"]), item.options)
                self.assertEqual(qres["type"], item.type)
                self.assertEqual(qres["length"], item.length)
                self.assertEqual(len(qres["target"]), len(item.target))
                for got, want in zip(qres["target"], item.target):
                    self.assertAlmostEqual(got, want, places=12)

    def test_image_fixture_trains_identically(self):
        rec = _record_with({"q": {"type": "noul", "instructions": "Damaged?"}}, state="photo <image>",
                           images=[_png_uri()], targets={"q": {"true": 1.0}})
        path, result = self._check(rec)
        self.assertFalse(self._blocking(result))
        items, over, long = trainer.load_items(path, self.tok, 8192)
        self.assertEqual((len(items), over, long), (1, 0, 0))
        qres = result["records"][0]["questions"][0]
        self.assertEqual(qres["length"], items[0].length)
        self.assertEqual(qres["length"], len(items[0].ids) + items[0].media.tokens[0] - 1)

    def test_too_long_prompt_is_rejected_and_length_unverified_without_tokenizer(self):
        path = _write_jsonl(self.dir, [_record()], name="long.jsonl")
        result = check.check_file(path, settings=check.Settings(max_length=4), tokenizer=self.tok)
        self.assertIn("E18", [f["rule"] for f in self._blocking(result)])
        items, over, long = trainer.load_items(path, self.tok, 4)
        self.assertEqual((items, over, long), ([], 0, 1))
        plain = check.check_file(path)
        self.assertEqual(self._blocking(plain), [])
        self.assertIsNone(plain["records"][0]["questions"][0]["length"])
        report = check.build_report([plain], [], check.Settings(), "train", None, "unknown")
        self.assertFalse(report["length_verified"])
        self.assertEqual(report["unverified"], ["E17", "E18"])
        self.assertEqual(report["mode"], "plain")

    def test_image_only_length_lower_bound_without_tokenizer(self):
        rec = _record_with({"q": {"type": "noul", "instructions": "Damaged?"}}, state="photo <image>",
                           images=[_png_uri()], targets={"q": {"true": 1.0}})
        path = _write_jsonl(self.dir, [rec], name="image_len.jsonl")
        self.assertEqual(check.image_info([_png_uri()])[2], [64])     # 8x8 -> 256x256 -> 64 tokens
        over = check.check_file(path, settings=check.Settings(max_length=32))
        self.assertIn("E18", [f["rule"] for f in self._blocking(over)])
        self.assertIsNone(over["records"][0]["questions"][0]["length"])
        under = check.check_file(path, settings=check.Settings(max_length=64))
        self.assertEqual(self._blocking(under), [])
        self.assertIsNone(under["records"][0]["questions"][0]["length"])
        report = check.build_report([under], [], check.Settings(max_length=64), "train", None, "unknown")
        self.assertEqual(report["unverified"], ["E17", "E18"])
        self.assertEqual(report["image_factor_source"], "default")
        with_against = check.build_report([under], [], check.Settings(max_length=64), "train", None, "unknown",
                                          against_given=True)
        self.assertEqual(with_against["unverified"], ["E18"])
        items, over_options, too_long = trainer.load_items(path, self.tok, 32)
        self.assertEqual((items, over_options, too_long), ([], 0, 1))

    def test_w02_is_per_question_and_option(self):
        questions = {"type": "choice", "instructions": "Pick", "criteria": ["a", "b"]}
        recs = [
            {"state": "s1", "questions": {"q1": dict(questions)}, "targets": {"q1": {"a": 0.5, "b": 0.5}}},
            {"state": "s2", "questions": {"q2": dict(questions)}, "targets": {"q2": {"a": 1.0}}},
        ]
        path = _write_jsonl(self.dir, recs, name="w02.jsonl")
        context = {"signatures": {}, "option_lines": {}, "target_keys": set()}
        check.check_file(path, tokenizer=self.tok, context=context)
        w02 = [f for f in check.finalize_observations(context) if f["rule"] == "W02"]
        self.assertEqual(len(w02), 1)
        self.assertEqual(w02[0]["question"], "q2")
        self.assertIn("选项 'b'", w02[0]["message"])
        self.assertNotIn("q1", [f["question"] for f in w02])

    def test_empty_question_key_is_an_observation_not_a_blocker(self):
        rec = {"state": "s", "questions": {"": {"type": "noul", "instructions": "x"}},
               "targets": {"": {"true": 1.0}}}
        path, result = self._check(rec)
        self.assertEqual(self._blocking(result), [])
        self.assertIn("W03", [f["rule"] for f in result["findings"]])
        items, over, long = trainer.load_items(path, self.tok, 8192)
        self.assertEqual((len(items), over, long), (1, 0, 0))

    def test_dev_list_state_is_rejected(self):
        rec = _record_with({}, state=[1, 2, 3])
        _path, train = self._check(rec, role="train")
        self.assertEqual(self._blocking(train), [])
        _path, dev = self._check(rec, role="dev")
        rules = [f["rule"] for f in self._blocking(dev)]
        self.assertIn("E02", rules)
        message = next(f["message"] for f in self._blocking(dev) if f["rule"] == "E02")
        self.assertIn("dev 集的 state 必须是字符串或对象", message)

    def test_non_finite_target_is_rejected(self):
        cases = {
            "nan": ({"q": {"billing": float("nan"), "shipping": 1.0}}, "E19"),
            "inf": ({"q": {"billing": float("inf"), "shipping": 1.0}}, "E19"),
            "mixed-infinities": ({"q": {"billing": float("inf"), "shipping": float("-inf")}}, "E19"),
            "fsum-overflow": ({"q": {"billing": 1e308, "shipping": 1e308}}, "E14"),
            "huge-int": ({"q": {"billing": 10 ** 400, "shipping": 1}}, "E14"),
        }
        for name, (targets, rule) in cases.items():
            with self.subTest(case=name):
                _path, result = self._check(_record_with({}, targets=targets))
                self.assertIn(rule, [f["rule"] for f in self._blocking(result)])

    def test_blank_lines_are_not_findings(self):
        path = os.path.join(self.dir, "blank.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n")
            f.write(json.dumps(_record()) + "\n")
            f.write("   \n")
        result = check.check_file(path, tokenizer=self.tok)
        self.assertEqual(self._blocking(result), [])
        self.assertEqual(result["counts"]["records"], 1)
        items, _over, _long = trainer.load_items(path, self.tok, 8192)
        self.assertEqual(len(items), 1)


def _record_with(questions=None, state="ticket 4411", targets=None, images=None):
    rec = _record()
    if questions:
        rec["questions"] = questions
    if targets is not None:
        rec["targets"] = targets
    rec["state"] = state
    if images is not None:
        rec["images"] = images
    return rec


@NEEDS_PIL
class ImageArithmeticTest(unittest.TestCase):
    def test_matches_load_media(self):
        for width, height in [(8, 8), (300, 300), (2000, 2000), (1, 200), (200, 1), (10, 2000)]:
            with self.subTest(size=(width, height)):
                uri = _png_uri(width, height)
                media = images_mod.load_media([uri])
                raw, sizes, tokens = check.image_info([uri])
                self.assertEqual(sizes, media.sizes)
                self.assertEqual(tokens, media.tokens)
                self.assertEqual(len(raw), len(media.raw))

    def test_near_and_over_the_aspect_limit(self):
        check.image_info([_png_uri(200, 1)])          # exactly 200x: accepted
        with self.assertRaises(ValueError):
            check.image_info([_png_uri(201, 1)])
        with self.assertRaises(ValueError):
            images_mod.load_media([_png_uri(201, 1)])

    def test_pixel_bounds_and_token_count(self):
        for width, height in [(1, 1), (4000, 4000)]:
            uri = _png_uri(width, height)
            _raw, sizes, tokens = check.image_info([uri])
            self.assertEqual(tokens, [(h // 32) * (w // 32) for h, w in sizes])
            self.assertGreaterEqual(sizes[0][0] * sizes[0][1], check.MIN_PIXELS)


@NEEDS_PIL
class EmitEvalTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="startlux_check_eval_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_equals_to_eval_data_convert(self):
        rows = [
            {"state": "a", "questions": {"q": {"type": "choice", "instructions": "Pick",
                                               "criteria": ["x", "y"]}}, "targets": {"q": {"x": 1.0}}},
            {"state": {"ticket": 1}, "questions": {"s": {"type": "score", "instructions": "How?",
                                                         "criteria": ["low", "high"]}},
             "targets": {"s": {"1": 1.0}}, "id": "custom"},
            {"state": "b", "images": [_png_uri()],
             "questions": {"n": {"type": "bool", "instructions": "Yes?"}}, "targets": {"n": {"false": 1.0}}},
            {"state": "c", "questions": {"q": {"type": "choice", "instructions": "Pick",
                                               "criteria": ["x", "y"]}, "r": {"type": "noul", "instructions": "?"}},
             "targets": {"q": {"y": 0.6, "x": 0.4}}},
            {"state": "d", "questions": {"q": {"type": "noul", "instructions": "?"}}, "targets": {}},
        ]
        path = _write_jsonl(self.dir, rows, name="dev.jsonl")
        expected_path = os.path.join(self.dir, "expected.jsonl")
        actual_path = os.path.join(self.dir, "actual.jsonl")
        to_eval.convert(path, expected_path)
        wrote, skipped, dropped = check.emit_eval([path], actual_path)
        self.assertEqual((wrote, skipped, dropped), to_eval.convert(path, os.path.join(self.dir, "again.jsonl")))
        with open(expected_path, encoding="utf-8") as f:
            expected = [json.loads(line) for line in f]
        with open(actual_path, encoding="utf-8") as f:
            actual = [json.loads(line) for line in f]
        self.assertEqual(actual, expected)

    def test_tie_raises_like_convert(self):
        path = _write_jsonl(self.dir, [{"state": "a", "questions": {"q": {"type": "choice", "instructions": "Pick",
                                                                         "criteria": ["x", "y"]}},
                                        "targets": {"q": {"x": 0.5, "y": 0.5}}}], name="tie.jsonl")
        with self.assertRaises(ValueError):
            to_eval.convert(path, os.path.join(self.dir, "e.jsonl"))
        with self.assertRaises(ValueError):
            check.emit_eval([path], os.path.join(self.dir, "a.jsonl"))

    def test_invalid_fixtures_raise_in_both_converters(self):
        cases = {
            "missing-state": {"questions": {"q": {"type": "noul", "instructions": "?"}},
                              "targets": {"q": {"true": 1.0}}},
            "list-state": {"state": [1, 2], "questions": {"q": {"type": "noul", "instructions": "?"}},
                           "targets": {"q": {"true": 1.0}}},
            "top-tie": {"state": "a", "questions": {"q": {"type": "choice", "instructions": "Pick",
                                                          "criteria": ["x", "y"]}},
                        "targets": {"q": {"x": 0.5, "y": 0.5}}},
            "score-non-numeric-key": {"state": "a", "questions": {"q": {"type": "score", "instructions": "How?",
                                                                       "criteria": ["low", "high"]}},
                                      "targets": {"q": {"high": 1.0}}},
        }
        for name, rec in cases.items():
            with self.subTest(case=name):
                path = _write_jsonl(self.dir, [rec], name=f"{name}.jsonl")
                with self.assertRaises(ValueError):
                    to_eval.convert(path, os.path.join(self.dir, f"{name}.expected.jsonl"))
                with self.assertRaises(ValueError):
                    check.emit_eval([path], os.path.join(self.dir, f"{name}.actual.jsonl"))

    def test_blank_lines_are_skipped(self):
        good = {"state": "a", "questions": {"q": {"type": "noul", "instructions": "?"}},
                "targets": {"q": {"true": 1.0}}}
        path = os.path.join(self.dir, "blank.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n" + json.dumps(good) + "\n\n")
        wrote, _skipped, _dropped = check.emit_eval([path], os.path.join(self.dir, "out.jsonl"))
        self.assertEqual(wrote, 1)


@NEEDS_PIL
class CliHonestyTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="startlux_check_cli_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = check.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_empty_tokenizer_behaves_like_no_flag(self):
        path = _write_jsonl(self.dir, [_record()], name="tok.jsonl")
        report_path = os.path.join(self.dir, "report.json")
        code, _out, _err = self._run([path, "--tokenizer", "", "--report", report_path])
        self.assertEqual(code, 0)
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        self.assertFalse(report["length_verified"])
        self.assertEqual(report["mode"], "plain")
        self.assertIsNone(report["model_folder"])
        self.assertIn("E18", report["unverified"])

    def test_missing_input_file_is_a_usage_error(self):
        missing = os.path.join(self.dir, "does-not-exist.jsonl")
        code, _out, err = self._run([missing])
        self.assertEqual(code, 2)
        self.assertIn("用法错误", err)
        self.assertNotIn("这是检查器的问题", err)

    def test_missing_against_file_is_a_usage_error(self):
        path = _write_jsonl(self.dir, [_record()], name="main.jsonl")
        missing = os.path.join(self.dir, "nope.jsonl")
        code, _out, err = self._run([path, "--against", missing])
        self.assertEqual(code, 2)
        self.assertIn("用法错误", err)

    def test_emit_eval_does_not_touch_the_file_when_blocked(self):
        rec = _record_with({}, state=[1, 2, 3])
        path = _write_jsonl(self.dir, [rec], name="dev.jsonl")
        out_path = os.path.join(self.dir, "eval.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("sentinel\n")
        code, _out, err = self._run([path, "--role", "dev", "--emit-eval", out_path])
        self.assertEqual(code, 1)
        self.assertIn("保持原样", err)
        with open(out_path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "sentinel\n")

    def test_dev_overflow_targets_are_data_errors(self):
        path = _write_jsonl(self.dir, [
            {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}},
             "targets": {"q": {"true": 10 ** 400, "false": 1}}}], name="dev_overflow.jsonl")
        code, out, err = self._run([path, "--role", "dev"])
        self.assertEqual(code, 1)
        self.assertIn("E14", out)
        self.assertNotIn("这是检查器的问题", out + err)
        self.assertNotIn("Traceback", err)
        path = _write_jsonl(self.dir, [
            {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}},
             "targets": {"q": {"true": float("inf"), "false": float("-inf")}}}], name="dev_inf.jsonl")
        code, out, err = self._run([path, "--role", "dev"])
        self.assertEqual(code, 1)
        self.assertIn("E19", out)
        self.assertNotIn("这是检查器的问题", out + err)

    def test_bad_flag_values_are_usage_errors(self):
        path = _write_jsonl(self.dir, [_record()], name="flags.jsonl")
        for argv in ([path, "--image-factor", "0"], [path, "--min-pixels", "0"],
                     [path, "--min-pixels", "100", "--max-pixels", "50"]):
            with self.subTest(argv=argv):
                code, _out, err = self._run(argv)
                self.assertEqual(code, 2)
                self.assertIn("用法错误", err)

    def test_model_image_factor_is_read_from_config(self):
        folder = os.path.join(self.dir, "model")
        os.makedirs(folder)
        with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"vision_config": {"patch_size": 8, "spatial_merge_size": 2}}, f)
        self.assertEqual(check._model_image_factor(folder), 16)
        with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"patch_size": 14, "spatial_merge_size": 2}, f)
        self.assertIsNone(check._model_image_factor(folder))   # 顶层键不被 vision_settings 读取
        with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"model_type": "text"}, f)
        self.assertIsNone(check._model_image_factor(folder))
        self.assertIsNone(check._model_image_factor(os.path.join(self.dir, "absent")))


@NEEDS_PIL
class RuleCompletenessTest(unittest.TestCase):
    SOURCES = {
        "prompt": os.path.join(ROOT, "startlux_decide", "prompt.py"),
        "images": os.path.join(ROOT, "startlux_decide", "images.py"),
        "finetune_lora": os.path.join(ROOT, "finetune", "finetune_lora.py"),
    }
    # prompt.py / images.py: every function in the file (None = 不筛选)；训练器只扫数据装载路径。
    FUNCTIONS_BY_MODULE = {
        "prompt": None,
        "images": None,
        "finetune_lora": {"load_items", "_record_media"},
    }

    @staticmethod
    def _message_text(node):
        def literal(value):
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            if isinstance(value, ast.JoinedStr):
                pieces = []
                for part in value.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        pieces.append(part.value)
                    elif isinstance(part, ast.FormattedValue):
                        expr = getattr(part.value, "id", "{}")
                        pieces.append("{" + expr + ("!r" if part.conversion == 114 else "") + "}")
                return "".join(pieces)
            return ""
        if isinstance(node, ast.Call) and node.args:
            return literal(node.args[0])
        return ""

    def test_every_raise_maps_to_a_rule(self):
        missing = []
        for module_name, path in self.SOURCES.items():
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
            for func in ast.walk(tree):
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                names = self.FUNCTIONS_BY_MODULE[module_name]
                if names is not None and func.name not in names:
                    continue
                for node in ast.walk(func):
                    if not isinstance(node, ast.Raise):
                        continue
                    message = self._message_text(node.exc)
                    mapped = any(f == func.name and needle in message for (f, needle) in check.RULES)
                    exempt = any(f == func.name and needle in message for (f, needle) in check.EXEMPT)
                    if not (mapped or exempt):
                        missing.append(f"{path}:{node.lineno} {func.name}: {message!r}")
        self.assertEqual(missing, [], "raise sites without a rule id or an EXEMPT reason")

    def test_load_items_continue_sites_are_tagged_in_order(self):
        path = self.SOURCES["finetune_lora"]
        with open(path, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=path)
        lines = source.splitlines()
        func = next(node for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "load_items")
        events, problems = [], []
        for node in ast.walk(func):
            if not isinstance(node, (ast.Continue, ast.Break)):
                continue
            line = lines[node.lineno - 1]
            if "# not-a-finding:" in line:
                events.append((node.lineno, "not-a-finding"))
            elif "# rule:" in line:
                marker = line.split("# rule:", 1)[1].strip().split()[0]
                events.append((node.lineno, marker))
            else:
                problems.append(f"{path}:{node.lineno}: {line.strip()}")
        events.sort()
        self.assertEqual(problems, [], "every continue/break in load_items needs '# rule: E..' or "
                                       "'# not-a-finding: ...'")
        self.assertEqual([marker for _line, marker in events],
                         ["not-a-finding", "E12", "E08", "E18"])
        line_numbers = [line for line, _marker in events]
        self.assertEqual(len(line_numbers), len(set(line_numbers)))
        self.assertTrue(all(a < b for a, b in zip(line_numbers, line_numbers[1:])))
        self.assertTrue(set(m for _line, m in events) - {"not-a-finding"} <= set(check.RULE_INFO))
        last_statement = func.body[-1].lineno
        early_returns = [node.lineno for node in ast.walk(func)
                         if isinstance(node, ast.Return) and node.lineno < last_statement]
        self.assertEqual(early_returns, [], f"load_items has a return before its final statement: {early_returns}")

    def test_rule_ids_and_tags_are_the_documented_set(self):
        self.assertEqual(sorted(check.RULE_INFO), sorted(
            [f"E{i:02d}" for i in range(1, 20)] + ["W01", "W02", "W03", "W04"]))
        self.assertEqual({rule: info[0] for rule, info in check.RULE_INFO.items() if rule.startswith("E")}, {
            "E01": "trainer", "E02": "contract", "E03": "trainer", "E04": "trainer", "E05": "trainer",
            "E06": "trainer", "E07": "trainer", "E08": "trainer", "E09": "trainer", "E10": "trainer",
            "E11": "trainer", "E12": "contract", "E13": "trainer", "E14": "trainer", "E15": "contract",
            "E16": "contract", "E17": "contract", "E18": "trainer", "E19": "contract",
        })
        values = set(check.RULES.values())
        self.assertTrue(values <= {f"E{i:02d}" for i in range(1, 20)})


@NEEDS_PIL
class ImportGuardTest(unittest.TestCase):
    ALLOWED = {"__future__", "argparse", "base64", "binascii", "hashlib", "io", "json", "math", "os", "re",
               "statistics", "string", "subprocess", "sys", "traceback", "PIL"}

    def test_imports_are_stdlib_plus_pil(self):
        with open(CHECK, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=CHECK)

        found = []

        class Visitor(ast.NodeVisitor):
            def visit_Import(self, node):
                found.extend(alias.name for alias in node.names)

            def visit_ImportFrom(self, node):
                found.append(node.module or "")

            def visit_FunctionDef(self, node):        # 函数内按需导入不算模块级依赖
                pass

            def visit_AsyncFunctionDef(self, node):
                pass

        Visitor().visit(tree)
        roots = {name.split(".")[0] for name in found}
        self.assertTrue(roots <= self.ALLOWED, f"unexpected module-level imports: {sorted(roots - self.ALLOWED)}")

    def test_never_imports_startlux_decide(self):
        with open(CHECK, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=CHECK)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            self.assertFalse(any(name.split(".")[0] == "startlux_decide" for name in names),
                             "check_data.py must not import startlux_decide")


@NEEDS_PIL
class SampleFileTest(unittest.TestCase):
    def test_sample_passes_and_renders(self):
        sample = os.path.join(ROOT, "examples", "finetune", "sample.jsonl")
        result = check.check_file(sample)
        blocking = [f for f in result["findings"] if f["kind"] == "E"]
        self.assertEqual(blocking, [])
        self.assertEqual(result["counts"]["trainable"], 41)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = check.main([sample])
        self.assertEqual(code, 0)
        text = out.getvalue()
        w02 = [line for line in text.splitlines() if line.startswith("[W02]")]
        self.assertEqual(len(w02), 3)
        self.assertEqual({line.split("问题 ", 1)[1] for line in w02}, {
            "'mood' 的选项 '0' 从未作为答案出现",
            "'mood' 的选项 '2' 从未作为答案出现",
            "'damaged' 的选项 'false' 从未作为答案出现",
        })
        self.assertIn("41 个可训练", text)

    def test_render_one_exits_zero(self):
        sample = os.path.join(ROOT, "examples", "finetune", "sample.jsonl")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = check.main([sample, "--render", "1"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("[system]", text)
        self.assertIn("Evidence:", text)


@NEEDS_PIL
class RealTokenizerTest(unittest.TestCase):
    """The --tokenizer path against a real folder; skipped without STARTLUX_TEST_MODEL_DIR + transformers."""

    @classmethod
    def setUpClass(cls):
        cls.model_dir = os.environ.get("STARTLUX_TEST_MODEL_DIR")
        if not cls.model_dir:
            raise unittest.SkipTest("STARTLUX_TEST_MODEL_DIR is not set")
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise unittest.SkipTest("transformers is not installed")
        cls.tok = AutoTokenizer.from_pretrained(cls.model_dir)

    def test_effective_length_matches_load_items(self):
        directory = tempfile.mkdtemp(prefix="startlux_check_real_tok_")
        try:
            rec = _record_with({"q": {"type": "noul", "instructions": "Damaged?"}}, state="photo <image>",
                               images=[_png_uri()], targets={"q": {"true": 1.0}})
            path = _write_jsonl(directory, [rec])
            result = check.check_file(path, tokenizer=self.tok)
            self.assertFalse([f for f in result["findings"] if f["kind"] == "E"])
            items, over, long = trainer.load_items(path, self.tok, 8192)
            self.assertEqual((len(items), over, long), (1, 0, 0))
            self.assertEqual(result["records"][0]["questions"][0]["length"], items[0].length)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
