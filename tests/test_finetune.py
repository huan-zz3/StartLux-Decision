"""The v1.1 Instant fine-tuning script (finetune/finetune_lora.py): data loading and target normalization, batching,
padding, temperature fitting, and (where torch + transformers 5.8 + STARTLUX_TEST_MODEL_DIR + peft are available) one
tiny CPU training step on a random model with the architecture of a real folder.

The pure tests need only the standard library plus the package's FakeTokenizer.  The tiny-model test is skipped unless
every optional dependency is present; it never downloads a model or uses a GPU.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from startlux_decide.backends.fake import FakeTokenizer
from tests.folders import make_folder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "finetune", "finetune_lora.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("startlux_finetune_lora", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finetune = _load_script()


def _row(**over):
    rec = {"state": "ticket 4411",
           "questions": {"q": {"type": "choice", "instructions": "Which team?",
                               "criteria": {"billing": "Payments", "shipping": "Delivery"}}},
           "targets": {"q": {"billing": 1, "shipping": 3}}}
    rec.update(over)
    return rec


def _write_jsonl(directory, rows, name="data.jsonl"):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


class LoadItemsTest(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()
        self.dir = tempfile.mkdtemp(prefix="startlux_finetune_items_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_normalizes_targets_in_option_order(self):
        path = _write_jsonl(self.dir, [_row()])
        items, skipped = finetune.load_items(path, self.tok, 8192)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(items), 1)
        ids, count, vec, typ = items[0]
        self.assertEqual((count, typ), (2, "choice"))
        self.assertEqual(vec, [0.25, 0.75])
        self.assertGreater(len(ids), 0)

    def test_questions_without_targets_are_skipped(self):
        rec = _row()
        rec["targets"] = {}
        path = _write_jsonl(self.dir, [rec])
        items, skipped = finetune.load_items(path, self.tok, 8192)
        self.assertEqual((items, skipped), ([], 0))

    def test_choice_over_26_options_is_skipped(self):
        crit = {f"o{i}": None for i in range(27)}
        rec = _row(questions={"q": {"type": "choice", "instructions": "Pick", "criteria": crit}},
                   targets={"q": {"o0": 1.0}})
        path = _write_jsonl(self.dir, [rec])
        items, skipped = finetune.load_items(path, self.tok, 8192)
        self.assertEqual(items, [])
        self.assertEqual(skipped, 1)

    def test_prompt_over_max_length_is_skipped(self):
        path = _write_jsonl(self.dir, [_row()])
        items, skipped = finetune.load_items(path, self.tok, 4)
        self.assertEqual(items, [])
        self.assertEqual(skipped, 1)

    def test_target_without_mass_raises_with_file_and_line(self):
        rec = _row(targets={"q": {"billing": 0, "shipping": 0}})
        path = _write_jsonl(self.dir, [rec])
        with self.assertRaises(ValueError) as ctx:
            finetune.load_items(path, self.tok, 8192)
        self.assertIn(path, str(ctx.exception))
        self.assertIn(":1", str(ctx.exception))
        self.assertIn("no mass", str(ctx.exception))

    def test_non_object_target_raises_with_file_and_line(self):
        rec = _row(targets={"q": 3})
        path = _write_jsonl(self.dir, [rec])
        with self.assertRaises(ValueError) as ctx:
            finetune.load_items(path, self.tok, 8192)
        self.assertIn(path, str(ctx.exception))
        self.assertIn(":1", str(ctx.exception))

    def test_bad_question_raises_with_file_and_line(self):
        rec = _row(questions={"q": {"type": "wat"}}, targets={"q": {"a": 1.0}})
        path = _write_jsonl(self.dir, [rec])
        with self.assertRaises(ValueError) as ctx:
            finetune.load_items(path, self.tok, 8192)
        self.assertIn(path, str(ctx.exception))
        self.assertIn(":1", str(ctx.exception))

    def test_bad_json_line_raises_with_file_and_line(self):
        path = os.path.join(self.dir, "bad.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json}\n")
        with self.assertRaises(ValueError) as ctx:
            finetune.load_items(path, self.tok, 8192)
        self.assertIn(path, str(ctx.exception))
        self.assertIn(":1", str(ctx.exception))


class BatchesTest(unittest.TestCase):
    def test_every_item_lands_in_exactly_one_bucket(self):
        lengths = [5, 130, 200, 129, 400, 50, 1200, 300]
        items = [(list(range(n)), 2, [1.0, 0.0], "noul") for n in lengths]
        plan = finetune.batches(items, 1024, False, 0)
        self.assertEqual(sorted(i for bucket in plan for i in bucket), list(range(len(items))))
        for bucket in plan:                     # a bucket larger than max_tokens holds a single long item
            longest = finetune.padded_length(max(lengths[i] for i in bucket))
            if len(bucket) > 1:
                self.assertLessEqual(longest * len(bucket), 1024)
        for bucket in plan[:-1]:                # length-sorted: buckets shrink
            self.assertGreaterEqual(max(lengths[i] for i in bucket),
                                    max(lengths[i] for i in plan[plan.index(bucket) + 1]))

    def test_shuffle_changes_bucket_order_only(self):
        items = [(list(range(n)), 2, [1.0, 0.0], "noul") for n in range(1, 60)]
        a = finetune.batches(items, 512, False, 0)
        b = finetune.batches(items, 512, True, 0)
        self.assertEqual(sorted(map(sorted, a)), sorted(map(sorted, b)))


class PaddedLengthTest(unittest.TestCase):
    def test_pads_up_to_the_next_bucket(self):
        self.assertEqual(finetune.padded_length(1), 128)
        self.assertEqual(finetune.padded_length(128), 128)
        self.assertEqual(finetune.padded_length(129), 192)
        self.assertEqual(finetune.padded_length(8193), 10240)
        self.assertEqual(finetune.padded_length(70000), 73728)


@unittest.skipIf(finetune.torch is None, "needs torch for temperature fitting")
class FitTemperaturesTest(unittest.TestCase):
    def test_fits_synthetic_logits(self):
        torch = finetune.torch
        choice = [(torch.tensor([2.0, 0.0]), torch.tensor([1.0, 0.0])),
                  (torch.tensor([3.0, 0.5]), torch.tensor([1.0, 0.0]))]
        noul = [(torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0]))]
        pooled, temps, report = finetune.fit_temperatures({"choice": choice, "noul": noul, "score": []})
        self.assertEqual(set(temps), {"choice", "noul", "score"})
        self.assertTrue(all(0.2 <= t <= 5.0 for t in temps.values()))
        self.assertEqual(temps["noul"], pooled)          # under the 200-example minimum
        self.assertEqual(temps["score"], pooled)         # no examples at all
        self.assertIn("choice", report)
        self.assertEqual(report["choice"]["n"], 2)
        self.assertLessEqual(report["choice"]["nll_after"], report["choice"]["nll_before"])

    def test_empty_input_is_neutral(self):
        pooled, temps, report = finetune.fit_temperatures({"choice": [], "noul": [], "score": []})
        self.assertEqual(pooled, 1.0)
        self.assertEqual(report, {})
        self.assertTrue(all(t == 1.0 for t in temps.values()))


try:
    from . import test_transformers_engine as engine_test

    TINY_READY = engine_test.READY
except Exception:                                # missing torch / transformers / STARTLUX_TEST_MODEL_DIR
    engine_test = None
    TINY_READY = False

try:
    import peft  # noqa: F401

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


class SaveTokenizer:
    """SmallTokenizer whose save_pretrained does not need the real tokenizer files the tiny folder lacks."""


if engine_test is not None:
    class _SaveTokenizer(engine_test.SmallTokenizer):
        def save_pretrained(self, path, **kwargs):
            os.makedirs(path, exist_ok=True)

    SaveTokenizer = _SaveTokenizer


@unittest.skipUnless(TINY_READY and HAS_PEFT, "needs torch, transformers 5.8, STARTLUX_TEST_MODEL_DIR and peft")
class TinyTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = engine_test.model_module()
        fast = {n: None for n in engine_test.FAST_PATHS if getattr(module, n, None) is not None}
        cls.patch = mock.patch.multiple(module, **fast) if fast else None
        if cls.patch:
            cls.patch.start()
        cls.folder = engine_test.tiny_folder(True)      # an Instant (prompt-bidirectional) folder
        cls.tok = SaveTokenizer()

    @classmethod
    def tearDownClass(cls):
        if cls.patch:
            cls.patch.stop()
        shutil.rmtree(cls.folder, ignore_errors=True)

    def test_one_step_merges_adapter_and_configs(self):
        work = tempfile.mkdtemp(prefix="startlux_finetune_run_")
        self.addCleanup(shutil.rmtree, work, True)
        train = _write_jsonl(work, [_row()], name="train.jsonl")
        out = os.path.join(work, "run")
        finetune.main(["--model", self.folder, "--train", train, "--out", out, "--epochs", "1", "--accum", "1",
                       "--max-tokens", "4096", "--max-length", "512", "--device", "cpu"], tokenizer=self.tok)
        self.assertTrue(os.path.isdir(os.path.join(out, "adapter")))
        self.assertTrue(os.path.isdir(os.path.join(out, "merged")))
        with open(os.path.join(out, "merged", "decision_config.json")) as f:
            dec = json.load(f)
        self.assertEqual(dec["prompt_attention"], "bidirectional")   # Instant keys travel with the merged model


if __name__ == "__main__":
    unittest.main()
