"""The v1.1 Instant fine-tuning script (finetune/finetune_lora.py): data loading (text and image records) and target
normalization, batching, padding, temperature fitting, and (where torch + transformers 5.8 + STARTLUX_TEST_MODEL_DIR +
peft are available) one tiny CPU training step on a random model with the architecture of a real folder.

The pure tests need only the standard library plus the package's FakeTokenizer.  The tiny-model test is skipped unless
every optional dependency is present; it never downloads a model or uses a GPU.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from startlux_decide.backends.fake import FakeTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "finetune", "finetune_lora.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("startlux_finetune_lora", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses resolves cls.__module__ through sys.modules
    spec.loader.exec_module(module)
    return module


finetune = _load_script()

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False


def _png_uri(width=8, height=8):
    """A tiny PNG as a data URI (needs Pillow)."""
    import base64
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 30, 200)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


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
        items, skipped_over_options, skipped_too_long = finetune.load_items(path, self.tok, 8192)
        self.assertEqual((skipped_over_options, skipped_too_long), (0, 0))
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual((item.options, item.type), (2, "choice"))
        self.assertEqual(item.target, [0.25, 0.75])
        self.assertGreater(len(item.ids), 0)
        self.assertEqual(item.length, len(item.ids))
        self.assertIsNone(item.prompt)
        self.assertIsNone(item.media)

    def test_questions_without_targets_are_skipped(self):
        rec = _row()
        rec["targets"] = {}
        path = _write_jsonl(self.dir, [rec])
        items, over_options, too_long = finetune.load_items(path, self.tok, 8192)
        self.assertEqual((items, over_options, too_long), ([], 0, 0))

    def test_unknown_target_key_raises_with_file_line_question_and_key(self):
        rec = _row(targets={"q": {"billling": 0.7, "shipping": 0.3}})
        path = _write_jsonl(self.dir, [rec])
        with self.assertRaises(ValueError) as ctx:
            finetune.load_items(path, self.tok, 8192)
        message = str(ctx.exception)
        self.assertIn(path, message)
        self.assertIn(":1", message)
        self.assertIn("'q'", message)
        self.assertIn("billling", message)

    def test_unknown_target_key_alongside_a_valid_one_still_raises(self):
        rec = _row(targets={"q": {"billing": 0.7, "billling": 0.3}})
        path = _write_jsonl(self.dir, [rec])
        with self.assertRaises(ValueError) as ctx:
            finetune.load_items(path, self.tok, 8192)
        self.assertIn("billling", str(ctx.exception))

    def test_choice_over_26_options_counts_only_the_options_skip(self):
        crit = {f"o{i}": None for i in range(27)}
        rec = _row(questions={"q": {"type": "choice", "instructions": "Pick", "criteria": crit}},
                   targets={"q": {"o0": 1.0}})
        path = _write_jsonl(self.dir, [rec])
        items, over_options, too_long = finetune.load_items(path, self.tok, 8192)
        self.assertEqual(items, [])
        self.assertEqual((over_options, too_long), (1, 0))

    def test_prompt_over_max_length_counts_only_the_length_skip(self):
        path = _write_jsonl(self.dir, [_row()])
        items, over_options, too_long = finetune.load_items(path, self.tok, 4)
        self.assertEqual(items, [])
        self.assertEqual((over_options, too_long), (0, 1))

    def test_target_too_large_for_a_float_raises_with_file_and_line(self):
        rec = _row(targets={"q": {"billing": 10 ** 400, "shipping": 0}})
        path = _write_jsonl(self.dir, [rec])
        with self.assertRaises(ValueError) as ctx:
            finetune.load_items(path, self.tok, 8192)
        message = str(ctx.exception)
        self.assertIn(path, message)
        self.assertIn(":1", message)
        self.assertIn("'q'", message)

    def test_target_with_mixed_infinities_raises_with_file_and_line(self):
        rec = _row(targets={"q": {"billing": float("inf"), "shipping": float("-inf")}})
        path = _write_jsonl(self.dir, [rec])
        with self.assertRaises(ValueError) as ctx:
            finetune.load_items(path, self.tok, 8192)
        message = str(ctx.exception)
        self.assertIn(path, message)
        self.assertIn(":1", message)
        self.assertIn("'q'", message)

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


@unittest.skipUnless(HAS_PIL, "needs Pillow to build image records")
class ImageLoadItemsTest(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()
        self.dir = tempfile.mkdtemp(prefix="startlux_finetune_images_")
        self.uri = _png_uri()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _load(self, rec, max_length=8192):
        path = _write_jsonl(self.dir, [rec])
        return finetune.load_items(path, self.tok, max_length)

    def test_image_record_expands_length_and_keeps_prompt_and_media(self):
        rec = _row(images=[self.uri])
        rec["state"] = "Photo: <image>"
        items, skipped_over_options, skipped_too_long = self._load(rec)
        self.assertEqual((skipped_over_options, skipped_too_long), (0, 0))
        item = items[0]
        self.assertIsNotNone(item.media)
        self.assertIsNotNone(item.prompt)
        self.assertIn(finetune.IMAGE_PLACEHOLDER, item.prompt)
        self.assertEqual(len(item.media), 1)
        self.assertEqual(item.media.tokens, [64])       # 8x8 -> 256x256 -> 8x8 image tokens
        self.assertEqual(item.length, len(item.ids) + item.media.tokens[0] - 1)
        self.assertGreater(item.length, len(item.ids))
        self.assertEqual(item.media.images[0].size, (256, 256))   # resized once, before the processor
        self.assertEqual(item.media.raw, [])                      # the encoded bytes are not kept

    def test_image_marks_place_images_in_order(self):
        rec = _row(images=[self.uri, self.uri])
        rec["state"] = "left <image> right <image>"
        items, _, _ = self._load(rec)
        item = items[0]
        self.assertEqual(item.prompt.count(finetune.IMAGE_PLACEHOLDER), 2)
        self.assertLess(item.prompt.index("left"), item.prompt.index(finetune.IMAGE_PLACEHOLDER))
        self.assertLess(item.prompt.index("right"), item.prompt.rindex(finetune.IMAGE_PLACEHOLDER))
        self.assertEqual(item.length, len(item.ids) + 2 * (item.media.tokens[0] - 1))

    def test_empty_image_list_is_a_text_record(self):
        items, over_options, too_long = self._load(_row(images=[]))
        self.assertEqual((over_options, too_long), (0, 0))
        self.assertIsNone(items[0].media)
        self.assertIsNone(items[0].prompt)
        self.assertEqual(items[0].length, len(items[0].ids))

    def test_without_marks_images_come_first(self):
        rec = _row(images=[self.uri])
        rec["state"] = "Photo attached"
        items, _, _ = self._load(rec)
        prompt = items[0].prompt
        self.assertIn(finetune.IMAGE_PLACEHOLDER, prompt)
        self.assertLess(prompt.index(finetune.IMAGE_PLACEHOLDER), prompt.index("Photo attached"))

    def test_max_length_uses_expanded_image_length(self):
        rec = _row(images=[self.uri])
        rec["state"] = "Photo: <image>"
        items, _, _ = self._load(rec)
        item = items[0]
        items, over_options, too_long = self._load(rec, max_length=len(item.ids))
        self.assertEqual(items, [])
        self.assertEqual((over_options, too_long), (0, 1))

    def test_marks_without_images_raise_with_file_and_line(self):
        rec = _row()
        rec["state"] = "Photo: <image>"
        path = _write_jsonl(self.dir, [rec])
        with self.assertRaises(ValueError) as ctx:
            finetune.load_items(path, self.tok, 8192)
        self.assertIn(path, str(ctx.exception))
        self.assertIn(":1", str(ctx.exception))
        self.assertIn("<image>", str(ctx.exception))

    def test_mark_count_mismatch_raises_with_file_and_line(self):
        rec = _row(images=[self.uri, self.uri])
        rec["state"] = "Photo: <image>"
        path = _write_jsonl(self.dir, [rec])
        with self.assertRaises(ValueError) as ctx:
            finetune.load_items(path, self.tok, 8192)
        self.assertIn(path, str(ctx.exception))
        self.assertIn(":1", str(ctx.exception))
        self.assertIn("marks", str(ctx.exception))

    def test_bad_image_raises_with_file_and_line(self):
        rec = _row(images=["not base64!"])
        rec["state"] = "Photo: <image>"
        path = _write_jsonl(self.dir, [rec])
        with self.assertRaises(ValueError) as ctx:
            finetune.load_items(path, self.tok, 8192)
        self.assertIn(path, str(ctx.exception))
        self.assertIn(":1", str(ctx.exception))


class BatchesTest(unittest.TestCase):
    def test_every_item_lands_in_exactly_one_bucket(self):
        lengths = [5, 130, 200, 129, 400, 50, 1200, 300]
        items = [finetune.Item(ids=list(range(n)), options=2, target=[1.0, 0.0], type="noul", length=n)
                 for n in lengths]
        plan = finetune.batches(items, 1024, False, 0)
        self.assertEqual(sorted(i for bucket in plan for i in bucket), list(range(len(items))))
        for bucket in plan:                     # a bucket larger than max_tokens holds a single long item
            longest = finetune.padded_length(max(lengths[i] for i in bucket))
            if len(bucket) > 1:
                self.assertLessEqual(longest * len(bucket), 1024)
        for bucket in plan[:-1]:                # length-sorted: buckets shrink
            self.assertGreaterEqual(max(lengths[i] for i in bucket),
                                    max(lengths[i] for i in plan[plan.index(bucket) + 1]))

    def test_image_items_are_singletons(self):
        items = [finetune.Item(ids=list(range(n)), options=2, target=[1.0, 0.0], type="noul", length=n)
                 for n in (50, 400, 60)]
        items.append(finetune.Item(ids=[1, 2, 3], options=2, target=[1.0, 0.0], type="noul", length=3,
                                   prompt="p", media=object()))
        plan = finetune.batches(items, 4096, False, 0)
        image_buckets = [b for b in plan if any(items[i].media is not None for i in b)]
        self.assertEqual(image_buckets, [[3]])
        self.assertEqual(sorted(i for bucket in plan for i in bucket), list(range(len(items))))

    def test_shuffle_changes_bucket_order_only(self):
        items = [finetune.Item(ids=list(range(n)), options=2, target=[1.0, 0.0], type="noul", length=n)
                 for n in range(1, 60)]
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


@unittest.skipIf(finetune.torch is None, "needs torch for the batch guard")
class StepLossGuardTest(unittest.TestCase):
    def test_an_image_item_in_a_text_batch_is_refused(self):
        items = [finetune.Item(ids=[1, 2], options=2, target=[1.0, 0.0], type="noul", length=2),
                 finetune.Item(ids=[1, 2, 3], options=2, target=[1.0, 0.0], type="noul", length=3,
                               prompt="p", media=object())]
        with self.assertRaises(ValueError) as ctx:
            finetune.step_loss(None, None, None, items, [0, 1], 0, finetune.torch.device("cpu"), None, None)
        self.assertIn("forward of its own", str(ctx.exception))


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
