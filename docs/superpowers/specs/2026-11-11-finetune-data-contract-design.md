# Fine-tuning data contract: a self-contained checker, a canonical form, and a handoff report

Status: design, awaiting approval. Not part of the user manuals in `docs/`.

## Intent

Two people produce a fine-tune, and they do not share a working environment:

- The **fine-tune developer** owns this repository, the model folder, `finetune/finetune_lora.py`, and the GPU.
- The **data provider** produces every field of every record — `state`, `questions` (including option sets and
  instruction wording), `targets`, and images — and never reads the engine.

Today the contract between them exists in three places with different authority: prose (`docs/finetuning.md`), one
example file (`examples/finetune/sample.jsonl`), and the loader (`finetune/finetune_lora.py:109`). Only the third is
binding, and it is the one the data provider cannot see.

The failure this design removes is not "bad data raises". It is **silent shrinkage**: `load_items` drops choice
questions with more than 26 options and prompts over `--max-length` by counting them, and silently fills 0.0 for a
target key that matches no option id (`finetune/finetune_lora.py:152`, `:161`, `:176`). The provider believes it
delivered 12,480 questions; training uses 11,901; nothing names the missing 579.

## Goals

1. The data provider can decide "this file is trainable" before handoff, with no repository, no model folder, and no
   training dependency.
2. Both sides can see **what the model will read**, not a paraphrase of it.
3. The handoff produces one number both sides accept: questions delivered, questions trainable, and the reason for
   every question that is not.
4. Rules cannot be narrowed for the checker's convenience: the checker's constants and verdicts are pinned to the
   trainer's by tests that fail in CI.
5. Nothing is reported as passing unless it was actually verified.

## Non-goals

- Not a general JSON Schema tool, a data-cleaning tool, or an annotation tool.
- No remote validation service; data stays on the provider's machine.
- No GUI, no packaged binary, no second script for the provider.
- The checker does not decide whether labels are *correct*, only whether they are *well-formed and trainable*. Label
  quality stays a human review item.

## Anti-goals

- A checker that passes a file the trainer will silently skip.
- A checker whose rules drift from `load_items` and `from_systemone`.
- A checker that needs the repository or torch/transformers on the provider's machine.
- A checker that turns "I could not verify this" into "pass".

## Constraints

- One file for the provider: `finetune/check_data.py`, standard library plus **Pillow only** (a hard requirement — no
  capability probing, no degraded mode). It must not import `startlux_decide`.
- Python 3.9 compatible, `from __future__ import annotations`, no lint/typecheck config (repository convention).
- The provider may install light packages with conda/pip (Pillow); never torch, transformers, or a model folder.
- Handoff acceptance is a **manual** run by the fine-tune developer, using the same file with an explicit
  `--tokenizer <model folder>` mode, in the environment that has transformers.

## Approaches considered

| Approach | Gains | Costs |
|---|---|---|
| **A. Self-contained single file + canonical form + conformance test** (chosen) | Provider self-serves with no repo; rules pinned by tests; handoff has an account | Rules exist twice; text token length needs the developer's environment |
| B. Documentation and samples only | Cheapest | Prose expires; the three silent paths stay silent |
| C. Pip-installable validator | One copy of the rules | Provider must configure an install and a Python environment |
| D. Remote validation service | One copy of the rules, zero local deps | Infrastructure; data leaves the machine |
| E. Contract derived by static analysis (AST) | No duplicated rules | Cannot express cross-field semantics; becomes the mechanism instead of a guard |

E is rejected as the medium and retained as a completeness guard (see *Drift guards*).

## Design

### Artifacts

1. `finetune/check_data.py` — the single provider-facing file.
2. `docs/zh/finetune_data.md` — the normative contract for the provider, one section per rule id, with valid and invalid
   examples; the checker's user manual.
3. Handoff report — terminal summary plus `--report <path>.json`, archived next to the data (note: `data/` is
   gitignored, so the report is an archived artifact, not a tracked file).
4. `tests/test_check_data.py` — the conformance suite.

### Interface

```
python3 check_data.py train.jsonl --render 5             # model's view of 5 sampled records
python3 check_data.py train.jsonl --canonical out.jsonl  # normalized records
python3 check_data.py train.jsonl --report train.report.json
python3 check_data.py train.jsonl --against dev.jsonl     # leak check, both files
python3 check_data.py train.jsonl --tokenizer MODEL       # exact prompt lengths (developer side only)
python3 check_data.py dev.jsonl --role dev --emit-eval eval.jsonl
```

- Exit status 0 only when no blocking finding exists (observation findings do not block).
- `--against` takes the other split; `--role dev` applies the dev-only rules to each file it is given.
- Defaults for `--max-length` (8192), `--min-pixels` (65536), `--max-pixels` (1048576) and the option/image caps equal
  the training defaults; any override is stamped into the report, because training must be run with the same values.
- Two output classes, never merged: a **data finding** (`E…`/`W…` with file, line, question, and the fix) and an
  **internal error** (traceback plus "this is a checker bug; send this record back"). A checker crash must never read
  as a data error.

### Two kinds of rule, labelled

The trainer is a library and is deliberately more permissive than the contract: it must keep working for partially
labelled files and for inference. So each rule carries one tag:

- `[trainer]` — the trainer also raises or skips here. The checker must reproduce it exactly.
- `[contract]` — the checker is stricter on purpose; the trainer would accept the record. Allowed only for input that
  would train a wrong or meaningless prompt, and each one is argued in `docs/zh/finetune_data.md`.

This is why the conformance suite proves **strictness, not equality**: for every fixture, if the trainer rejects or
skips it the checker must reject it; if the checker accepts it the trainer must accept it with the same canonical
output. Whether the checker also rejects something the trainer accepts is a per-rule decision, never an accident.

### What "the model's view" means

`messages()` (`startlux_decide/prompt.py:129`) needs no tokenizer: `from_systemone` → `state_text` → `option_lines` is
pure Python. The checker reproduces that rendering, including `state_text`'s compact-JSON form with `_index` injected
into arrays of 8 or more (`prompt.py:41`), and can therefore print, byte for byte, the system and user text the model
will read, with letter-prefixed options and `<|image_pad|>` placeholders. This is the lossless part of the transfer:
the prompt body is exactly reproducible offline; only the chat-template skeleton and tokenization are not.

`--render` shows it. `--canonical` writes the normalized record the trainer consumes: targets expanded into option
order and normalized to sum 1, state rendered to text, images kept as data URIs, an `id` added.

### Blocking findings (`E`)

Shape and identity:

- `E01 [trainer]` a non-blank line that is unparseable, or is not a JSON object. Blank lines are skipped, as the
  trainer skips them.
- `E02 [contract]` `state` missing, `null`, `{}`, `[]`, or blank text. The trainer renders all of these to empty
  evidence, silently; the contract requires evidence. Any other non-string JSON value is accepted, except in `--role
  dev`, where eval conversion only takes a string or an object.
- `E03 [trainer]` `images` not a list, more than 16 entries, an entry that is not a base64 string or data URI
  (`images.decode`), or bytes Pillow cannot decode. Pillow is required for this; there is no fallback.
- `E04 [trainer]` `<image>` marks in `state` disagree with the number of `images` (`images.place_images:37`).
- `E05 [trainer]` an image whose sides differ by more than a factor of 200 (`images.smart_resize:52`).
- `E06 [trainer]` `questions` missing or not an object; a question spec that is not an object. An empty question key
  is a `W03` observation: the trainer trains it, and the key is never rendered, so it cannot be a blocking rule.

Question shape (all mirrored from `prompt.validate_row`, `prompt.py:75`):

- `E07 [trainer]` `type` present but not `choice`, `noul`, `bool`, or `score`. An absent `type` means `choice`, as in
  `from_systemone`; `bool` is the documented alias of `noul`. Both defaults are documented, not errors.
- `E08 [trainer]` `choice`: `criteria` neither an object nor a list, or an option count outside 2..26. Note the split:
  more than 26 options is a *skip* in the trainer, one option is a *raise*.
- `E09 [trainer]` `score`: `criteria` neither a list nor an object of levels, or a level count outside 2..10
  (`prompt.MAX_LEVELS`, not 26). Level ids always render as `"0"`..`"n-1"`, so there is no id rule to check.
- `E10 [trainer]` an option id that is empty, contains a newline, or is longer than 200 characters
  (`prompt.py:91`). Duplicate ids cannot survive JSON parsing, so they are not a rule.
- `E11 [trainer]` blank `instructions` on `choice`/`score`. `noul` without instructions gets
  `NOUL_INSTRUCTIONS`, which is documented; a blank instruction string is still rejected.

Targets:

- `E12 [contract]` a question with no entry in `targets`.
- `E13 [trainer]` a `target` key that is not one of the question's option ids. This is today's silent 0.0 fill; it
  becomes a raise in the loader as well.
- `E14 [trainer]` a target value that is not a number, or whose total mass is zero.
- `E15 [contract]` a negative target value. The trainer normalizes it; the contract does not ship negative mass.
- `E16 [contract]` `--role dev`: two or more options tied within 1e-9 at the top (`to_eval_data.py:55`). Training
  tolerates ties; eval-data conversion refuses them.
- `E17 [contract]` `--against`: train/dev `state` overlap. Key: the rendered `state_text` plus the sorted digests of
  the record's image bytes, because that is what the model distinguishes. A record with the same text and different
  images is not a leak.

Length:

- `E18 [trainer]` effective prompt length over `--max-length`. **Only decidable in `--tokenizer` mode.** The trainer's
  length is `len(encoded prompt) + sum(image tokens - 1)`; image tokens are exact once the folder's
  `patch_size × spatial_merge_size` is known, the encoded prompt is not. Without `--tokenizer` the report records
  `"length_verified": false`, lists this rule as unverified, and never counts it as passing.
- `E19 [contract]` a target value that is not finite (`NaN`, `Infinity`). Both sides accept it today and the training
  loss becomes NaN; a probability is a finite number.

### Observation findings (`W`) — reported, never blocking

- `W01` the same question key has a different `type`, `instructions`, or option set in different records. Varying
  option sets is legal (the server accepts any option list per request), but it weakens learning; list the conflicting
  lines.
- `W02` an option id that no target uses, counted per question key (an id used by another question does not count).
- `W03` unknown top-level keys (the trainer ignores them), `noul` `criteria` that the renderer discards, an empty
  question key, duplicate `state` within a split (text only; `E17` also compares image bytes), a `state` far longer
  than the dataset median.
- `W04` `--role dev`: fewer than 200 questions of a type (calibration then falls back to the pooled temperature,
  `docs/finetuning.md:143`).

### The length gap, stated plainly

Offline, exact text token counts are impossible without the model's tokenizer and chat template. The design closes
what it can and names what it cannot:

- Image token counts are exact with Pillow alone — `(h/32) × (w/32)` after the same `smart_resize` arithmetic — so a
  record whose images alone exceed `--max-length` is caught without the model.
- The full prompt length needs `--tokenizer`, which the developer runs at acceptance. In that mode the checker
  reproduces `plain_prompt` end to end: `place_images` → `multimodal_messages` → the folder's `apply_chat_template`
  with `enable_thinking=False` → encode, then add `sum(tokens - 1)`. Tokenizing `messages()` text alone would
  miscount every image record, which is the failure class this design exists to remove.
- The report records whether that mode ran. `"length_verified": false` is a visible, archivable fact, not a silent
  pass.
- The loader's own skip counters are split (over 26 options vs too long) so the training log names the reason. The
  skip itself stays: the loader is a library, the gate is the contract.

### Loader changes (project side)

1. An unknown `target` key raises with file, line, and question instead of filling 0.0.
2. The printed summary separates the two skip reasons.
3. Module docstrings and `CHANGELOG.md` updated; `tests/test_finetune.py` extended for both.

No change: a question without a target stays skipped by the loader (useful for partially labelled files and
inference); blank `state` and negative probabilities stay accepted by the loader. The gate is stricter on purpose.

### Data flow

`raw record → validate → canonical record`; `--canonical` writes the canonical record, `--emit-eval` runs on the **raw**
record and mirrors `to_eval_data.convert` (which copies question specs verbatim and requires `state`), so its output can
be compared with `convert` byte for byte. The two outputs serve different consumers and are never chained.

### Drift guards (`tests/test_check_data.py`)

The file is loaded by path, the same way `tests/test_finetune.py:26` loads the trainer.

1. **Constants**: `MAX_OPTIONS`, `MAX_LEVELS`, `MAX_IMAGES`, `MIN_PIXELS`, `MAX_PIXELS`, the 32-pixel factor and the
   default `--max-length` in `check_data.py` equal the trainer's values. Narrowing fails CI.
2. **Rendering**: for a corpus covering all four types, absent `type`, criteria-as-list, score-as-object, a long list
   state (the `_index` path), images and blank instructions, the checker's rendered messages equal
   `startlux_decide.prompt.messages` exactly.
3. **Verdicts**: for every fixture the trainer raises or skips, the checker rejects; for every fixture the checker
   accepts, `load_items` also accepts it and produces the same option count, target vector, type and length.
4. **Image arithmetic**: computed token counts equal `images.load_media` for sizes inside and outside the pixel bounds
   and at the 200× aspect boundary.
5. **Dev conversion**: `--emit-eval` output equals `finetune/to_eval_data.convert` on the same fixtures.
6. **Rule completeness (AST)**: the scan covers the transitive call closure of `load_items` within `prompt`, `images`
   and `finetune_lora` (stopping at `transformers`); every `raise` and every skip site in that closure maps to a rule
   id in the checker's rule table, keyed by function and message literal. A new rejection site with no rule fails the
   test. The checker's imports are standard library plus `PIL`, and nothing from `startlux_decide`.
7. **The example**: `examples/finetune/sample.jsonl` passes with zero blocking findings.
8. **Both modes**: the `--tokenizer` path is tested with a stub tokenizer and, when `STARTLUX_TEST_MODEL_DIR` and
   transformers are present, against the real one; otherwise skipped, per repository convention.

### The report

`--report` writes machine-readable JSON: checker version and the commit it was cut from, a snapshot of every default
constant, the mode (`tokenizer` or not, `length_verified`), per-file and per-question counts (delivered / trainable /
blocked, with reasons), every blocking finding, every observation, and the model folder the acceptance used. It is
committed next to the data so the "what training actually consumed" number is never unreconstructable.

### Verification of this change

- `python3 -m unittest -q tests.test_check_data tests.test_finetune`
- `python3 finetune/check_data.py examples/finetune/sample.jsonl --render 2` → zero blocking findings, readable output
- One manual acceptance run on a real dataset with `--tokenizer <model folder>`; the report is the evidence.

## Left to the builder

- Exact rule ids, message wording, and the report's JSON key names.
- Whether observation findings need a `--strict` switch that promotes them to blocking.
- Whether `finetune/to_eval_data.py` stays as an internal tool (proposal: keep it; the checker carries the same
  conversion for the provider).
- The internal structure of `check_data.py` (one pass vs. two). One streaming pass with a small per-record validator is
  preferred; datasets are read once.
