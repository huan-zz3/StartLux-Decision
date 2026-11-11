"""Turn StartLux-Decision fine-tuning data into the request file ``startlux_decide eval --data`` takes.

    python finetune/to_eval_data.py data.jsonl -o eval.jsonl
    startlux-decide eval --data eval.jsonl --url http://127.0.0.1:8090

Training data, one JSON object per line, is ``{"state", "questions", "targets"}``; targets hold one probability per
option (a one-hot label or annotator vote shares).  Eval data is ``{"id", "state", "questions", "gold"}`` with a single
label per question.  This script picks each question's most probable target option and normalizes it for the scorer:

* ``choice``: the option id string, as is.
* ``noul`` / ``bool``: ``"true"`` or ``"false"``; any other target key is an error, not a silent coercion.
* ``score``: an ``int`` level index.  ``startlux_decide.evaluate.gold_label`` accepts an int or a criteria key there;
  a numeric string such as ``"1"`` is rejected, so the conversion to int happens here.

Questions without a target (no key in ``targets``, or no ``targets`` object at all) are skipped and counted.  A record
whose questions are all skipped is not written and is counted separately.  A top-probability tie (within 1e-9 across
two or more options) is ambiguous and raises with the file, line, question and tied options; this script never picks the
first one.  ``images`` is record-level and is copied through unchanged.  A record id, when present, is copied; otherwise
the id is ``line-<n>`` with the 1-based line number.

Every input line must be a JSON object with a ``state`` (string or object) and a ``questions`` object.  Blank lines and
malformed JSON are errors with their line number, because silently dropping a row would quietly change the eval set.
The output is written only after the whole input converts, so a failed run leaves no half-written file.  Standard
library only.
"""
from __future__ import annotations

import argparse
import json
import sys

TIE_EPS = 1e-9


def _error(path, line, message):
    """An input problem tagged with the file and 1-based line it came from."""
    return ValueError(f"{path}:{line}: {message}")


def gold_label(path, line, key, spec, target):
    """The single gold label of one question, normalized for startlux_decide.evaluate."""
    if not isinstance(spec, dict):
        raise _error(path, line, f'question {key!r}: the question must be an object')
    if not isinstance(target, dict) or not target:
        raise _error(path, line, f"question {key!r}: target must be a non-empty object mapping option ids to "
                                 "probabilities")
    values = {}
    for option, probability in target.items():
        try:
            values[option] = float(probability)
        except (TypeError, ValueError):
            raise _error(path, line, f"question {key!r}: target probabilities must be numbers, not {probability!r}") \
                from None
    top = max(values.values())
    tied = [option for option, probability in values.items() if top - probability <= TIE_EPS]
    if len(tied) > 1:
        raise _error(path, line, f"question {key!r}: tied target probabilities for options "
                                 f"{', '.join(map(repr, tied))}")
    option = tied[0]
    kind = spec.get("type", "choice")
    if kind == "score":
        try:
            return int(option)
        except (TypeError, ValueError):
            raise _error(path, line, f"question {key!r}: a score target option must be a level id 0..n-1, not "
                                     f"{option!r}") from None
    if kind in ("noul", "bool"):
        if option not in ("true", "false"):
            raise _error(path, line, f"question {key!r}: a yes/no target option must be 'true' or 'false', not "
                                     f"{option!r}")
        return option
    return option


def convert(path, out_path):
    """-> (requests written, questions skipped, records dropped); raises ValueError with file:line on bad input."""
    requests, skipped, dropped = [], 0, 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise _error(path, n, f"invalid JSON: {e}") from None
            if not isinstance(rec, dict):
                raise _error(path, n, "each line must be a JSON object")
            if "state" not in rec:
                raise _error(path, n, 'each line needs a "state"')
            if not isinstance(rec["state"], (str, dict)):
                raise _error(path, n, "state must be a string or an object")
            if not isinstance(rec.get("questions"), dict):
                raise _error(path, n, 'each line needs a "questions" object')
            targets = rec.get("targets")
            if targets is None:
                targets = {}
            elif not isinstance(targets, dict):
                raise _error(path, n, "targets must be an object")
            questions, gold = {}, {}
            for key, spec in rec["questions"].items():
                target = targets.get(key)
                if target is None:
                    skipped += 1
                    continue
                questions[key] = spec
                gold[key] = gold_label(path, n, key, spec, target)
            if not questions:
                dropped += 1
                continue
            item = {"id": rec.get("id", f"line-{n}"), "state": rec["state"], "questions": questions, "gold": gold}
            if "images" in rec:
                item["images"] = rec["images"]
            requests.append(item)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in requests:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(requests), skipped, dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="fine-tuning JSONL (state / questions / targets)")
    ap.add_argument("--out", "-o", required=True, help="eval JSONL to write")
    args = ap.parse_args(argv)
    try:
        wrote, skipped, dropped = convert(args.data, args.out)
    except (ValueError, OSError) as e:
        print(e, file=sys.stderr)
        return 1
    print(f"wrote {wrote} requests ({skipped} question{'s' if skipped != 1 else ''} skipped, "
          f"{dropped} record{'s' if dropped != 1 else ''} dropped) to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
