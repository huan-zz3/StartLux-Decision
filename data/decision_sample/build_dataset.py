"""Build a small, decision-index-style fixture set from mghafiri/decision-model-scenarios.

Outputs (all UTF-8, one JSON object per line):
  decision_sample_300.jsonl          core format: string state, choice + noul, flat option->prob targets
  decision_sample_extended_40.jsonl  wider valid space: object/array state, null criteria, score type
  invalid_cases.jsonl                20 rows, one injected defect each (negative tests)
  invalid_cases_expected.json        sidecar: expected error per invalid line

Usage:
  python build_dataset.py                # reads ./data_raw/*.jsonl
  python build_dataset.py --minimal      # drop id/domain from the core file
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 20261111
HERE = Path(__file__).resolve().parent
RAW = HERE / "data_raw"
OUT = HERE / "out"


def labelize(key: str) -> str:
    """Mechanical human-readable label for an option key that upstream left as null."""
    text = key.replace("_", " ").replace("-", " ").strip()
    return " ".join(w[:1].upper() + w[1:] for w in text.split())


SCORE_LABEL_KEYS = ("level", "label", "rating", "situation", "decision")


def score_level_text(item) -> str:
    """Upstream score levels are either plain strings or dicts under 5 different label keys."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in SCORE_LABEL_KEYS:
            if isinstance(item.get(key), str) and item[key].strip():
                text = item[key].strip()
                if isinstance(item.get("definition"), str) and item["definition"].strip():
                    text = f"{text}: {item['definition'].strip()}"
                return text
    return str(item)


def normalize(probs: dict[str, float]) -> dict[str, float]:
    """Round to 6dp and put the rounding residue on the largest option so the sum is exactly 1."""
    rounded = {k: round(float(v), 6) for k, v in probs.items()}
    if not rounded:
        return rounded
    residue = round(1.0 - sum(rounded.values()), 6)
    top = max(rounded, key=lambda k: rounded[k])
    rounded[top] = round(rounded[top] + residue, 6)
    return rounded


def load_raw() -> list[dict]:
    rows = []
    for split in ("train", "validation", "test"):
        path = RAW / f"{split}.jsonl"
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_split"] = split
                rows.append(row)
    return rows


def harden(probs: dict[str, float]) -> dict[str, float]:
    """Collapse a distribution to a one-hot map, for validators that expect a single gold key."""
    top = max(probs, key=lambda k: probs[k])
    return {k: (1.0 if k == top else 0.0) for k in probs}


def build_core_row(row: dict, keep_meta: bool = True, hard: bool = False) -> dict:
    """choice + noul only, string state, targets flattened to {option: probability}."""
    questions_in = json.loads(row["questions"])
    targets_in = json.loads(row["targets"])

    questions, targets = {}, {}
    for key, q in questions_in.items():
        if q["type"] == "choice":
            criteria = {}
            for opt, desc in q["criteria"].items():
                criteria[opt] = desc if isinstance(desc, str) and desc.strip() else labelize(opt)
            questions[key] = {
                "type": "choice",
                "instructions": q["instructions"],
                "criteria": criteria,
            }
            probs = normalize(targets_in[key]["probabilities"])
            targets[key] = harden(probs) if hard else probs
        elif q["type"] == "noul":
            questions[key] = {"type": "noul", "instructions": q["instructions"]}
            p_yes = float(targets_in[key]["noul"])
            probs = normalize({"true": p_yes, "false": 1.0 - p_yes})
            targets[key] = harden(probs) if hard else probs
        # score questions are dropped from the core file

    out = {"state": row["state"], "questions": questions, "targets": targets}
    if keep_meta:
        out["id"] = row["id"]
        out["domain"] = row["domain"]
    return out


def build_extended_row(row: dict) -> dict:
    """Full valid space: any state type, null criteria preserved, score questions kept."""
    questions_in = json.loads(row["questions"])
    targets_in = json.loads(row["targets"])

    questions, targets = {}, {}
    for key, q in questions_in.items():
        if q["type"] == "choice":
            questions[key] = {
                "type": "choice",
                "instructions": q["instructions"],
                "criteria": dict(q["criteria"]),  # nulls stay null: option keys only
            }
            targets[key] = normalize(targets_in[key]["probabilities"])
        elif q["type"] == "noul":
            questions[key] = {"type": "noul", "instructions": q["instructions"]}
            p_yes = float(targets_in[key]["noul"])
            targets[key] = normalize({"true": p_yes, "false": 1.0 - p_yes})
        else:  # score
            levels = q["criteria"]  # list[str] or list[dict] with 8 different label keys upstream
            questions[key] = {
                "type": "score",
                "instructions": q["instructions"],
                "criteria": [
                    {"level": str(i), "definition": score_level_text(lv)}
                    for i, lv in enumerate(levels)
                ],
            }
            raw = targets_in[key]["probabilities"]
            targets[key] = normalize({str(i): float(raw[str(i)]) for i in range(len(levels))})

    return {
        "state": json.loads(row["state"]),
        "questions": questions,
        "targets": targets,
        "id": row["id"],
        "domain": row["domain"],
    }


def stratified_pick(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Proportional-by-domain allocation with a floor of 1, then a shuffled global order."""
    by_domain = defaultdict(list)
    for row in pool:
        by_domain[row["domain"]].append(row)
    if n <= len(by_domain):  # too few rows to give every domain a slot
        return rng.sample(pool, n)

    quota = {d: max(1, round(n * len(rs) / len(pool))) for d, rs in by_domain.items()}
    # trim or top up so the total is exactly n
    while sum(quota.values()) > n:
        donor = max((d for d, q in quota.items() if q > 1), key=lambda d: quota[d])
        quota[donor] -= 1
    while sum(quota.values()) < n:
        receiver = max(by_domain, key=lambda d: len(by_domain[d]) - quota[d])
        quota[receiver] += 1

    picked = []
    for domain, count in quota.items():
        picked.extend(rng.sample(by_domain[domain], min(count, len(by_domain[domain]))))
    rng.shuffle(picked)
    return picked


def invalid_cases(valid: list[dict]) -> tuple[list[dict], list[dict]]:
    """One defect per row, so a validator can be checked for both rejection and the error it reports."""
    cases, expectations = [], []
    used: set[int] = set()

    def ifirst(row):
        return next(iter(row["questions"].values()))

    def is_choice(row) -> bool:
        q = ifirst(row)
        return q["type"] == "choice" and len(q.get("criteria", {})) >= 3

    def is_noul(row) -> bool:
        return any(q["type"] == "noul" for q in row["questions"].values())

    def is_multi(row) -> bool:
        return len(row["questions"]) >= 2 and is_choice(row)

    def is_any(row) -> bool:
        return True

    def add(pred, defect: str, expected: str, mutate, also: str | None = None) -> None:
        idx = next((i for i, r in enumerate(valid) if pred(r) and i not in used),
                   next(i for i, r in enumerate(valid) if pred(r)))
        used.add(idx)
        row = json.loads(json.dumps(valid[idx]))
        mutate(row)
        cases.append(row)
        entry = {
            "line": len(cases),
            "id": f"invalid-{len(cases):02d}",
            "base_row_id": row.get("id"),
            "defect": defect,
            "expected": expected,
        }
        if also:
            entry["also_violates"] = also
        expectations.append(entry)

    def first_q(row):
        return row["questions"][next(iter(row["questions"]))]

    def first_option(row):
        return next(iter(row["targets"][next(iter(row["questions"]))]))

    def noul_key(row) -> str:
        return next(k for k, q in row["questions"].items() if q["type"] == "noul")

    add(is_any, "missing 'state'", "missing required field: state", lambda r: r.pop("state"))
    add(is_any, "missing 'targets'", "missing required field: targets", lambda r: r.pop("targets"))
    add(is_any, "'questions' is a list, not an object", "questions must be a mapping",
        lambda r: r.update(questions=list(r["questions"].values())))
    add(is_choice, "question missing 'type'", "question.type is required",
        lambda r: first_q(r).pop("type"))
    add(is_choice, "unknown question type", "question.type must be one of choice|noul|score",
        lambda r: first_q(r).update(type="multichoice"))
    add(is_choice, "choice has a single option", "choice requires >= 2 criteria",
        lambda r: first_q(r).update(criteria=dict(list(first_q(r)["criteria"].items())[:1])),
        also="target keys no longer match criteria keys")
    add(is_choice, "choice missing 'criteria'", "choice.criteria is required",
        lambda r: first_q(r).pop("criteria"))
    add(is_choice, "empty 'instructions'", "instructions must be non-empty",
        lambda r: first_q(r).update(instructions=""))
    add(is_choice, "target options do not match criteria keys", "target keys must equal criteria keys",
        lambda r: r["targets"].update({next(iter(r["questions"])): {"__unknown_option__": 1.0}}))
    add(is_choice, "probability does not sum to 1", "target distribution must sum to 1",
        lambda r: r["targets"].update({next(iter(r["questions"])): {
            k: min(1.0, round(v * 1.4, 6))
            for k, v in r["targets"][next(iter(r["questions"]))].items()}}))
    add(is_choice, "negative probability", "probabilities must be in [0, 1]",
        lambda r: r["targets"].update({next(iter(r["questions"])): {
            **r["targets"][next(iter(r["questions"]))], first_option(r): -0.2}}),
        also="distribution no longer sums to 1")
    add(is_choice, "probability as string", "probabilities must be numbers",
        lambda r: r["targets"].update({next(iter(r["questions"])): {
            k: str(v) for k, v in r["targets"][next(iter(r["questions"]))].items()}}))
    add(is_multi, "target missing a question key", "every question needs a target",
        lambda r: r["targets"].pop(list(r["questions"])[-1]))
    add(is_any, "target for a question that does not exist", "targets must not reference unknown questions",
        lambda r: r["targets"].update({"_ghost_question": {"a": 1.0}}))
    add(is_any, "state is a number", "state must be a string or JSON object", lambda r: r.update(state=42))
    add(is_any, "empty question key", "question keys must be non-empty",
        lambda r: r["questions"].update({"": r["questions"].pop(next(iter(r["questions"])))}))
    add(is_any, "'targets' is a list", "targets must be a mapping", lambda r: r.update(targets=[r["targets"]]))
    add(is_noul, "noul question carries 'criteria'", "noul must not define criteria",
        lambda r: r["questions"].update({noul_key(r): {
            "type": "noul", "instructions": "Is this true?", "criteria": {"a": "x", "b": "y"}}}),
        also="noul target keys no longer match criteria keys")
    add(is_noul, "noul target uses yes/no keys", "noul target keys must be true/false",
        lambda r: r["targets"].update({noul_key(r): {"yes": 0.6, "no": 0.4}}))
    add(is_choice, "probability above 1", "probabilities must be in [0, 1]",
        lambda r: r["targets"].update({next(iter(r["questions"])): {
            **r["targets"][next(iter(r["questions"]))], first_option(r): 1.5}}),
        also="distribution no longer sums to 1")

    return cases, expectations


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def core_eligible(row: dict) -> bool:
    """String state, at least one choice question, plain-string instructions everywhere.

    Keeps the core file inside the narrowest shape: 9,407 of the 9,716 upstream questions use
    string instructions, the remaining 309 wrap them in an object; those live in the extended file.
    """
    if not isinstance(json.loads(row["state"]), str):
        return False
    questions = json.loads(row["questions"]).values()
    if not any(q["type"] == "choice" for q in questions):
        return False
    return all(isinstance(q.get("instructions"), str) and q["instructions"].strip()
               for q in json.loads(row["questions"]).values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimal", action="store_true", help="drop id/domain from the core file")
    parser.add_argument("--n", type=int, default=300, help="core rows to emit (default 300)")
    parser.add_argument("--hard-labels", action="store_true",
                        help="collapse targets to one-hot (argmax = 1.0) instead of full distributions")
    args = parser.parse_args()

    rng = random.Random(SEED)
    rows = load_raw()
    OUT.mkdir(exist_ok=True)

    core_pool = [r for r in rows if core_eligible(r)]
    core = [build_core_row(r, keep_meta=not args.minimal, hard=args.hard_labels)
            for r in stratified_pick(core_pool, args.n, rng)]
    write_jsonl(OUT / "decision_sample_300.jsonl", core)

    ext_pool = [r for r in rows if not isinstance(json.loads(r["state"]), str)]
    with_score = [r for r in ext_pool if any(q["type"] == "score" for q in json.loads(r["questions"]).values())]
    extended = [build_extended_row(r) for r in rng.sample(with_score, 30)]
    extended += [build_extended_row(r) for r in rng.sample(
        [r for r in ext_pool if r not in with_score], 10)]
    rng.shuffle(extended)
    write_jsonl(OUT / "decision_sample_extended_40.jsonl", extended)

    cases, expectations = invalid_cases(core)
    write_jsonl(OUT / "invalid_cases.jsonl", cases)
    (OUT / "invalid_cases_expected.json").write_text(
        json.dumps(expectations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"core      {len(core):>3} rows  -> out/decision_sample_300.jsonl")
    print(f"extended  {len(extended):>3} rows  -> out/decision_sample_extended_40.jsonl")
    print(f"invalid   {len(cases):>3} rows  -> out/invalid_cases.jsonl")


if __name__ == "__main__":
    main()
