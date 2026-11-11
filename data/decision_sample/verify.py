"""Independent strict checker for the generated fixtures.

Deliberately re-implemented from the documented format instead of importing build_dataset,
so it can serve as a second opinion when comparing against your own validator.

  python verify.py
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
TOL = 1e-6


def fail(problems: list[str], line: int, msg: str) -> None:
    problems.append(f"line {line}: {msg}")


def check_row(row, line: int, problems: list[str], *, strict: bool) -> None:
    """strict=True -> core contract (string state, choice/noul only, string criteria)."""
    if not isinstance(row, dict):
        return fail(problems, line, "row is not a JSON object")

    if "state" not in row:
        return fail(problems, line, "missing state")
    state = row["state"]
    if strict:
        if not isinstance(state, str):
            fail(problems, line, f"state must be a string, got {type(state).__name__}")
        elif not state.strip():
            fail(problems, line, "state is empty")
    elif not isinstance(state, (str, dict, list)):
        fail(problems, line, f"state has unsupported type {type(state).__name__}")

    questions = row.get("questions")
    if not isinstance(questions, dict) or not questions:
        return fail(problems, line, "questions must be a non-empty object")

    targets = row.get("targets")
    if not isinstance(targets, dict) or not targets:
        return fail(problems, line, "targets must be a non-empty object")

    allowed = {"choice", "noul"} if strict else {"choice", "noul", "score"}
    for qkey, q in questions.items():
        if not qkey:
            fail(problems, line, "empty question key")
        if not isinstance(q, dict):
            fail(problems, line, f"{qkey!r}: question is not an object")
            continue
        qtype = q.get("type")
        if qtype not in allowed:
            fail(problems, line, f"{qkey!r}: bad type {qtype!r}")
            continue
        instr = q.get("instructions")
        if strict:
            if not isinstance(instr, str) or not instr.strip():
                fail(problems, line, f"{qkey!r}: instructions must be a non-empty string")
        elif isinstance(instr, dict):
            if not instr:
                fail(problems, line, f"{qkey!r}: instructions object is empty")
        elif not isinstance(instr, str) or not instr.strip():
            fail(problems, line, f"{qkey!r}: instructions must be a non-empty string or object")

        target = targets.get(qkey)
        if target is None:
            fail(problems, line, f"{qkey!r}: no target")
            continue
        if not isinstance(target, dict):
            fail(problems, line, f"{qkey!r}: target is not an object")
            continue

        if qtype == "choice":
            crit = q.get("criteria")
            if not isinstance(crit, dict):
                fail(problems, line, f"{qkey!r}: choice needs criteria object")
                continue
            if len(crit) < 2:
                fail(problems, line, f"{qkey!r}: choice needs >= 2 options, got {len(crit)}")
            for opt, desc in crit.items():
                if strict and (not isinstance(desc, str) or not desc.strip()):
                    fail(problems, line, f"{qkey!r}/{opt}: criteria value must be a non-empty string")
            expected = set(crit)
        elif qtype == "score":
            crit = q.get("criteria")
            if not isinstance(crit, list) or len(crit) < 2:
                fail(problems, line, f"{qkey!r}: score needs a criteria list with >= 2 levels")
                continue
            expected = {str(i) for i in range(len(crit))}
        else:
            if "criteria" in q:
                fail(problems, line, f"{qkey!r}: noul must not carry criteria")
            expected = {"true", "false"}

        if set(target) != expected:
            fail(problems, line, f"{qkey!r}: target keys {sorted(target)} != expected {sorted(expected)}")
            continue
        for opt, p in target.items():
            if not isinstance(p, (int, float)) or isinstance(p, bool):
                fail(problems, line, f"{qkey!r}/{opt}: probability is not a number ({p!r})")
                continue
            if not (0.0 <= p <= 1.0):
                fail(problems, line, f"{qkey!r}/{opt}: probability {p} outside [0,1]")
        total = sum(v for v in target.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
        if abs(total - 1.0) > TOL:
            fail(problems, line, f"{qkey!r}: probabilities sum to {total}, not 1")

    extra = set(targets) - set(questions)
    if extra:
        fail(problems, line, f"targets reference unknown questions: {sorted(extra)}")


def load(path: Path) -> list[tuple[int, object]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((i, json.loads(line)))
            except json.JSONDecodeError as exc:
                rows.append((i, exc))
    return rows


def report(path: Path, strict: bool) -> list[str]:
    problems: list[str] = []
    for line, payload in load(path):
        if isinstance(payload, json.JSONDecodeError):
            fail(problems, line, f"not valid JSON: {payload}")
            continue
        check_row(payload, line, problems, strict=strict)
    return problems


def main() -> None:
    print("== strict contract (core) ==")
    problems = report(OUT / "decision_sample_300.jsonl", strict=True)
    print("PASS" if not problems else "\n".join(problems))

    print("\n== relaxed contract (extended) ==")
    problems = report(OUT / "decision_sample_extended_40.jsonl", strict=False)
    print("PASS" if not problems else "\n".join(problems))

    print("\n== negative cases (must ALL fail) ==")
    expected = json.loads((OUT / "invalid_cases_expected.json").read_text(encoding="utf-8"))
    rows = load(OUT / "invalid_cases.jsonl")
    failures = 0
    for (line, payload), exp in zip(rows, expected):
        if isinstance(payload, json.JSONDecodeError):
            print(f"  invalid-{line:02d} rejected as malformed JSON  <- {exp['defect']}")
            continue
        found: list[str] = []
        check_row(payload, line, found, strict=True)
        if not found:
            failures += 1
            print(f"  !! line {line} ACCEPTED but should be rejected: {exp['defect']}")
        else:
            print(f"  line {line:02d} rejected  <- {exp['defect']}")
            print(f"            first error: {found[0].split(': ', 1)[1]}")
    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: {len(rows) - failures}/{len(rows)} negative cases rejected")


if __name__ == "__main__":
    main()
