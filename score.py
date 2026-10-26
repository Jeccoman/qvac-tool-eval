"""Scores harness results against the cases.

    python3 evals/score.py evals/results/*.jsonl

Overseer cases list acceptable answers as `expected` (`tool: null` means no
tool should be called). BFCL cases are scored with BFCL's AST rules: one call
to the expected function, no unknown parameters, every parameter without an
empty-string option present, and every value in its list of accepted values.
"""

import collections
import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).parent
CASES = {}
for path in (ROOT / "cases").glob("*.jsonl"):
    for line in path.read_text().splitlines():
        if line.strip():
            case = json.loads(line)
            CASES[case["id"]] = case


def standardize(value):
    return re.sub(r"[ ,./\-_*^]", "", value.lower()).replace('"', "'")


def value_matches(actual, accepted):
    if isinstance(accepted, str) and isinstance(actual, str):
        return standardize(actual) == standardize(accepted)
    if isinstance(accepted, bool) or isinstance(actual, bool):
        return actual is accepted
    if isinstance(accepted, (int, float)) and isinstance(actual, (int, float)):
        return float(actual) == float(accepted)
    if isinstance(accepted, list) and isinstance(actual, list):
        return len(actual) == len(accepted) and all(map(value_matches, actual, accepted))
    if isinstance(accepted, dict) and isinstance(actual, dict):
        return all(
            k in actual and any(value_matches(actual[k], a) for a in (v if isinstance(v, list) else [v]))
            or (k not in actual and isinstance(v, list) and "" in v)
            for k, v in accepted.items()
        ) and set(actual) <= set(accepted)
    return actual == accepted


def bfcl_check(case, calls):
    if case["bfcl_expected"] is None:
        return (not calls, "called a tool on an irrelevant prompt")
    if len(calls) != 1:
        return (False, "no tool call" if not calls else "more than one call")
    expected = case["bfcl_expected"][0]
    call = calls[0]
    if call["name"] not in expected:
        return (False, "wrong function")
    accepted = expected[call["name"]]
    args = call["arguments"] if isinstance(call["arguments"], dict) else {}
    for name in args:
        if name not in accepted:
            return (False, "unexpected parameter")
    for name, values in accepted.items():
        if name not in args:
            if "" not in values:
                return (False, "missing parameter")
        elif not any(value_matches(args[name], v) for v in values if v != ""):
            return (False, "wrong value")
    return (True, "")


def arg_matches(actual, rule):
    if "equals" in rule:
        return value_matches(actual, rule["equals"])
    if "contains_any" in rule:
        text = json.dumps(actual).lower() if not isinstance(actual, str) else actual.lower()
        return any(s.lower() in text for s in rule["contains_any"])
    return True


def overseer_check(case, calls):
    for option in case["expected"]:
        if option["tool"] is None:
            if not calls:
                return (True, "")
            continue
        if not calls or calls[0]["name"] != option["tool"]:
            continue
        args = calls[0]["arguments"] if isinstance(calls[0]["arguments"], dict) else {}
        if all(k in args and arg_matches(args[k], r) for k, r in option.get("args", {}).items()):
            return (True, "")
        return (False, "right tool, wrong arguments")
    if not calls:
        return (False, "no tool call")
    if all(o["tool"] is None for o in case["expected"]):
        return (False, "called a tool when none was needed")
    return (False, f"wrong tool ({calls[0]['name']})")


def score(result):
    case = CASES[result["id"]]
    if "error" in result:
        return (False, "request error")
    calls = result.get("tool_calls") or []
    return bfcl_check(case, calls) if "bfcl_expected" in case else overseer_check(case, calls)


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def main(paths):
    for path in paths:
        results = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
        if not results:
            continue
        by_cat = collections.defaultdict(lambda: [0, 0])
        reasons = collections.Counter()
        failures = []
        for r in results:
            ok, reason = score(r)
            cat = CASES[r["id"]].get("category", "?")
            by_cat[cat][0] += ok
            by_cat[cat][1] += 1
            if not ok:
                reasons[reason] += 1
                failures.append((r["id"], reason))
        total = sum(c[0] for c in by_cat.values())
        lat = [r["latency_ms"] for r in results]
        print(f"## {pathlib.Path(path).name}")
        print(f"model {results[0]['model']}: {total}/{len(results)} = {100 * total / len(results):.1f}%  "
              f"latency p50 {statistics.median(lat) / 1000:.1f}s p95 {percentile(lat, 95) / 1000:.1f}s")
        for cat, (ok, n) in sorted(by_cat.items()):
            print(f"  {cat:<12} {ok:>3}/{n:<3} {100 * ok / n:5.1f}%")
        for reason, n in reasons.most_common():
            print(f"  fail: {reason:<36} {n}")
        if "-v" in sys.argv:
            for case_id, reason in failures:
                print(f"    {case_id}: {reason}")
        print()


if __name__ == "__main__":
    main([p for p in sys.argv[1:] if p != "-v"])
