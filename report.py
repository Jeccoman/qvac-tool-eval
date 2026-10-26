"""Prints the accuracy and speed tables from results/.

    python3 report.py
"""

import collections
import json
import pathlib
import statistics

import score

ROOT = pathlib.Path(__file__).parent


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def accuracy():
    rows = collections.defaultdict(lambda: {"runs": [], "latency": [], "malformed": 0, "categories": collections.defaultdict(list)})
    for path in sorted((ROOT / "results" / "final").glob("*.jsonl")):
        suite, runtime, _run = path.stem.split(".")
        results = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        passed = [score.score(r)[0] for r in results]
        row = rows[(runtime, suite)]
        row["runs"].append(100 * sum(passed) / len(passed))
        row["latency"] += [r["latency_ms"] / 1000 for r in results]
        row["malformed"] += sum(len(r.get("tool_errors") or []) for r in results)
        for result, ok in zip(results, passed):
            row["categories"][score.CASES[result["id"]]["category"]].append(ok)

    print("| Runtime | Suite | Mean | Runs | p50 latency | p95 latency | Malformed calls |")
    print("|---|---|---|---|---|---|---|")
    for (runtime, suite), row in sorted(rows.items()):
        print(f"| {runtime} | {suite} | {statistics.mean(row['runs']):.1f}% | "
              f"{', '.join(f'{v:.1f}' for v in row['runs'])} | {statistics.median(row['latency']):.1f} s | "
              f"{pct(row['latency'], 95):.1f} s | {row['malformed']} |")

    print("\n| Runtime | Held-out accuracy by category |")
    print("|---|---|")
    for (runtime, suite), row in sorted(rows.items()):
        if suite == "overseer_heldout":
            cells = ", ".join(f"{c} {100 * sum(v) / len(v):.0f}%" for c, v in sorted(row["categories"].items()))
            print(f"| {runtime} | {cells} |")


def speed():
    by_runtime = collections.defaultdict(list)
    for path in sorted((ROOT / "results").glob("speed*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                by_runtime[record["runtime"]].append(record)
    if not by_runtime:
        return

    print("\n| Runtime | Prompt tokens processed | Time to first token | Decode tok/s | Generated tokens | Total | Accuracy |")
    print("|---|---|---|---|---|---|---|")
    for runtime, records in sorted(by_runtime.items()):
        med = lambda key: statistics.median(r[key] for r in records if r.get(key) is not None)
        scored = [r for r in records if "tool_calls" in r]
        accuracy = (f"{100 * sum(score.score(r)[0] for r in scored) / len(scored):.1f}%" if scored else "n/a")
        print(f"| {runtime} | {med('prompt_tokens'):.0f} | {med('ttft_ms') / 1000:.2f} s | {med('decode_tps'):.1f} | "
              f"{med('generated_tokens'):.0f} | {med('total_ms') / 1000:.1f} s | {accuracy} |")


if __name__ == "__main__":
    accuracy()
    speed()
