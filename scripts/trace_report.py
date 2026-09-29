#!/usr/bin/env python3
"""Summarise a sashiko trace dump (--trace JSONL) into where the time went.

Usage: scripts/trace_report.py [trace.jsonl]

Reads the event stream written by `sashiko review --trace` and reports:
  - wall-clock and per-event-kind counts
  - model-call time: totals, in-flight utilisation, max concurrency
  - time-to-first-byte and generation-time distributions (is it prefill?)
  - time lost waiting: concurrency gate, quota windows, backoff sleeps
  - turns per stage, per-turn duration, tokens
  - tool calls by name, duration and result size
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def percentile(values, pct):
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, int(len(values) * pct / 100))
    return values[idx]


def fmt_ms(ms: float) -> str:
    if ms >= 60_000:
        return f"{ms / 60_000:.1f}min"
    if ms >= 1_000:
        return f"{ms / 1_000:.1f}s"
    return f"{ms:.0f}ms"


def merge_intervals(intervals):
    """Union of (start, end) datetime intervals -> total seconds, max overlap."""
    if not intervals:
        return 0.0, 0
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda e: (e[0], e[1]))
    total = 0.0
    overlap = 0
    current = 0
    last = None
    for when, delta in events:
        if last is not None and current > 0:
            total += (when - last).total_seconds()
        current += delta
        overlap = max(overlap, current)
        last = when
    return total, overlap


def main(path):
    events = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"skipping malformed line: {line[:80]}")

    if not events:
        print("no events")
        return

    wall = (parse_ts(events[-1]["ts"]) - parse_ts(events[0]["ts"])).total_seconds()
    kinds = Counter(e["event"] for e in events)
    print(f"== Run =={path}")
    for key in ("provider", "model", "max_concurrent_requests", "deadline_secs",
                "patches", "api_timeout_secs", "review_concurrency"):
        for e in events:
            if e["event"] == "run_settings" and key in e:
                print(f"  {key}: {e[key]}")
                break
    print(f"  wall clock: {fmt_ms(wall * 1000)}, {len(events)} events")
    print("  events:", ", ".join(f"{k}={v}" for k, v in kinds.most_common()))

    # Model calls (each HTTP attempt) ---------------------------------------
    attempts = [e for e in events if e["event"] == "llm_attempt"]
    ok = [e for e in attempts if e.get("outcome") == "ok"]
    failed = [e for e in attempts if e.get("outcome") != "ok"]
    durations = [e["duration_ms"] for e in attempts]
    print(f"\n== Model calls ({len(attempts)}, {len(failed)} failed) ==")
    print(f"  duration   p50={fmt_ms(percentile(durations, 50))} "
          f"p90={fmt_ms(percentile(durations, 90))} "
          f"max={fmt_ms(max(durations, default=0))}")
    if ok:
        tokens_in = sum(e.get("tokens_in", 0) for e in ok)
        tokens_out = sum(e.get("tokens_out", 0) for e in ok)
        print(f"  tokens: in={tokens_in} out={tokens_out} on {len(ok)} ok calls")
    if failed:
        print(f"  failures by outcome: "
              f"{', '.join(f'{k}={v}' for k, v in Counter(e['outcome'] for e in failed).most_common())}")
        for e in failed[:5]:
            print(f"    {e['ts']} {e['outcome']} after {fmt_ms(e['duration_ms'])}: "
                  f"{e.get('error', '')[:120]}")

    intervals = []
    for e in attempts:
        end = parse_ts(e["ts"])
        start = end.timestamp() - e["duration_ms"] / 1000
        intervals.append((datetime.fromtimestamp(start, tz=timezone.utc), end))
    busy, max_overlap = merge_intervals(intervals)
    util = 100 * busy / wall if wall else 0
    print(f"  in-flight: {fmt_ms(busy * 1000)} of {fmt_ms(wall * 1000)} wall "
          f"({util:.0f}% of one slot), max {max_overlap} concurrent")

    # Streaming split: prefill vs generation --------------------------------
    streams = [e for e in events if e["event"] == "llm_stream"]
    if streams:
        ttfb = [e["ttfb_ms"] for e in streams]
        gen = [max(0, e["duration_ms"] - e["ttfb_ms"]) for e in streams]
        print(f"\n== Stream split ({len(streams)} streams) ==")
        print(f"  time to first byte p50={fmt_ms(percentile(ttfb, 50))} "
              f"p90={fmt_ms(percentile(ttfb, 90))}")
        print(f"  generation time    p50={fmt_ms(percentile(gen, 50))} "
              f"p90={fmt_ms(percentile(gen, 90))}")

    # Waiting ----------------------------------------------------------------
    waits = defaultdict(int)
    counts = defaultdict(int)
    for e in events:
        if e["event"] == "gate_wait":
            waits["concurrency gate"] += e["waited_ms"]
            counts["concurrency gate"] += 1
        elif e["event"] == "quota_wait":
            waits["quota window"] += e["waited_ms"]
            counts["quota window"] += 1
        elif e["event"] == "backoff":
            waits[f"backoff ({e['reason']})"] += e.get("sleep_ms", 0) or e.get("retry_after_ms", 0)
            counts[f"backoff ({e['reason']})"] += 1
    if waits:
        print("\n== Waiting (summed over calls; overlaps) ==")
        for name, ms in sorted(waits.items(), key=lambda kv: -kv[1]):
            print(f"  {name}: {fmt_ms(ms)} over {counts[name]} events")

    # Turns and stages --------------------------------------------------------
    turns = [e for e in events if e["event"] == "turn"]
    by_ctx = defaultdict(list)
    for e in turns:
        by_ctx[e.get("ctx") or "(no ctx)"].append(e)
    print(f"\n== Turns ({len(turns)} total) ==")
    for ctx, items in sorted(by_ctx.items()):
        total_ms = sum(e.get("duration_ms", 0) for e in items)
        outcomes = Counter(e["outcome"] for e in items)
        print(f"  {ctx}: {len(items)} turns, {fmt_ms(total_ms)} "
              f"({', '.join(f'{k}={v}' for k, v in outcomes.most_common())})")

    stages = [e for e in events if e["event"] == "stage" and e["phase"] == "end"]
    if stages:
        print(f"\n== Stages ({len(stages)} completed) ==")
        for e in stages:
            print(f"  {e.get('ctx') or '(no ctx)'} {e['name']}: "
                  f"{fmt_ms(e['duration_ms'])}, {e['tokens_in']} in / {e['tokens_out']} out")

    # Tools -------------------------------------------------------------------
    tools = [e for e in events if e["event"] == "tool"]
    if tools:
        by_tool = defaultdict(list)
        for e in tools:
            by_tool[e["tool"]].append(e)
        print(f"\n== Tools ({len(tools)} calls) ==")
        for name, items in sorted(by_tool.items(), key=lambda kv: -sum(i["duration_ms"] for i in kv[1])):
            total_ms = sum(i["duration_ms"] for i in items)
            chars = [i["result_chars"] for i in items]
            errors = sum(1 for i in items if not i.get("ok", True))
            print(f"  {name}: {len(items)} calls, {fmt_ms(total_ms)}, "
                  f"results p50={percentile(chars, 50)} max={max(chars, default=0)} chars"
                  + (f", {errors} errored" if errors else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "trace.jsonl")
