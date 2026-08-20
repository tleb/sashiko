# Bug T4: one failed patch review, zero output

"T4" labels the pair of defects described here. They surfaced when a
review stage hit the `ai.max_interactions` limit during a local review.

## The symptom

    $ sashiko review COMMITID --format json
    (stderr: "Session exceeded max turns limit", three times)
    Error: Review subprocess produced no output (exit code: 1)

The tool exited with no JSON on stdout. Before dying, it paid roughly
three times the cost of a full review. This page explains why, layer by
layer.

## Background: who runs a local review

A local review is a stack of five layers. Each layer trusted the one
below it and treated any error as fatal:

| Layer | Code | Role |
|---|---|---|
| Session | `SessionRunner::run`, `src/ai/session.rs` | one LLM conversation, with a turn budget |
| Stage | `Worker::run`, `src/worker/prompts.rs` | stages 1-11; stages 1-7 run in parallel |
| Patch | `review_single_patch`, `src/local_review.rs` | one patch, up to 3 whole-worker attempts |
| Patchset | `run_worker_in_worktree`, `src/local_review.rs` | applies patches, aggregates reviews |
| Binary | `src/bin/review.rs` | reads JSON on stdin, prints the result JSON |

The failure walked all five layers. Each layer made the outcome worse.

## The failure chain

1. **Session.** `SessionRunner::run` counts tool-call turns. At
   `turns > max_turns` it aborted the session with
   `anyhow::bail!("Session exceeded max turns limit")`. The
   conversation history lived only in a local variable, so the abort
   threw it away.

2. **Stage.** Stages 1-7 run through `futures::future::try_join_all`.
   The `?` on that call cancels the stages still running and drops the
   results of the stages already finished. One hungry stage starved its
   siblings.

3. **Patch.** `review_single_patch` ran the whole `Worker::run()` up to
   three times, on any error, with no classification. Turn exhaustion is
   deterministic: same patch, same budget, same tool loop. The extra
   attempts burned tokens for the same outcome. This is defect **T4a**.

4. **Patchset.** The drain loop did `results.push(res?)`. The first
   failed patch propagated out of `run_worker_in_worktree`. Reviews
   that had succeeded were discarded, and reviews still in flight were
   cancelled. This is defect **T4b**.

5. **Binary.** `src/bin/review.rs` returns the error through `?`. The
   process exits non-zero without printing: `print_worker_json` only
   runs on `Ok`.

The caller then reports "Review subprocess produced no output".
Correct, and useless.

## The fix

Two changes, both in how the error travels. The retry machinery inside
the AI stack did not change.

### T4a: classify turn exhaustion, skip the blind retry

`SessionRunner::run` now returns a typed `TurnLimitError` instead of an
ad-hoc `bail!`. `review_single_patch` checks it with
`e.downcast_ref::<TurnLimitError>()` and returns immediately.

The typed error follows the pattern of `classify_ai_error()`
(`src/ai/mod.rs`), which classifies provider errors the same way. Other
error classes keep the old behavior: three attempts.

### T4b: a failed patch becomes data, not an abort

The per-patch futures in `run_worker_in_worktree` fold an `Err` into
`failed_review_entry(p, &e)`. That is a result entry shaped like a
success, with `review: null` and the error string. The drain loop never
sees an `Err`, so it keeps draining.

After aggregation, `surface_review_failures` does two things:

- it annotates the matching entry of the `patches` array with the
  error,
- it sets a top-level `"error": "N of M patch reviews failed"`.

The CLI prints the JSON before it checks `result_has_error()` and exits
3 (`src/main.rs`). The contract is therefore: partial output first,
then a failing exit code. This mirrors the existing patch-application
failure path, which also returns output plus a top-level `"error"`.

The server (`src/reviewer.rs`) runs the review binary once per patch,
so the partial-failure path matters for local reviews only.

## Before / after

| Situation | Before | After |
|---|---|---|
| one patch hits the turn limit | 3x cost, no output, exit 1 | 1x cost, JSON with `error`, exit 3 |
| patch 2 of 5 fails | whole result discarded | patches 1, 3, 4, 5 reviewed, patch 2 annotated |

## What this does not fix

- The conversation history is still lost when a session aborts. The
  runner drops it on the error path.
- Hitting the limit at all. The planned follow-up injects a wrap-up
  instruction a few turns before the limit, then forces a final answer
  instead of aborting. See `designs/DESIGN_LOOP_PREVENTION.md`,
  section 2.3, "System Probe".
- Stage-level cancellation: one failed stage still cancels its siblings
  through `try_join_all`.

## Pointers

- `src/ai/session.rs` — `TurnLimitError`, `SessionRunner::run`
- `src/local_review.rs` — `review_single_patch`, `failed_review_entry`,
  `surface_review_failures`, `run_worker_in_worktree`
- `src/main.rs` — `result_has_error`, exit code 3
- `src/reviewer.rs` — server-side consumption, one patch per subprocess
- `designs/DESIGN_LOOP_PREVENTION.md` — budgets, stall detection, and
  the "System Probe" idea
- `docs/configuration.md` — the `ai.max_interactions` setting
