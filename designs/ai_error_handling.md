# AI Provider Error Handling and Rate Limiting Design

## Objective
Currently, Sashiko's handling of AI provider errors (like 503 Service Unavailable, 529 Overloaded, and 429 Too Many Requests) is fragmented and localized. Some errors cause individual worker threads to sleep and retry locally (which can lead to thread exhaustion and timeout issues), while only specific quota errors trigger the global `QuotaManager`. 

The goal of this redesign is to:
1.  **Unify error handling:** All transient and rate-limiting errors from any LLM provider (Gemini, Claude) should trigger a global backoff/delay mechanism.
2.  **Global Pause:** When an AI provider is overloaded or returns a transient error, *all* AI interactions across *all* workers should be delayed to allow the provider to recover and to prevent exacerbating the issue.
3.  **Infinite Retries:** There should be no hard limit on the number of retries for transient/overloaded errors. The system should patiently wait for the provider to become available again.
4.  **Timeout Exemption:** The time spent waiting for global delays (due to quota or transient errors) must *not* be counted towards any per-worker or per-task timeouts. Worker timeouts should only apply to actual processing and execution time.

## Proposed Architecture

### 1. Unified Global AI Rate Limiter / Quota Manager
The existing `QuotaManager` in `src/ai/proxy.rs` (or potentially moved to a better location like `src/ai/mod.rs` or `src/ai/quota.rs`) will be expanded and renamed to `GlobalAiBackoffManager` (or similar).
*   It will manage a global `blocked_until` timestamp.
*   It will keep track of consecutive transient failures to implement global exponential backoff.
*   It will provide a way for any worker to report *any* type of retryable AI error (Quota Exceeded, Overloaded, Transient 5xx).
*   It will provide a `wait_for_access()` method that workers must call *before* attempting any AI request.
*   **Crucially**, `wait_for_access()` will return the `Duration` that the task actually spent sleeping.
*   **Global Pause:** When *any* worker receives a 503 (or other retryable error), it will report it to the manager. The manager will update the global `blocked_until` timestamp. This ensures that *all other workers* will pause and wait when they call `wait_for_access()`, preventing the system from bombarding an overloaded provider.

### 2. Error Categorization & Backoff Strategy
Errors from providers will be categorized into:
*   **Fatal Errors (400, 401, 403 - except Gemini cache expiration, 404):** These should fail immediately. No retries.
*   **Rate Limit / Quota (429):** Providers often include a `Retry-After` header. If present, the global delay is set to this value. If not, a default (e.g., 60s) is used.
*   **Overloaded / Transient (500, 502, 503, 504, 529):** These will trigger a global exponential backoff.
    *   The backoff will start small (e.g., 5s) and increase (e.g., multiply by 1.5x or 2x) on consecutive failures, up to a maximum cap (e.g., 5 minutes).
    *   A successful request will reset the consecutive failure counter.

### 3. Provider Client Changes (`src/ai/gemini.rs`, `src/ai/claude.rs`)
*   Remove the internal `loop` and retry logic (`max_retries`) from `post_request_with_retry` (Claude) and `generate_content` / `generate_content_with_cache` (Gemini).
*   The clients should simply execute the request once and return the result or the categorized error. Both clients should map HTTP status codes to specific Rust enum variants (e.g., `GeminiError::QuotaExceeded`, `GeminiError::TransientError`).

### 4. Worker Timeout Exemption (Dynamic Deadline)
In `src/reviewer.rs`, the AI interaction is currently wrapped in a fixed `tokio::time::timeout`:
```rust
let interaction_result = tokio::time::timeout(
    std::time::Duration::from_secs(settings.review.timeout_seconds),
    async { ... }
).await;
```
If `wait_for_access()` sleeps for 5 minutes inside that `async` block, the timeout will fire incorrectly.

*   **Solution:** We will replace `tokio::time::timeout` with a custom implementation or loop that tracks the *active* deadline. 
*   Because `tokio::time::timeout` cannot have its deadline extended once started, we should track the deadline manually using `tokio::time::sleep` combined with `tokio::select!`.
*   Alternatively, we can track the accumulated "sleep time" inside the async block.
*   **Best approach for Sashiko:** The AI loop is already yielding events (reading lines from the child process). 
*   We can create a struct `ActiveTimeout`:
```rust
struct ActiveTimeout {
    base_duration: Duration,
    total_sleep_added: Duration,
    start: Instant,
}
impl ActiveTimeout {
    fn new(base: Duration) -> Self { ... }
    fn add_sleep(&mut self, d: Duration) { self.total_sleep_added += d; }
    fn is_expired(&self) -> bool {
        self.start.elapsed() > (self.base_duration + self.total_sleep_added)
    }
    fn remaining(&self) -> Duration { ... }
}
```
*   And wrap the reading of lines in a `timeout` that is re-calculated for each line or each AI request. Or better yet, run the entire child process reading loop inside a custom future that we poll alongside a sleep future that gets its deadline updated.

Let's refine the timeout approach. Instead of wrapping the whole block in `tokio::time::timeout`, we can do:

```rust
let deadline = Instant::now() + Duration::from_secs(settings.review.timeout_seconds);
// ... inside the loop handling messages ...
    let resp_payload = loop {
        let slept = quota_manager.wait_for_access().await;
        deadline += slept; // Extend deadline by the time we slept!
        
        // Check timeout manually before heavy work, or rely on read timeouts
        if Instant::now() > deadline {
             break Err(anyhow!("Review tool timed out (active time exceeded)"));
        }
        
        match provider.generate_content(req.clone()).await {
            Ok(resp) => {
                quota_manager.report_success().await;
                break Ok(resp);
            }
            Err(e) => {
                if is_retryable(&e) {
                     quota_manager.report_error(e).await;
                     continue; // Loop indefinitely until success or fatal error
                }
                break Err(e);
            }
        }
    }
```
However, the `tokio::time::timeout` also guards against the *child process* hanging while producing stdout. We still need a timeout around the `lines.next_line().await`.

**Updated Timeout Strategy:**
1. We will use a mutable `deadline` variable: `let mut deadline = Instant::now() + Duration::from_secs(timeout_seconds);`
2. We wrap the `lines.next_line().await` (and other I/O) in `tokio::time::timeout_at(deadline.into(), ...)`
3. Whenever `quota_manager.wait_for_access().await` returns a `Duration > 0`, we do `deadline += duration`.

## Execution Plan

1.  **Refactor `QuotaManager`:**
    *   Rename to `QuotaManager` (keep name to minimize churn) but add tracking for consecutive transient failures.
    *   Update `wait_for_access` to return the `Duration` slept.
    *   Add `report_transient_error()` to trigger global exponential backoff.
    *   Add `report_success()` to reset transient error counter.
2.  **Update `gemini.rs` and `claude.rs`:**
    *   Remove local `retry_count` loops and `tokio::time::sleep`.
    *   Return errors directly (ensure 503/529 map to `TransientError` and 429 maps to `QuotaExceeded`).
3.  **Update Callers (`reviewer.rs`, `proxy.rs`):**
    *   Implement infinite loop around `provider.generate_content`.
    *   Call `wait_for_access()` and capture the sleep duration.
    *   Extend timeouts (`deadline += slept`).
    *   Report errors (`report_quota_error` or `report_transient_error`) or success (`report_success`).
4.  **Implement `timeout_at` logic in `reviewer.rs`:** Replace the single block `tokio::time::timeout` with loop-level `timeout_at` to allow deadline extension.

## Deadline degradation (implemented)

A stage whose review ran out of active-time budget mid-investigation used to
fail whole: every tool result it had gathered was dropped, and a review whose
slowest stages both hit the deadline produced no findings at all.

The deadline now degrades instead of aborting:

*   `DeadlineBudget` (`src/ai/backoff_provider.rs`) returns a typed
    `ActiveTimeExceededError` on expiry. Exactly once, on that first expiry,
    it extends the deadline by a fixed grace window (`FINAL_SYNTHESIS_GRACE`,
    180s): a bounded exception for wrap-up turns, not an extension of the
    review. Every later check fails as before. The error's display text is
    the marker `reviewer.rs` matches on for its kill path, so it must not
    change.
*   `SessionRunner` (`src/ai/session.rs`) catches that error, injects a
    "TIME BUDGET EXHAUSTED" turn with the tools withdrawn, and spends the
    grace window forcing the model to synthesize its verdict from the
    evidence already in the history — the same mechanism the max-turns
    budget uses. A session with no gathered evidence fails immediately, and
    a second deadline error after the degrade fails the session.
*   The grace is shared by every concurrent session of the same budget: the
    LLM gate serialises their synthesis calls, so the window bounds the
    extra work to a handful of final turns. Reviews driven through the
    daemon's stdio bridge see the error as a generic provider failure and
    keep the old hard-stop behaviour; only in-process (local review) runs
    degrade today.