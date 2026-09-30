// Copyright 2026 The Sashiko Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A pass-through [`AiProvider`] decorator that retries with backoff on
//! rate-limit and transient (e.g. 503 "overloaded") errors.
//!
//! The daemon classifies provider errors and backs off before retrying them.
//! A worker running reviews in-process calls the provider directly, so its only
//! recourse is the stage loop retrying immediately and blindly. This decorator
//! gives that path the same behaviour, reusing [`QuotaManager`] and the typed
//! [`AiErrorClass`] classification the providers already produce:
//!
//! - `RateLimit` (429 / quota): account-wide, so it is reported to a shared
//!   [`QuotaManager`] and every concurrent request waits out the window.
//! - `Transient` (503 overloaded, 500/502/504, 529): a momentary server-side
//!   failure, so only this call backs off, exponentially and with jitter so
//!   concurrent stages do not resynchronise onto the same retry instant.
//! - `Fatal`: propagated immediately.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use tokio::time::sleep;
use tracing::warn;

use crate::ai::quota::QuotaManager;
use crate::ai::trace;
use crate::ai::{
    AiErrorClass, AiProvider, AiRequest, AiResponse, CacheStats, ProviderCapabilities,
    classify_ai_error, get_log_prefix,
};
use serde_json::json;

/// Maximum number of attempts (initial try plus retries) for a single call
/// before the last error is propagated. Bounds wall-clock so a sustained
/// outage cannot hang a review indefinitely; transient blips resolve well
/// within this.
const MAX_ATTEMPTS: u32 = 6;

/// Lets a caller account for the time spent waiting on the shared quota gate
/// and stop the retry loop early.
///
/// The daemon bounds a review by an activity deadline rather than by an attempt
/// count, and does not want a rate-limit wait to consume that budget, so it
/// credits the wait back and fails once the deadline passes.
pub trait RetryBudget: Send + Sync {
    /// Time spent blocked on the shared quota gate before an attempt.
    fn credit_wait(&self, slept: Duration);
    /// Return an error to abandon further retries.
    fn check(&self) -> Result<()>;
}

/// A [`RetryBudget`] backed by a wall-clock deadline shared with the caller.
///
/// Time spent parked on the quota gate is credited back, so waiting out a rate
/// limit does not consume the caller's budget.
pub struct DeadlineBudget {
    deadline: Arc<std::sync::Mutex<tokio::time::Instant>>,
    last_credit_end: std::sync::Mutex<Option<tokio::time::Instant>>,
    /// Whether the one-shot wrap-up grace was already granted.
    grace_granted: AtomicBool,
}

/// Window a budget grants once after expiring, for final synthesis turns.
///
/// Sized for one synthesis turn per straggler session (a full prompt re-send
/// plus a short generation) plus one transient backoff, through the LLM
/// gate's serialisation. It is a bounded exception to the deadline, not an
/// extension of it: once spent, every further check fails.
const FINAL_SYNTHESIS_GRACE: Duration = Duration::from_secs(180);

impl DeadlineBudget {
    pub fn new(deadline: Arc<std::sync::Mutex<tokio::time::Instant>>) -> Self {
        Self {
            deadline,
            last_credit_end: std::sync::Mutex::new(None),
            grace_granted: AtomicBool::new(false),
        }
    }
}

impl RetryBudget for DeadlineBudget {
    fn credit_wait(&self, slept: Duration) {
        if slept.is_zero() {
            return;
        }
        let now = tokio::time::Instant::now();
        let sleep_start = now.checked_sub(slept).unwrap_or(now);
        let mut last_end = self.last_credit_end.lock().unwrap();
        let effective_start = match *last_end {
            Some(end) if end > sleep_start => end,
            _ => sleep_start,
        };
        if now > effective_start {
            let uncredited = now - effective_start;
            // Filter out sub-millisecond thread scheduling jitter between
            // concurrent tasks waking from the same rate-limit window.
            if uncredited >= Duration::from_millis(1) {
                let mut d = self.deadline.lock().unwrap();
                *d += uncredited;
                *last_end = Some(now);
            }
        }
    }

    fn check(&self) -> Result<()> {
        let now = tokio::time::Instant::now();
        if now <= { *self.deadline.lock().unwrap() } {
            return Ok(());
        }
        // The first expiry grants every caller one short wrap-up window:
        // sessions spend it forcing a final synthesis turn out of the
        // evidence already gathered instead of dropping it. The error still
        // fires, so callers learn the budget is over and can degrade.
        if !self.grace_granted.swap(true, Ordering::SeqCst) {
            let mut deadline = self.deadline.lock().unwrap();
            *deadline += FINAL_SYNTHESIS_GRACE;
            warn!(
                "review active-time budget exhausted; granting {}s of grace for final synthesis turns",
                FINAL_SYNTHESIS_GRACE.as_secs()
            );
        }
        Err(ActiveTimeExceededError.into())
    }
}

/// The review's active-time budget ran out (see [`DeadlineBudget`]).
///
/// Typed so the session runner can degrade to a final synthesis turn
/// instead of dropping the evidence a stage already gathered. The display
/// text is also the marker reviewer.rs matches on for its kill path, so it
/// must not change.
#[derive(Debug, thiserror::Error)]
#[error("Review tool timed out (active time exceeded)")]
pub struct ActiveTimeExceededError;

/// Adds rate-limit and transient retry with backoff around an inner provider.
pub struct BackoffProvider {
    inner: Arc<dyn AiProvider>,
    /// Shared across the run, so one rate-limit response backs every
    /// concurrent request off together.
    quota: Arc<QuotaManager>,
    /// Base unit of the exponential transient backoff. Tests use a tiny value
    /// so they run in real time.
    base_delay: Duration,
    /// Attempt ceiling, or None to retry until `budget` says to stop.
    max_attempts: Option<u32>,
    budget: Option<Arc<dyn RetryBudget>>,
}

impl BackoffProvider {
    /// With a `budget`, retries until it reports the caller's deadline has
    /// passed. Without one, falls back to a fixed attempt ceiling, which bounds
    /// wall-clock for callers that have no deadline of their own.
    pub fn new(
        inner: Arc<dyn AiProvider>,
        quota: Arc<QuotaManager>,
        budget: Option<Arc<dyn RetryBudget>>,
    ) -> Self {
        Self {
            inner,
            quota,
            base_delay: Duration::from_secs(1),
            max_attempts: budget.is_none().then_some(MAX_ATTEMPTS),
            budget,
        }
    }

    fn class_str(class: &AiErrorClass) -> &'static str {
        match class {
            AiErrorClass::Fatal => "fatal",
            AiErrorClass::RateLimit { .. } => "rate_limit",
            AiErrorClass::Transient { .. } => "transient",
        }
    }
}

#[async_trait]
impl AiProvider for BackoffProvider {
    async fn generate_content(&self, request: AiRequest) -> Result<AiResponse> {
        // The tag rides a task local so the gate and the HTTP client below
        // this loop can name their patch and stage in trace events.
        let tag = request.context_tag.clone();
        trace::scope_ctx(tag, self.generate_with_retries(request)).await
    }

    async fn open_session(
        &self,
        request: AiRequest,
    ) -> Result<Option<Box<dyn crate::ai::ProviderSession>>> {
        let tag = request.context_tag.clone();
        let session_tag = tag.clone();
        trace::scope_ctx(tag, async move {
            Ok(self.inner.open_session(request).await?.map(|inner| {
                Box::new(BackoffSession {
                    inner,
                    quota: self.quota.clone(),
                    base_delay: self.base_delay,
                    max_attempts: self.max_attempts,
                    budget: self.budget.clone(),
                    context_tag: session_tag,
                }) as Box<dyn crate::ai::ProviderSession>
            }))
        })
        .await
    }

    fn get_capabilities(&self) -> ProviderCapabilities {
        self.inner.get_capabilities()
    }

    fn cache_stats(&self) -> Option<CacheStats> {
        self.inner.cache_stats()
    }
}

/// A provider-native session wrapped in the same retry policy as a
/// stateless request: quota waits, budget checks and transient backoff
/// apply per send, so iterating a session cannot bypass them.
struct BackoffSession {
    inner: Box<dyn crate::ai::ProviderSession>,
    quota: Arc<QuotaManager>,
    base_delay: Duration,
    max_attempts: Option<u32>,
    budget: Option<Arc<dyn RetryBudget>>,
    context_tag: Option<String>,
}

#[async_trait]
impl crate::ai::ProviderSession for BackoffSession {
    async fn send(&mut self, messages: Vec<crate::ai::AiMessage>) -> Result<AiResponse> {
        let tag = self.context_tag.clone();
        trace::scope_ctx(tag, async {
            run_with_retries(
                self.quota.clone(),
                self.base_delay,
                self.max_attempts,
                self.budget.clone(),
                Box::new(SendAttempt {
                    inner: &mut self.inner,
                    messages,
                }),
            )
            .await
        })
        .await
    }

    async fn close(self: Box<Self>) -> Result<()> {
        self.inner.close().await
    }
}

impl BackoffProvider {
    async fn generate_with_retries(&self, request: AiRequest) -> Result<AiResponse> {
        run_with_retries(
            self.quota.clone(),
            self.base_delay,
            self.max_attempts,
            self.budget.clone(),
            Box::new(RequestAttempt {
                inner: self.inner.clone(),
                request,
            }),
        )
        .await
    }
}

/// One retryable model attempt: a stateless request or a native-session
/// send. A trait (rather than a closure) because the retry loop holds the
/// attempt across awaits and must stay future-type agnostic.
#[async_trait]
trait Attempt {
    async fn attempt(&mut self) -> Result<AiResponse>;
}

/// A stateless request replayed verbatim on every retry.
struct RequestAttempt {
    inner: Arc<dyn AiProvider>,
    request: AiRequest,
}

#[async_trait]
impl Attempt for RequestAttempt {
    async fn attempt(&mut self) -> Result<AiResponse> {
        self.inner.generate_content(self.request.clone()).await
    }
}

/// A native-session delta: replayed verbatim while it keeps failing.
struct SendAttempt<'a> {
    inner: &'a mut Box<dyn crate::ai::ProviderSession>,
    messages: Vec<crate::ai::AiMessage>,
}

#[async_trait]
impl Attempt for SendAttempt<'_> {
    async fn attempt(&mut self) -> Result<AiResponse> {
        self.inner.send(self.messages.clone()).await
    }
}

/// The retry loop shared by stateless requests and native-session sends:
/// honour the quota window, check the caller's budget, run one attempt,
/// and back off or propagate per the error class. Routing both transports
/// through one loop keeps their retry policy from drifting apart.
async fn run_with_retries<'a>(
    quota: Arc<QuotaManager>,
    base_delay: Duration,
    max_attempts: Option<u32>,
    budget: Option<Arc<dyn RetryBudget>>,
    mut attempt: Box<dyn Attempt + Send + 'a>,
) -> Result<AiResponse> {
    let mut attempt_no: u32 = 0;
    let mut transient_streak: i32 = 0;
    loop {
            // Honour any active global rate-limit window before trying. The
            // wait is reported so a caller can keep it off its own deadline.
            let slept = quota.wait_for_access().await;
            if slept > Duration::ZERO {
                trace::event(
                    "quota_wait",
                    json!({ "waited_ms": slept.as_millis() as u64 }),
                );
            }
            if let Some(budget) = &budget {
                budget.credit_wait(slept);
                budget.check()?;
            }

            let attempt_started = std::time::Instant::now();
            match attempt.attempt().await {
                Ok(response) => {
                    quota.report_success().await;
                    trace::event(
                        "llm_attempt",
                        json!({
                            "attempt": attempt_no + 1,
                            "duration_ms": attempt_started.elapsed().as_millis() as u64,
                            "outcome": "ok",
                            "tokens_in": response.usage.as_ref().map_or(0, |u| u.prompt_tokens),
                            "tokens_out": response.usage.as_ref().map_or(0, |u| u.completion_tokens),
                        }),
                    );
                    return Ok(response);
                }
                Err(e) => {
                    attempt_no += 1;
                    let class = classify_ai_error(&e);
                    trace::event(
                        "llm_attempt",
                        json!({
                            "attempt": attempt_no,
                            "duration_ms": attempt_started.elapsed().as_millis() as u64,
                            "outcome": BackoffProvider::class_str(&class),
                            "error": e.to_string(),
                        }),
                    );
                    if let Some(max) = max_attempts
                        && attempt_no >= max
                    {
                        return Err(e);
                    }
                    match class {
                        AiErrorClass::RateLimit { retry_after } => {
                            // Account-wide: block every concurrent request
                            // until it clears. The next iteration's
                            // wait_for_access() performs the sleep.
                            trace::event(
                                "backoff",
                                json!({
                                    "reason": "rate_limit",
                                    "retry_after_ms": retry_after.as_millis() as u64,
                                }),
                            );
                            quota.report_quota_error(retry_after).await;
                        }
                        AiErrorClass::Transient { retry_after } => {
                            // Server-side blip. Exponential backoff with
                            // jitter, floored by any server-suggested delay.
                            // Per-call, not global.
                            transient_streak += 1;
                            let mult = 2.0_f64.powi(transient_streak - 1).min(60.0);
                            let backoff = base_delay.mul_f64(mult).max(retry_after);
                            let jittered = backoff + backoff.mul_f64(0.25 * fastrand::f64());
                            let retry_target = match max_attempts {
                                Some(max) => format!("{attempt_no}/{max}"),
                                None => format!("{attempt_no}"),
                            };
                            warn!(
                                "{}Transient AI error (streak {}); backing off {:.1}s then retry {}: {}",
                                get_log_prefix(),
                                transient_streak,
                                jittered.as_secs_f64(),
                                retry_target,
                                e
                            );
                            trace::event(
                                "backoff",
                                json!({
                                    "reason": "transient",
                                    "streak": transient_streak,
                                    "retry_after_ms": retry_after.as_millis() as u64,
                                    "sleep_ms": jittered.as_millis() as u64,
                                }),
                            );
                            sleep(jittered).await;
                        }
                        AiErrorClass::Fatal => return Err(e),
                    }
                }
            }
        }
    }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::gemini::GeminiError;
    use crate::ai::{AiMessage, AiRole, ProviderSession};
    use std::sync::atomic::{AtomicU32, Ordering};

    enum Behaviour {
        Transient,
        RateLimit,
        Fatal,
    }

    struct MockProvider {
        calls: AtomicU32,
        fail_times: u32,
        behaviour: Behaviour,
        rate_limit_after: Duration,
    }

    #[async_trait]
    impl AiProvider for MockProvider {
        async fn generate_content(&self, _request: AiRequest) -> Result<AiResponse> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_times {
                let err = match self.behaviour {
                    Behaviour::Transient => {
                        GeminiError::TransientError(Duration::from_secs(0), "503 overloaded".into())
                    }
                    Behaviour::RateLimit => GeminiError::QuotaExceeded(self.rate_limit_after),
                    Behaviour::Fatal => GeminiError::PermissionDenied("nope".into()),
                };
                return Err(err.into());
            }
            Ok(AiResponse {
                content: Some("ok".into()),
                thought: None,
                thought_signature: None,
                tool_calls: None,
                usage: None,
                truncated: false,
            })
        }

        fn get_capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                model_name: "mock".into(),
                context_window_size: 1000,
            }
        }
    }

    fn mock(
        behaviour: Behaviour,
        fail_times: u32,
        rate_limit_after: Duration,
    ) -> Arc<MockProvider> {
        Arc::new(MockProvider {
            calls: AtomicU32::new(0),
            fail_times,
            behaviour,
            rate_limit_after,
        })
    }

    /// A BackoffProvider with a tiny base delay so transient backoff runs in
    /// real time without needing the tokio virtual clock.
    fn fast(inner: Arc<dyn AiProvider>) -> BackoffProvider {
        BackoffProvider {
            inner,
            quota: Arc::new(QuotaManager::new()),
            base_delay: Duration::from_millis(1),
            max_attempts: Some(MAX_ATTEMPTS),
            budget: None,
        }
    }

    fn dummy_request() -> AiRequest {
        AiRequest {
            system: None,
            messages: vec![AiMessage {
                role: AiRole::User,
                content: Some("hi".into()),
                thought: None,
                thought_signature: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: None,
            temperature: None,
            response_format: None,
            context_tag: None,
        }
    }

    #[tokio::test]
    async fn transient_backs_off_then_succeeds() {
        let m = mock(Behaviour::Transient, 3, Duration::ZERO);
        let resp = fast(m.clone())
            .generate_content(dummy_request())
            .await
            .unwrap();
        assert_eq!(resp.content.as_deref(), Some("ok"));
        assert_eq!(m.calls.load(Ordering::SeqCst), 4); // 3 transient failures + success
    }

    #[tokio::test]
    async fn transient_gives_up_after_max_attempts() {
        let m = mock(Behaviour::Transient, u32::MAX, Duration::ZERO);
        let err = fast(m.clone()).generate_content(dummy_request()).await;
        assert!(err.is_err());
        assert_eq!(m.calls.load(Ordering::SeqCst), MAX_ATTEMPTS); // bounded
    }

    #[tokio::test]
    async fn fatal_propagates_without_retry() {
        let m = mock(Behaviour::Fatal, u32::MAX, Duration::ZERO);
        let err = fast(m.clone()).generate_content(dummy_request()).await;
        assert!(err.is_err());
        assert_eq!(m.calls.load(Ordering::SeqCst), 1); // no retry on fatal
    }

    #[tokio::test]
    async fn expired_budget_stops_before_calling() {
        struct Expired;
        impl RetryBudget for Expired {
            fn credit_wait(&self, _slept: Duration) {}
            fn check(&self) -> Result<()> {
                Err(anyhow::anyhow!("deadline exceeded"))
            }
        }
        let m = mock(Behaviour::Transient, u32::MAX, Duration::ZERO);
        let provider = BackoffProvider::new(
            m.clone(),
            Arc::new(QuotaManager::new()),
            Some(Arc::new(Expired)),
        );
        assert!(provider.generate_content(dummy_request()).await.is_err());
        // The budget gates the loop, so the inner provider is never reached.
        assert_eq!(m.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn rate_limit_waits_then_succeeds() {
        // QuotaManager uses std::Instant, so use a tiny real delay here.
        let m = mock(Behaviour::RateLimit, 2, Duration::from_millis(5));
        let resp = fast(m.clone())
            .generate_content(dummy_request())
            .await
            .unwrap();
        assert_eq!(resp.content.as_deref(), Some("ok"));
        assert_eq!(m.calls.load(Ordering::SeqCst), 3); // 2 rate-limit failures + success
    }

    #[tokio::test]
    async fn deadline_budget_credit_wait_deduplicates_overlapping_sleeps() {
        let deadline = Arc::new(std::sync::Mutex::new(
            tokio::time::Instant::now() + Duration::from_secs(60),
        ));
        let budget = DeadlineBudget::new(deadline.clone());

        // First sleep of 10s credits approximately 10s.
        budget.credit_wait(Duration::from_secs(10));
        let d1 = *deadline.lock().unwrap();

        // An immediately following call reporting the same 10s sleep should not add another 10s.
        budget.credit_wait(Duration::from_secs(10));
        let d2 = *deadline.lock().unwrap();
        assert_eq!(d1, d2);

        // Zero sleep should not affect deadline.
        budget.credit_wait(Duration::ZERO);
        let d3 = *deadline.lock().unwrap();
        assert_eq!(d2, d3);
    }

    #[tokio::test]
    async fn deadline_budget_grants_one_wrap_up_grace() {
        let deadline = Arc::new(std::sync::Mutex::new(
            tokio::time::Instant::now() - Duration::from_millis(1),
        ));
        let budget = DeadlineBudget::new(deadline.clone());

        // First expiry: the error fires, but the deadline jumps into the
        // future by the grace window so a degraded caller's next check
        // passes.
        assert!(budget.check().is_err());
        assert!(*deadline.lock().unwrap() > tokio::time::Instant::now());
        assert!(budget.check().is_ok());

        // Once the grace is spent, expiry is permanent: no second grant.
        {
            let mut d = deadline.lock().unwrap();
            *d -= FINAL_SYNTHESIS_GRACE + Duration::from_millis(1);
        }
        assert!(budget.check().is_err());
        assert!(budget.check().is_err());
        assert!(
            *deadline.lock().unwrap() <= tokio::time::Instant::now(),
            "no further grace may be granted"
        );
    }

    #[test]
    fn active_time_error_keeps_its_display_text() {
        // reviewer.rs matches the kill path on this exact text.
        assert_eq!(
            ActiveTimeExceededError.to_string(),
            "Review tool timed out (active time exceeded)"
        );
    }

    /// Observable state of a scripted native session, shared with the
    /// consuming handle so a test can inspect it after close().
    struct SessionState {
        calls: AtomicU32,
        closed: AtomicBool,
    }

    struct MockSession {
        state: Arc<SessionState>,
        fail_times: u32,
        fatal: bool,
    }

    #[async_trait]
    impl crate::ai::ProviderSession for MockSession {
        async fn send(&mut self, _messages: Vec<AiMessage>) -> Result<AiResponse> {
            let n = self.state.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_times {
                let err = if self.fatal {
                    GeminiError::PermissionDenied("nope".into())
                } else {
                    GeminiError::TransientError(Duration::from_secs(0), "503".into())
                };
                return Err(err.into());
            }
            Ok(AiResponse {
                content: Some("ok".into()),
                thought: None,
                thought_signature: None,
                tool_calls: None,
                usage: None,
                truncated: false,
            })
        }

        async fn close(self: Box<Self>) -> Result<()> {
            self.state.closed.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    /// A BackoffSession with a tiny base delay, like `fast` above.
    fn fast_session(inner: Box<dyn crate::ai::ProviderSession>) -> BackoffSession {
        BackoffSession {
            inner,
            quota: Arc::new(QuotaManager::new()),
            base_delay: Duration::from_millis(1),
            max_attempts: Some(MAX_ATTEMPTS),
            budget: None,
            context_tag: None,
        }
    }

    #[tokio::test]
    async fn native_session_sends_retry_through_the_shared_loop() {
        let state = Arc::new(SessionState {
            calls: AtomicU32::new(0),
            closed: AtomicBool::new(false),
        });
        let mut session = fast_session(Box::new(MockSession {
            state: state.clone(),
            fail_times: 2,
            fatal: false,
        }));

        let response = session.send(vec![AiMessage {
            role: AiRole::User,
            content: Some("delta".into()),
            thought: None,
            thought_signature: None,
            tool_calls: None,
            tool_call_id: None,
        }]).await.unwrap();

        assert_eq!(response.content.as_deref(), Some("ok"));
        assert_eq!(state.calls.load(Ordering::SeqCst), 3); // 2 failures + success
        Box::new(session).close().await.unwrap();
        assert!(state.closed.load(Ordering::SeqCst), "close must propagate");
    }

    #[tokio::test]
    async fn native_session_fatal_errors_do_not_retry() {
        let state = Arc::new(SessionState {
            calls: AtomicU32::new(0),
            closed: AtomicBool::new(false),
        });
        let mut session = fast_session(Box::new(MockSession {
            state: state.clone(),
            fail_times: 1,
            fatal: true,
        }));

        let error = session
            .send(vec![])
            .await
            .err()
            .expect("fatal error must propagate");
        assert!(error.to_string().contains("nope"));
        assert_eq!(state.calls.load(Ordering::SeqCst), 1);
    }
}
