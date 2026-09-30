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

//! A pass-through [`AiProvider`] decorator that caps the number of concurrent
//! `generate_content` calls against a process-wide semaphore.
//!
//! A review fans its analysis stages out concurrently, so a single patch can
//! issue one model call per stage at once, and the worker reviews several
//! patches in parallel on top of that. The daemon adds the bug worker and the
//! bug-filing endpoint on the same subscription. A process-wide gate covers
//! every one of them: any provider wrapped here draws on the same permits, so
//! `[ai] max_concurrent_requests` is a ceiling on the process, not on one
//! caller that remembered to ask.

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::Semaphore;

use crate::ai::trace;
use crate::ai::{AiProvider, AiRequest, AiResponse, CacheStats, ProviderCapabilities};
use serde_json::json;

static LLM_GATE: OnceLock<Semaphore> = OnceLock::new();

/// Sizes the process-wide gate from `[ai] max_concurrent_requests`.
///
/// Called by each front end (the daemon, a local review, the benchmark)
/// before any model call is made. The first call wins: a later one with a
/// different value is ignored — and warned about — as the permits are
/// already out. A process that never calls this gets the setting's default.
pub fn init_llm_gate(permits: usize) {
    let permits = permits.max(1);
    if LLM_GATE.set(Semaphore::new(permits)).is_err() {
        // Either another front end sized the gate first, or a model call
        // already fell back to the default before this ran: the knob is
        // dead either way and the process keeps the earlier ceiling.
        tracing::warn!(
            "max_concurrent_requests = {permits} ignored: the process-wide LLM gate was already sized"
        );
    }
}

/// The process-wide pool of in-flight model-call permits.
pub fn llm_gate() -> &'static Semaphore {
    LLM_GATE.get_or_init(|| {
        Semaphore::new(crate::settings::default_max_concurrent_requests().max(1))
    })
}

/// Limits concurrent model calls to the permits of the process-wide gate. All
/// other behaviour is delegated unchanged to the inner provider.
pub struct ConcurrencyLimitedProvider {
    inner: Arc<dyn AiProvider>,
}

impl ConcurrencyLimitedProvider {
    pub fn new(inner: Arc<dyn AiProvider>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl AiProvider for ConcurrencyLimitedProvider {
    async fn generate_content(&self, request: AiRequest) -> Result<AiResponse> {
        let started = std::time::Instant::now();
        let _permit = llm_gate()
            .acquire()
            .await
            .map_err(|e| anyhow::anyhow!("concurrency semaphore closed: {e}"))?;
        let waited = started.elapsed();
        if waited > std::time::Duration::from_millis(10) {
            trace::event(
                "gate_wait",
                json!({ "waited_ms": waited.as_millis() as u64 }),
            );
        }
        self.inner.generate_content(request).await
    }

    fn get_capabilities(&self) -> ProviderCapabilities {
        self.inner.get_capabilities()
    }

    fn cache_stats(&self) -> Option<CacheStats> {
        self.inner.cache_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_gate_is_never_sized_shut() {
        // A closed gate (zero permits) would deadlock every model call. The
        // exact count is not asserted: the gate is process-wide, so another
        // test may already have sized it, and only the clamp is this test's
        // to guarantee.
        init_llm_gate(0);
        assert!(llm_gate().try_acquire().is_ok());
    }
}
