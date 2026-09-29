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

//! Runtime debug dump: one JSONL file recording every model call, retry,
//! wait, tool call and stage of a run, with timestamps.
//!
//! The question this answers is "where did the wall-clock go": how much time
//! requests spent in flight, waiting on the concurrency gate, backing off, or
//! running tools, and what each turn cost. The file is written line by line
//! and flushed per event, so an interrupted run keeps everything that
//! happened before the interrupt.
//!
//! The sink is process-wide and opt-in: `sashiko review --trace [FILE]`
//! opens it, every layer appends through [`event`], and a run without the
//! flag pays one relaxed atomic load per event. The caller's context tag
//! (which patch, which stage) reaches the provider layers through a task
//! local, the same trick LOG_CONTEXT uses.

use serde_json::{Value, json};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);
static SEQ: AtomicU64 = AtomicU64::new(0);
// Errors disable the sink rather than kill the review over a debug dump.
static SINK: Mutex<Option<std::fs::File>> = Mutex::new(None);

/// Opens the dump file and enables tracing. Events written before this are
/// dropped, which keeps the flag's position in startup harmless.
pub fn init(path: &std::path::Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    *SINK.lock().unwrap() = Some(file);
    ENABLED.store(true, Ordering::SeqCst);
    Ok(())
}

/// Default location for a run that passes --trace with no value: one file
/// per run, timestamped, under the data directory beside the prompt bundles.
pub fn default_run_path() -> PathBuf {
    let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    crate::utils::data_home()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("sashiko")
        .join("trace")
        .join(format!("review-{stamp}.jsonl"))
}

/// Appends one event. `fields` is whatever the call site knows; the
/// timestamp and sequence number are added here so no event can lack them.
pub fn event(kind: &str, fields: Value) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let mut guard = SINK.lock().unwrap();
    let Some(file) = guard.as_mut() else {
        return;
    };
    let ts = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let mut record = json!({ "ts": ts, "seq": seq, "event": kind });
    if let (Value::Object(record), Value::Object(fields)) = (&mut record, fields) {
        record.extend(fields);
    }
    if writeln!(file, "{record}").is_err() || file.flush().is_err() {
        *guard = None;
        ENABLED.store(false, Ordering::SeqCst);
        tracing::warn!("Trace dump disabled: writing it failed");
    }
}

tokio::task_local! {
    static TRACE_CTX: Option<String>;
}

/// Runs `fut` with the given context tag visible to every trace event
/// emitted below it on this task, so provider-layer events can name their
/// patch and stage without threading the tag through the provider stack.
pub async fn scope_ctx<T>(ctx: Option<String>, fut: impl std::future::Future<Output = T>) -> T {
    TRACE_CTX.scope(ctx, fut).await
}

/// The context tag of the current call, or None when no layer scoped one.
pub fn ctx() -> Option<String> {
    TRACE_CTX.try_with(|c| c.clone()).ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_dump_records_events_with_envelope_and_ctx() {
        // The sink is process-wide, so this one test owns every assertion
        // about it: sequencing it against another trace test would race on
        // the shared sequence counter.
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("trace.jsonl");
        init(&path).unwrap();
        event("kept", json!({ "field": 2 }));
        event("kept", Value::Null);

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);
        let mut last_seq = None;
        for line in &lines {
            let parsed: Value = serde_json::from_str(line).unwrap();
            assert_eq!(parsed["event"], "kept");
            assert!(parsed["ts"].as_str().is_some());
            let seq = parsed["seq"].as_u64().unwrap();
            if let Some(last) = last_seq {
                assert!(seq > last, "sequence numbers must advance");
            }
            last_seq = Some(seq);
        }
        // Fields merge into the record; a non-object payload keeps the
        // envelope intact rather than failing the write.
        let first: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["field"], 2);

        // The tag rides a task local down to nested events, and is invisible
        // outside the scope.
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(scope_ctx(Some("[ps:1 p:2 s:goal]".to_string()), async {
            event("probe", json!({ "ctx": ctx() }));
            assert_eq!(ctx().as_deref(), Some("[ps:1 p:2 s:goal]"));
        }));
        assert!(ctx().is_none());
        let probe_line = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .find(|l| l.contains("\"probe\""))
            .map(str::to_string)
            .unwrap();
        let probe: Value = serde_json::from_str(&probe_line).unwrap();
        assert_eq!(probe["ctx"], "[ps:1 p:2 s:goal]");
    }
}
