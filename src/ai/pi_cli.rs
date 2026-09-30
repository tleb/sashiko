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

//! AI provider that shells out to the `pi` CLI, using the local pi
//! installation's providers and credentials (a subscription plan, an
//! already-working streaming setup) instead of sashiko's own HTTP clients.
//!
//! ## Safety
//!
//! The CLI runs with every tool, session, extension, skill and context-file
//! discovery disabled: it reads a prompt from stdin and writes one completion
//! to stdout. It cannot touch the filesystem, run commands or persist
//! anything, so it is a plain completion backend. Sashiko's tool protocol
//! travels inside the prompt text, exactly as the claude CLI provider does
//! it: the model answers with the tool_calls JSON that sashiko parses,
//! executes against its own ToolBox, and feeds back through the flattened
//! conversation.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::time::{Instant, sleep_until, timeout};
use tracing::{debug, warn};

use crate::ai::claude_cli::build_prompt;
use crate::ai::claude_cli::parse_inner_response;
use crate::ai::{
    AiErrorClass, AiProvider, AiRequest, AiResponse, AiUsage, ClassifyAiError, ProviderCapabilities,
};

/// Hard ceiling per CLI call, as a backstop behind IDLE_TIMEOUT_SECS: a
/// model that streams forever still dies, bounded well under the review
/// timeout. A merely slow model that keeps emitting deltas is never killed
/// by this on its own.
pub const CALL_TIMEOUT_SECS: u64 = 1800;

/// Kill a call whose event stream goes silent for this long. Healthy
/// generation emits message_update deltas tens of milliseconds apart, so
/// silence means a wedged stream, not slow thinking.
pub const IDLE_TIMEOUT_SECS: u64 = 90;

/// pi's thinking levels, lowest to highest, as accepted by --thinking.
const THINKING_LADDER: [&str; 7] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

#[derive(Debug, thiserror::Error)]
pub enum PiCliError {
    #[error("Failed to spawn pi CLI: {0}")]
    Spawn(String),
    #[error("pi CLI timed out after {0}s")]
    Timeout(u64),
    #[error("pi CLI stream stalled for {0}s (killed at {1}s elapsed)")]
    Stalled(u64, u64),
    #[error("pi CLI wait error: {0}")]
    Wait(String),
    #[error("pi CLI error: {0}")]
    Cli(String),
    #[error("Failed to parse pi CLI output: {0}")]
    Parse(String),
}

impl ClassifyAiError for PiCliError {
    fn ai_error_class(&self) -> AiErrorClass {
        match self {
            PiCliError::Spawn(_) => AiErrorClass::Fatal,
            PiCliError::Timeout(_) | PiCliError::Stalled(_, _) => AiErrorClass::Transient {
                retry_after: Duration::from_secs(30),
            },
            PiCliError::Wait(_) => AiErrorClass::Transient {
                retry_after: Duration::from_secs(30),
            },
            PiCliError::Cli(_) => AiErrorClass::Fatal,
            PiCliError::Parse(_) => AiErrorClass::Fatal,
        }
    }
}

pub struct PiCliProvider {
    /// Passed as `--model`; empty means pi's own default, which is the point
    /// of this provider: pi resolves the provider, endpoint and credentials.
    pub model: String,
    /// Binary to run, overridable so tests can point at a fake.
    pub binary: String,
    /// Where pi writes its session file, for visualising runs: None uses
    /// pi's own global session store (~/.pi/agent/sessions/), "default" is
    /// a synonym, anything else is a directory. See PiCliSettings.
    /// Sessions always persist: the native-session support builds on them.
    pub session_dir: Option<String>,
    /// Thinking level passed as `--thinking`; None keeps pi's own default.
    pub thinking_level: Option<String>,
    /// Hard per-call budget; the backstop behind idle_secs.
    pub timeout_secs: u64,
    /// No-progress limit: killed when no stream delta arrives for this long.
    pub idle_secs: u64,
}

impl PiCliProvider {
    fn command(&self, session_name: Option<&str>, thinking: Option<&str>) -> Command {
        let mut cmd = Command::new(&self.binary);
        cmd.args([
            "-p",
            "--mode",
            "json",
            "--no-tools",
            "--no-extensions",
            "--no-skills",
            "--no-context-files",
            // Neutralise pi's coding-assistant persona: the stage's system
            // prompt already travels inside the prompt text.
            "--system-prompt",
            "",
        ]);
        if let Some(level) = thinking {
            cmd.args(["--thinking", level]);
        }
        match self.session_dir.as_deref() {
            // Unset keeps pi's own global store (~/.pi/agent/sessions/):
            // sessions always persist — the native-session support relies
            // on it, and ephemeral calls left no trail to debug.
            None => {}
            Some("default") => {}
            Some(dir) => {
                cmd.args(["--session-dir", dir]);
            }
        }
        // A name carrying the patch and stage makes the session findable in
        // pi's list.
        if let Some(name) = session_name {
            cmd.args(["--name", name]);
        }
        if !self.model.is_empty() {
            cmd.args(["--model", &self.model]);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        cmd
    }

    /// The thinking level one rung below the call's starting level, or None
    /// at the bottom of the ladder.
    ///
    /// An unset setting starts from "high": that is pi's default for the
    /// reasoning models this provider is used with, and the level whose
    /// calls were observed outgrowing a whole call ceiling. A configured
    /// level that matches no rung degrades to None, keeping the retry a
    /// strict step down rather than a guess.
    fn degraded_thinking_level(&self) -> Option<&'static str> {
        let base = self.thinking_level.as_deref().unwrap_or("high");
        let index = THINKING_LADDER.iter().position(|level| *level == base)?;
        (index > 0).then(|| THINKING_LADDER[index - 1])
    }

    /// Spawns the CLI, feeds the prompt on stdin and reads the run to
    /// completion under the two call budgets. The thinking level rides the
    /// command line; None leaves pi's own default in place.
    async fn run_pi(
        &self,
        prompt: &str,
        session_name: Option<&str>,
        thinking: Option<&str>,
    ) -> Result<(String, String, std::process::ExitStatus), PiCliError> {
        let mut child = spawn_retrying(|| self.command(session_name, thinking))
            .map_err(|e| PiCliError::Spawn(e.to_string()))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(prompt.as_bytes())
                .await
                .map_err(|e| PiCliError::Wait(e.to_string()))?;
            stdin
                .flush()
                .await
                .map_err(|e| PiCliError::Wait(e.to_string()))?;
        }

        self.read_to_completion(child).await
    }

    /// Reads the CLI's stdout event stream until the process exits, under
    /// two budgets: the hard per-call ceiling (timeout_secs) and the
    /// no-progress limit (idle_secs). Only streamed deltas reset the
    /// no-progress timer, so a slow model that keeps thinking is never
    /// killed while a silent stream is. Returns the raw event stream, the
    /// drained stderr and the exit status.
    async fn read_to_completion(
        &self,
        mut child: tokio::process::Child,
    ) -> Result<(String, String, std::process::ExitStatus), PiCliError> {
        // Piped by command(), the only constructor of the child.
        let stdout = child
            .stdout
            .take()
            .expect("pi CLI child always pipes stdout");
        let stderr = child
            .stderr
            .take()
            .expect("pi CLI child always pipes stderr");

        // Drain stderr concurrently: the child blocks once the pipe fills,
        // and its output is only needed for error reporting.
        let stderr_task = tokio::spawn(async move {
            let mut collected = String::new();
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if !line.trim().is_empty() {
                    debug!("[pi-cli stderr] {line}");
                }
                collected.push_str(&line);
                collected.push('\n');
            }
            collected
        });

        let started = Instant::now();
        let deadline = started + Duration::from_secs(self.timeout_secs);
        let mut idle_deadline = started + Duration::from_secs(self.idle_secs);
        let mut lines = BufReader::new(stdout).lines();
        let mut events = String::new();

        let status = loop {
            let total = sleep_until(deadline);
            tokio::pin!(total);
            let idle = sleep_until(idle_deadline);
            tokio::pin!(idle);
            tokio::select! {
                line = lines.next_line() => match line {
                    Ok(Some(line)) => {
                        if is_progress_event(&line) {
                            idle_deadline =
                                Instant::now() + Duration::from_secs(self.idle_secs);
                        }
                        events.push_str(&line);
                        events.push('\n');
                    }
                    // The stream ended; only the exit status is missing.
                    Ok(None) => {
                        let status = self.await_exit(child, deadline).await;
                        break status;
                    }
                    Err(e) => return Err(PiCliError::Wait(e.to_string())),
                },
                _ = &mut idle => {
                    let _ = child.start_kill();
                    return Err(PiCliError::Stalled(
                        self.idle_secs,
                        started.elapsed().as_secs(),
                    ));
                }
                _ = &mut total => {
                    let _ = child.start_kill();
                    return Err(PiCliError::Timeout(self.timeout_secs));
                }
            }
        };

        // The child has exited, so its pipes close; this cannot hang long.
        let stderr = match timeout(Duration::from_secs(5), stderr_task).await {
            Ok(Ok(stderr)) => stderr,
            _ => String::new(),
        };
        Ok((events, stderr, status?))
    }

    /// Waits for the child's exit under the remaining call budget: a child
    /// that closed its stream but never exits must not outlive the ceiling.
    async fn await_exit(
        &self,
        mut child: tokio::process::Child,
        deadline: Instant,
    ) -> Result<std::process::ExitStatus, PiCliError> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match timeout(remaining, child.wait()).await {
            Ok(status) => status.map_err(|e| PiCliError::Wait(e.to_string())),
            Err(_) => {
                let _ = child.start_kill();
                Err(PiCliError::Timeout(self.timeout_secs))
            }
        }
    }
}

#[async_trait]
impl AiProvider for PiCliProvider {
    async fn generate_content(&self, request: AiRequest) -> Result<AiResponse> {
        let prompt = build_prompt(&request);
        debug!("pi-cli prompt length: {} chars", prompt.len());
        // The name joins the patch and stage into pi's session list, so a
        // visualised run can be matched to the review that produced it.
        let session_name = request
            .context_tag
            .as_ref()
            .map(|tag| format!("sashiko {tag}"));

        let (stdout, stderr, status) = match self
            .run_pi(&prompt, session_name.as_deref(), self.thinking_level.as_deref())
            .await
        {
            Ok(run) => run,
            Err(PiCliError::Timeout(secs)) => {
                // A call that streamed for a whole ceiling was reasoning, not
                // wedged: retry once one rung down so the model spends its
                // second ceiling answering instead of thinking.
                let Some(lower) = self.degraded_thinking_level() else {
                    return Err(PiCliError::Timeout(secs).into());
                };
                warn!(
                    "pi CLI hit the {secs}s call ceiling; retrying once with --thinking {lower}"
                );
                self.run_pi(&prompt, session_name.as_deref(), Some(lower))
                    .await?
            }
            Err(e) => return Err(e.into()),
        };

        if !status.success() {
            return Err(
                PiCliError::Cli(format!("exited with {}: {}", status, stderr.trim())).into(),
            );
        }

        let (text, usage, truncated) = parse_events(&stdout)?;
        let mut response = parse_inner_response(&text, usage)?;
        response.truncated = truncated;
        Ok(response)
    }

    fn get_capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            model_name: self.model.clone(),
            context_window_size: 200_000,
        }
    }
}

/// Spawns the command, retrying ETXTBSY a few times: executing a binary
/// microseconds after it was written - a package manager swapping the CLI,
/// a test fixture, a dev-loop rebuild - fails with Text file busy while the
/// inode still reads as open for writing. The window is momentary.
fn spawn_retrying(build: impl Fn() -> Command) -> std::io::Result<tokio::process::Child> {
    let mut last = None;
    for _ in 0..5 {
        match build().spawn() {
            Ok(child) => return Ok(child),
            Err(e) if e.kind() == std::io::ErrorKind::ExecutableFileBusy => {
                last = Some(e);
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => return Err(e),
        }
    }
    Err(last.expect("retry loop ran at least once"))
}

/// Whether an event line proves the model is still generating: pi's
/// message_update records exist only for streamed deltas (thinking, text,
/// tool arguments) and message_end completes a message. Spawn-time events
/// such as agent_start carry no such proof.
fn is_progress_event(line: &str) -> bool {
    #[derive(serde::Deserialize)]
    struct Event {
        #[serde(rename = "type")]
        kind: String,
    }
    match serde_json::from_str::<Event>(line) {
        Ok(event) => matches!(event.kind.as_str(), "message_update" | "message_end"),
        Err(_) => false,
    }
}

/// Extracts the last assistant message from a `pi --mode json` event stream:
/// its text blocks are the completion, its usage block the accounting, and
/// its stopReason says whether the output cap was hit.
fn parse_events(raw: &str) -> Result<(String, Option<AiUsage>, bool)> {
    let mut text: Option<String> = None;
    let mut usage = None;
    let mut truncated = false;
    let mut last_error = None;
    for line in raw.lines() {
        let event: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            // A non-JSON line is not one of pi's events; the run that produced
            // it answered the protocol badly enough to fail below.
            Err(e) => {
                last_error = Some(format!("line rejected as JSON: {e}"));
                continue;
            }
        };
        if event["type"] != "message_end" || event["message"]["role"] != "assistant" {
            continue;
        }
        let message = &event["message"];
        let blocks = message["content"].as_array();
        text = Some(
            blocks
                .map(|blocks| {
                    blocks
                        .iter()
                        .filter(|b| b["type"] == "text")
                        .filter_map(|b| b["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default(),
        );
        truncated = message["stopReason"] == "length";
        let u = &message["usage"];
        if u.is_object() {
            // pi reports the uncached input and the cache hits separately;
            // prompt_tokens covers both, cached_tokens is the breakdown.
            let input = u["input"].as_u64().unwrap_or(0) as usize;
            let cache_read = u["cacheRead"].as_u64().unwrap_or(0) as usize;
            let cache_write = u["cacheWrite"].as_u64().unwrap_or(0) as usize;
            let output = u["output"].as_u64().unwrap_or(0) as usize;
            usage = Some(AiUsage {
                prompt_tokens: input + cache_read + cache_write,
                completion_tokens: output,
                total_tokens: input + cache_read + cache_write + output,
                cached_tokens: (cache_read > 0).then_some(cache_read),
            });
        }
    }
    match text {
        Some(text) => Ok((text, usage, truncated)),
        None => Err(PiCliError::Parse(format!(
            "no assistant message in pi output ({}): {}",
            last_error.unwrap_or_else(|| "no parsable lines".to_string()),
            crate::utils::utf8_prefix(raw, 200)
        ))
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::classify_ai_error;
    use crate::ai::{AiMessage, AiRole};
    use serde_json::json;

    fn dummy_request() -> AiRequest {
        AiRequest {
            system: None,
            messages: vec![AiMessage {
                role: AiRole::User,
                content: Some("hi".to_string()),
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

    /// Writes an executable fake pi running the given shell script body,
    /// via write-to-temp plus rename: executing a file that was just written
    /// can fail with ETXTBSY while the inode still has a writer attached.
    fn script_pi(dir: &std::path::Path, body: &str) -> std::path::PathBuf {
        let path = dir.join("pi");
        let staging = dir.join("pi.tmp");
        std::fs::write(&staging, format!("#!/bin/sh\n{body}")).unwrap();
        std::fs::rename(&staging, &path).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    /// A minimal streamed thinking delta, the event that proves a live
    /// model during a call.
    fn thinking_delta() -> String {
        r#"{"type":"message_update","assistantMessageEvent":{"type":"thinking_delta","contentIndex":0,"delta":"x"}}"#.to_string()
    }

    fn fake_pi(dir: &std::path::Path, stdout: &str, exit: Option<&str>) -> std::path::PathBuf {
        std::fs::write(dir.join("events.jsonl"), stdout).unwrap();
        // The events land in a file the script cats, because pushing JSON
        // through shell quoting gets mangled differently by every echo
        // implementation.
        let record_args = r#"echo "$@" > "$(dirname "$0")/args.txt""#;
        let body = match exit {
            Some(err) => format!("{record_args}\necho {err:?} >&2\nexit 1\n"),
            None => format!("{record_args}\ncat \"$(dirname \"$0\")/events.jsonl\"\n"),
        };
        script_pi(dir, &body)
    }

    fn assistant_end(text: &str, usage: &str, stop: &str) -> String {
        format!(
            concat!(
                r#"{{"type":"agent_start"}}"#,
                '\n',
                r#"{{"type":"message_end","message":{{"role":"assistant","content":[{{"type":"thinking","thinking":"muse"}},{{"type":"text","text":"{text}"}}],"usage":{usage},"stopReason":"{stop}"}}}}"#,
                '\n',
                r#"{{"type":"agent_end","messages":[]}}"#,
                '\n',
            ),
            text = text,
            usage = usage,
            stop = stop
        )
    }

    fn provider(binary: &std::path::Path) -> PiCliProvider {
        PiCliProvider {
            model: String::new(),
            binary: binary.to_string_lossy().into_owned(),
            session_dir: None,
            thinking_level: None,
            timeout_secs: CALL_TIMEOUT_SECS,
            idle_secs: IDLE_TIMEOUT_SECS,
        }
    }

    fn tagged_request() -> AiRequest {
        let mut request = dummy_request();
        request.context_tag = Some("[ps:0 p:1 s:4]".to_string());
        request
    }

    #[tokio::test]
    async fn test_a_completion_comes_back_with_usage() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        let events = assistant_end(
            r#"{\"content\":\"looks fine\"}"#,
            r#"{"input":10,"output":2,"cacheRead":5,"cacheWrite":0,"totalTokens":17}"#,
            "stop",
        );
        let pi = fake_pi(temp.path(), &events, None);

        let response = provider(&pi).generate_content(dummy_request()).await?;

        assert_eq!(response.content.as_deref(), Some("looks fine"));
        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 15);
        assert_eq!(usage.cached_tokens, Some(5));
        assert_eq!(usage.completion_tokens, 2);
        assert!(!response.truncated);
        Ok(())
    }

    #[tokio::test]
    async fn test_the_tool_protocol_and_safety_flags_reach_the_cli() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        let events = assistant_end(
            r#"{\"tool_calls\":[{\"id\":\"c1\",\"function_name\":\"git_log\",\"arguments\":{\"n\":1}}]}"#,
            r#"{"input":1,"output":1,"totalTokens":2}"#,
            "stop",
        );
        let pi = fake_pi(temp.path(), &events, None);

        let response = provider(&pi).generate_content(tagged_request()).await?;

        let calls = response.tool_calls.unwrap();
        assert_eq!(calls[0].function_name, "git_log");
        // The prompt reaches the CLI on stdin and the safety flags on the
        // command line; both are load-bearing, so check the recorded args.
        let args = std::fs::read_to_string(temp.path().join("args.txt")).unwrap();
        for flag in [
            "--no-tools",
            "--no-extensions",
            "--no-skills",
            "--no-context-files",
            "--mode json",
        ] {
            assert!(args.contains(flag), "missing {flag} in: {args}");
        }
        assert!(args.contains("--name"), "sessions are always named now");
        assert!(!args.contains("--model"), "empty model must stay unset");
        // Unset session_dir means pi's own global store: no --no-session
        // escape hatch, no explicit --session-dir either.
        assert!(!args.contains("--no-session"));
        assert!(!args.contains("--session-dir"));
        Ok(())
    }

    #[tokio::test]
    async fn test_a_session_dir_persists_named_sessions() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        let sessions = temp.path().join("sessions");
        let events = assistant_end(
            r#"{\"content\":\"ok\"}"#,
            r#"{"input":1,"output":1,"totalTokens":2}"#,
            "stop",
        );
        let pi = fake_pi(temp.path(), &events, None);
        let provider = PiCliProvider {
            session_dir: Some(sessions.to_string_lossy().into_owned()),
            ..provider(&pi)
        };

        provider.generate_content(tagged_request()).await?;

        let args = std::fs::read_to_string(temp.path().join("args.txt")).unwrap();
        assert!(args.contains(&format!("--session-dir {}", sessions.display())));
        assert!(args.contains("--name sashiko [ps:0 p:1 s:4]"));
        assert!(
            !args.contains("--no-session"),
            "a dir means sessions are written"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_the_default_session_store_is_pis_own() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        let events = assistant_end(
            r#"{\"content\":\"ok\"}"#,
            r#"{"input":1,"output":1,"totalTokens":2}"#,
            "stop",
        );
        let pi = fake_pi(temp.path(), &events, None);
        let provider = PiCliProvider {
            session_dir: Some("default".to_string()),
            ..provider(&pi)
        };

        provider.generate_content(tagged_request()).await?;

        // Neither an explicit directory nor the ephemeral flag: pi's own
        // store (~/.pi/agent/sessions/) is the default target.
        let args = std::fs::read_to_string(temp.path().join("args.txt")).unwrap();
        assert!(!args.contains("--no-session"));
        assert!(!args.contains("--session-dir"));
        assert!(args.contains("--name"));
        Ok(())
    }

    #[tokio::test]
    async fn test_a_length_stop_reason_marks_the_response_truncated() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        let events = assistant_end(
            "half a sentence",
            r#"{"input":1,"output":1,"totalTokens":2}"#,
            "length",
        );
        let pi = fake_pi(temp.path(), &events, None);

        let response = provider(&pi).generate_content(dummy_request()).await?;
        assert!(response.truncated);
        Ok(())
    }

    #[tokio::test]
    async fn test_a_failing_cli_is_a_fatal_error() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        let pi = fake_pi(temp.path(), "", Some("boom"));
        let error = provider(&pi)
            .generate_content(dummy_request())
            .await
            .unwrap_err();
        assert!(matches!(classify_ai_error(&error), AiErrorClass::Fatal));
        Ok(())
    }

    #[tokio::test]
    async fn test_a_stream_without_an_assistant_message_fails_to_parse() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        let pi = fake_pi(temp.path(), r#"{"type":"agent_end","messages":[]}"#, None);
        let error = provider(&pi)
            .generate_content(dummy_request())
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("no assistant message"),
            "wrong error: {error:#}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_a_silent_cli_is_stall_killed_as_transient() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        let pi = script_pi(temp.path(), "sleep 30\n");
        let provider = PiCliProvider {
            model: String::new(),
            binary: pi.to_string_lossy().into_owned(),
            session_dir: None,
            thinking_level: None,
            timeout_secs: 60,
            idle_secs: 1,
        };
        let error = provider
            .generate_content(dummy_request())
            .await
            .unwrap_err();
        assert!(
            matches!(
                error.downcast_ref::<PiCliError>(),
                Some(PiCliError::Stalled(..))
            ),
            "a silent CLI must stall-kill, got: {error:#}"
        );
        assert!(matches!(
            classify_ai_error(&error),
            AiErrorClass::Transient { .. }
        ));
        Ok(())
    }

    #[tokio::test]
    async fn test_a_stream_that_dries_up_mid_call_is_stall_killed() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        let delta = thinking_delta();
        let pi = script_pi(
            temp.path(),
            &format!("printf '%s\\n' '{delta}'\nsleep 30\n"),
        );
        let provider = PiCliProvider {
            idle_secs: 1,
            timeout_secs: 60,
            ..provider(&pi)
        };
        let error = provider
            .generate_content(dummy_request())
            .await
            .unwrap_err();
        assert!(
            matches!(
                error.downcast_ref::<PiCliError>(),
                Some(PiCliError::Stalled(..))
            ),
            "a dried-up stream must stall-kill, got: {error:#}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_a_slow_but_streaming_call_is_not_killed() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        let delta = thinking_delta();
        let mut body = String::new();
        for _ in 0..3 {
            body.push_str(&format!("printf '%s\\n' '{delta}'\nsleep 0.5\n"));
        }
        body.push_str(&format!(
            "printf '%s\\n' '{}'\n",
            r#"{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"alive"}],"stopReason":"stop"}}"#
        ));
        let pi = script_pi(temp.path(), &body);
        let provider = PiCliProvider {
            idle_secs: 2,
            timeout_secs: 60,
            ..provider(&pi)
        };

        // 500ms gaps between deltas are far below the 2s stall limit, so the
        // call must run to completion instead of being killed as stalled.
        let response = provider.generate_content(dummy_request()).await?;
        assert_eq!(response.content.as_deref(), Some("alive"));
        Ok(())
    }

    #[tokio::test]
    async fn test_an_endless_stream_hits_the_hard_cap_as_transient() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        let delta = thinking_delta();
        let pi = script_pi(
            temp.path(),
            &format!("while :; do printf '%s\\n' '{delta}'\nsleep 0.2\ndone\n"),
        );
        let provider = PiCliProvider {
            timeout_secs: 1,
            idle_secs: 60,
            ..provider(&pi)
        };
        let error = provider
            .generate_content(dummy_request())
            .await
            .unwrap_err();
        assert!(
            matches!(
                error.downcast_ref::<PiCliError>(),
                Some(PiCliError::Timeout(_))
            ),
            "an endless stream must hit the ceiling, got: {error:#}"
        );
        assert!(matches!(
            classify_ai_error(&error),
            AiErrorClass::Transient { .. }
        ));
        Ok(())
    }

    /// A fake pi that streams a live delta forever on its first invocation
    /// (hitting the hard ceiling) and answers from a file on every later one,
    /// recording its command line per invocation.
    fn ceiling_then_answer_pi(dir: &std::path::Path) -> std::path::PathBuf {
        let delta = thinking_delta();
        let body = format!(
            concat!(
                "d=$(dirname \"$0\")\n",
                "n=$(cat \"$d/count\" 2>/dev/null || echo 0)\n",
                "echo \"$@\" > \"$d/args$n.txt\"\n",
                "echo $((n + 1)) > \"$d/count\"\n",
                "if [ \"$n\" = \"0\" ]; then\n",
                "  printf '%s\\n' '{delta}'\n",
                "  sleep 30\n",
                "else\n",
                "  cat \"$d/events.jsonl\"\n",
                "fi\n",
            ),
            delta = delta,
        );
        script_pi(dir, &body)
    }

    fn answer_events() -> String {
        assistant_end(
            r#"{\"content\":\"degraded but answered\"}"#,
            r#"{"input":1,"output":1,"totalTokens":2}"#,
            "stop",
        )
    }

    #[tokio::test]
    async fn test_a_ceiling_timeout_retries_once_at_a_lower_rung() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("events.jsonl"), answer_events()).unwrap();
        let pi = ceiling_then_answer_pi(temp.path());
        let provider = PiCliProvider {
            timeout_secs: 1,
            idle_secs: 60,
            ..provider(&pi)
        };

        let response = provider.generate_content(dummy_request()).await.unwrap();

        assert_eq!(response.content.as_deref(), Some("degraded but answered"));
        let first = std::fs::read_to_string(temp.path().join("args0.txt")).unwrap();
        let second = std::fs::read_to_string(temp.path().join("args1.txt")).unwrap();
        assert!(!first.contains("--thinking"), "first call: {first}");
        // "high" is the assumed starting rung for an unset setting.
        assert!(second.contains("--thinking medium"), "retry: {second}");
    }

    #[tokio::test]
    async fn test_a_ceiling_timeout_at_the_bottom_rung_does_not_retry() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("events.jsonl"), answer_events()).unwrap();
        let pi = ceiling_then_answer_pi(temp.path());
        let provider = PiCliProvider {
            timeout_secs: 1,
            idle_secs: 60,
            thinking_level: Some("off".to_string()),
            ..provider(&pi)
        };

        let error = provider
            .generate_content(dummy_request())
            .await
            .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<PiCliError>(),
            Some(PiCliError::Timeout(_))
        ));
        // Only the ceiling-hitting call ran; the answering script never did.
        assert!(temp.path().join("args0.txt").exists());
        assert!(!temp.path().join("args1.txt").exists());
    }

    #[test]
    fn test_only_delta_and_end_events_count_as_progress() {
        assert!(is_progress_event(&thinking_delta()));
        assert!(is_progress_event(
            r#"{"type":"message_end","message":{"role":"assistant"}}"#
        ));
        assert!(!is_progress_event(r#"{"type":"agent_start"}"#));
        assert!(!is_progress_event(
            r#"{"type":"message_start","message":{}}"#
        ));
        assert!(!is_progress_event("node: some warning"));
    }

    #[test]
    fn test_parse_events_skips_non_json_lines() {
        let raw = format!(
            "node: some warning\n{}\nnot json at all\n",
            r#"{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"ok"}],"stopReason":"stop"}}"#
        );
        let (text, usage, truncated) = parse_events(&raw).unwrap();
        assert_eq!(text, "ok");
        assert!(usage.is_none());
        assert!(!truncated);
    }

    #[test]
    fn test_parse_usage_shape_matches_pi_events() {
        // Recorded from a real `pi -p --mode json` run against z.ai.
        let raw = concat!(
            r#"{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"OK"}],"#,
            r#""usage":{"input":45,"output":3,"cacheRead":0,"cacheWrite":0,"totalTokens":368,"cost":{}},"stopReason":"stop"}}"#
        );
        let (text, usage, truncated) = parse_events(raw).unwrap();
        assert_eq!(text, "OK");
        let usage = usage.unwrap();
        assert_eq!(usage.prompt_tokens, 45);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.cached_tokens, None);
        assert!(!truncated);
        let _ = json!({});
    }

    /// Run with `cargo test --lib pi_cli -- --ignored` on a machine with pi
    /// installed and authenticated. Costs one tiny completion.
    #[tokio::test]
    #[ignore = "needs a real pi installation"]
    async fn test_against_a_real_pi_installation() -> Result<()> {
        let provider = PiCliProvider {
            model: String::new(),
            binary: "pi".to_string(),
            session_dir: None,
            thinking_level: None,
            timeout_secs: CALL_TIMEOUT_SECS,
            idle_secs: IDLE_TIMEOUT_SECS,
        };
        let response = provider.generate_content(dummy_request()).await?;
        assert!(response.content.is_some(), "no content: {response:?}");
        Ok(())
    }
}
