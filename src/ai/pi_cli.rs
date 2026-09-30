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
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::debug;

use crate::ai::claude_cli::build_prompt;
use crate::ai::claude_cli::parse_inner_response;
use crate::ai::{
    AiErrorClass, AiProvider, AiRequest, AiResponse, AiUsage, ClassifyAiError, ProviderCapabilities,
};

/// Hard ceiling per CLI call, matching the claude CLI provider: pi retries
/// transport failures itself, so a call that exceeds this is stuck, not slow.
pub const CALL_TIMEOUT_SECS: u64 = 600;

#[derive(Debug, thiserror::Error)]
pub enum PiCliError {
    #[error("Failed to spawn pi CLI: {0}")]
    Spawn(String),
    #[error("pi CLI timed out after {0} minutes")]
    Timeout(u64),
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
            PiCliError::Timeout(_) => AiErrorClass::Transient {
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
    pub timeout_secs: u64,
}

impl PiCliProvider {
    fn command(&self) -> Command {
        let mut cmd = Command::new(&self.binary);
        cmd.args([
            "-p",
            "--mode",
            "json",
            "--no-tools",
            "--no-session",
            "--no-extensions",
            "--no-skills",
            "--no-context-files",
            // Neutralise pi's coding-assistant persona: the stage's system
            // prompt already travels inside the prompt text.
            "--system-prompt",
            "",
        ]);
        if !self.model.is_empty() {
            cmd.args(["--model", &self.model]);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        cmd
    }
}

#[async_trait]
impl AiProvider for PiCliProvider {
    async fn generate_content(&self, request: AiRequest) -> Result<AiResponse> {
        let prompt = build_prompt(&request);
        debug!("pi-cli prompt length: {} chars", prompt.len());

        let mut child =
            spawn_retrying(|| self.command()).map_err(|e| PiCliError::Spawn(e.to_string()))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(prompt.as_bytes()).await?;
            stdin.flush().await?;
        }

        let output = timeout(
            Duration::from_secs(self.timeout_secs),
            child.wait_with_output(),
        )
        .await
        .map_err(|_| PiCliError::Timeout(self.timeout_secs / 60))?
        .map_err(|e| PiCliError::Wait(e.to_string()))?;

        for line in String::from_utf8_lossy(&output.stderr).lines() {
            if !line.trim().is_empty() {
                debug!("[pi-cli stderr] {line}");
            }
        }

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PiCliError::Cli(format!(
                "exited with {}: {}",
                output.status,
                stderr.trim()
            ))
            .into());
        }

        let (text, usage, truncated) = parse_events(&String::from_utf8_lossy(&output.stdout))?;
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

/// Extracts the last assistant message from a `pi --mode json` event stream:
/// its text blocks are the completion, its usage block the accounting, and
/// its stopReason says whether the output cap was hit.
fn parse_events(raw: &str) -> Result<(String, Option<AiUsage>, bool)> {
    let mut text: Option<String> = None;
    let mut usage = None;
    let mut truncated = false;
    for line in raw.lines() {
        let event: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            // A non-JSON line is not one of pi's events; the run that produced
            // it answered the protocol badly enough to fail below.
            Err(_) => continue,
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
            "no assistant message in pi output: {}",
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
    use std::io::Write as _;

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

    /// Writes a fake pi binary: it records its arguments, then prints the
    /// given stdout lines (or sleeps past the timeout). The script lands via
    /// write-to-temp plus rename: executing a file that was just written can
    /// fail with ETXTBSY while the inode still has a writer attached.
    fn fake_pi(dir: &std::path::Path, stdout: &str, exit: Option<&str>) -> std::path::PathBuf {
        let path = dir.join("pi");
        let staging = dir.join("pi.tmp");
        let mut script = String::from("#!/bin/sh\necho \"$@\" > \"$(dirname \"$0\")/args.txt\"\n");
        if let Some(err) = exit {
            script.push_str(&format!("echo {err:?} >&2\nexit 1\n"));
        } else {
            for line in stdout.lines() {
                script.push_str(&format!("echo {line:?}\n"));
            }
        }
        {
            let mut file = std::fs::File::create(&staging).unwrap();
            file.write_all(script.as_bytes()).unwrap();
            file.sync_all().unwrap();
        }
        std::fs::rename(&staging, &path).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
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
            timeout_secs: CALL_TIMEOUT_SECS,
        }
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

        let response = provider(&pi).generate_content(dummy_request()).await?;

        let calls = response.tool_calls.unwrap();
        assert_eq!(calls[0].function_name, "git_log");
        // The prompt reaches the CLI on stdin and the safety flags on the
        // command line; both are load-bearing, so check the recorded args.
        let args = std::fs::read_to_string(temp.path().join("args.txt")).unwrap();
        for flag in [
            "--no-tools",
            "--no-session",
            "--no-extensions",
            "--no-skills",
            "--no-context-files",
            "--mode json",
        ] {
            assert!(args.contains(flag), "missing {flag} in: {args}");
        }
        assert!(!args.contains("--model"), "empty model must stay unset");
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
    async fn test_a_hung_cli_times_out_as_transient() -> Result<()> {
        let temp = tempfile::tempdir().unwrap();
        let pi = fake_pi(temp.path(), "", None);
        // Overwrite the fake with one that hangs past the tiny timeout, via
        // write-to-temp plus rename for the same ETXTBSY reason.
        let staging = temp.path().join("hang.tmp");
        std::fs::write(&staging, "#!/bin/sh\nsleep 30\n").unwrap();
        std::fs::rename(&staging, &pi).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&pi, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let provider = PiCliProvider {
            model: String::new(),
            binary: pi.to_string_lossy().into_owned(),
            timeout_secs: 1,
        };
        let error = provider
            .generate_content(dummy_request())
            .await
            .unwrap_err();
        assert!(
            matches!(classify_ai_error(&error), AiErrorClass::Transient { .. }),
            "a hung CLI must classify as transient, got: {error:#}"
        );
        Ok(())
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
            timeout_secs: CALL_TIMEOUT_SECS,
        };
        let response = provider.generate_content(dummy_request()).await?;
        assert!(response.content.is_some(), "no content: {response:?}");
        Ok(())
    }
}
