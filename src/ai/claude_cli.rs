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

//! AI provider that shells out to the `claude` CLI instead of calling the API directly.
//! This uses the local Claude Code installation (subscription auth) rather than API credits.
//!
//! ## Safety
//!
//! The `claude --print` flag runs in text-completion mode: no tools, no file
//! access, no session persistence, no network calls. The CLI reads a prompt
//! from stdin and writes a response to stdout — it cannot modify the
//! filesystem or execute commands. This makes it inherently safe for use as
//! a completion backend without any additional sandboxing.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::{debug, warn};

use crate::ai::{
    AiErrorClass, AiProvider, AiRequest, AiResponse, AiRole, AiUsage, ClassifyAiError,
    ProviderCapabilities, ToolCall, cache_identity_with,
};
use crate::utils::utf8_prefix;

#[derive(Debug, thiserror::Error)]
pub enum ClaudeCliError {
    #[error("Failed to spawn claude CLI: {0}")]
    Spawn(String),
    #[error("claude CLI timed out after 10 minutes")]
    Timeout,
    #[error("claude CLI wait error: {0}")]
    Wait(String),
    #[error("claude CLI error: {0}")]
    Cli(String),
    #[error("Failed to parse claude CLI JSON output: {0}")]
    Parse(String),
}

impl ClassifyAiError for ClaudeCliError {
    fn ai_error_class(&self) -> AiErrorClass {
        match self {
            ClaudeCliError::Spawn(_) => AiErrorClass::Fatal,
            ClaudeCliError::Timeout => AiErrorClass::Transient {
                retry_after: Duration::from_secs(30),
            },
            ClaudeCliError::Wait(_) => AiErrorClass::Transient {
                retry_after: Duration::from_secs(30),
            },
            ClaudeCliError::Cli(_) => AiErrorClass::Fatal,
            ClaudeCliError::Parse(_) => AiErrorClass::Fatal,
        }
    }
}

pub struct ClaudeCliProvider {
    pub model: String,
    pub effort: Option<String>,
}

#[async_trait]
impl AiProvider for ClaudeCliProvider {
    async fn generate_content(&self, request: AiRequest) -> Result<AiResponse> {
        let prompt = build_prompt(&request);

        debug!("claude-cli prompt length: {} chars", prompt.len());

        let mut args = vec![
            "--print".to_string(),
            "--output-format".to_string(),
            "json".to_string(),
            "--no-session-persistence".to_string(),
        ];

        args.push("--model".to_string());
        args.push(self.model.clone());

        if let Some(effort) = &self.effort {
            args.push("--effort".to_string());
            args.push(effort.clone());
        }

        let mut child = Command::new("claude")
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| ClaudeCliError::Spawn(e.to_string()))?;

        // Write prompt to stdin then close it
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(prompt.as_bytes()).await?;
            stdin.flush().await?;
        }

        // 10-minute timeout per CLI call — a hung claude process won't block forever
        let output = timeout(Duration::from_secs(600), child.wait_with_output())
            .await
            .map_err(|_| ClaudeCliError::Timeout)?
            .map_err(|e| ClaudeCliError::Wait(e.to_string()))?;

        if !output.stderr.is_empty() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            for line in stderr.lines() {
                if !line.trim().is_empty() {
                    debug!("[claude-cli stderr] {}", line);
                }
            }
        }

        let raw = String::from_utf8_lossy(&output.stdout);

        if !output.status.success() {
            // Try to extract the actual error message from the JSON output.
            // The CLI emits a JSON object with is_error=true and the reason
            // in the "result" field even when it exits non-zero.
            if let Ok(outer) = serde_json::from_str::<Value>(&raw)
                && let Some(msg) = outer["result"].as_str()
            {
                return Err(ClaudeCliError::Cli(msg.to_string()).into());
            }
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ClaudeCliError::Cli(format!(
                "exited with {}: {}",
                output.status,
                stderr.trim()
            ))
            .into());
        }
        let outer = parse_cli_output(&raw)?;

        if outer["is_error"].as_bool().unwrap_or(false) {
            return Err(ClaudeCliError::Cli(
                outer["result"]
                    .as_str()
                    .unwrap_or("unknown error")
                    .to_string(),
            )
            .into());
        }

        let result_text = outer["result"].as_str().unwrap_or("").trim().to_string();

        // Parse usage from the outer JSON
        let usage = parse_usage(&outer);

        // Parse the inner response — tool calls or content
        parse_inner_response(&result_text, usage)
    }

    fn get_capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            model_name: self.model.clone(),
            context_window_size: context_window_for_model(&self.model),
        }
    }

    fn cache_identity(&self) -> String {
        cache_identity_with(&self.model, &[("effort", self.effort.as_deref())])
    }
}

fn parse_cli_output(raw: &str) -> Result<Value, ClaudeCliError> {
    serde_json::from_str(raw)
        .map_err(|e| ClaudeCliError::Parse(format!("{}\nRaw: {}", e, utf8_prefix(raw, 200))))
}

/// Pick the context window to advertise for a given model name. The value is
/// currently metadata only (no consumer gates on it), so the mapping is coarse.
/// Opus 4.7 ships with a 1M window by default via Claude Code, and any model
/// can be selected with the `[1m]` suffix to opt into the 1M variant.
/// Verified against Claude Code 2.1.132 for opus-4-7, sonnet-4-6, sonnet-4-6[1m],
/// haiku-4-5.
fn context_window_for_model(model: &str) -> usize {
    if model.contains("[1m]") || model.contains("opus-4-7") {
        1_000_000
    } else {
        200_000
    }
}

/// Build the full text prompt from the AiRequest.
/// Embeds system prompt, conversation history, tool definitions, and instructions.
pub fn build_prompt(request: &AiRequest) -> String {
    let mut out = String::new();

    // System prompt
    if let Some(sys) = &request.system {
        out.push_str("<system>\n");
        out.push_str(sys);
        out.push_str("\n</system>\n\n");
    }

    // Conversation history
    for msg in &request.messages {
        match &msg.role {
            AiRole::System => {
                // Already handled above; skip embedded system messages
            }
            AiRole::User => {
                out.push_str("<user>\n");
                if let Some(c) = &msg.content {
                    out.push_str(c);
                }
                out.push_str("\n</user>\n\n");
            }
            AiRole::Assistant => {
                out.push_str("<assistant>\n");
                if let Some(c) = &msg.content {
                    out.push_str(c);
                }
                if let Some(calls) = &msg.tool_calls {
                    for call in calls {
                        out.push_str(&format!(
                            "<tool_call id=\"{}\" name=\"{}\">\n{}\n</tool_call>\n",
                            call.id, call.function_name, call.arguments
                        ));
                    }
                }
                out.push_str("</assistant>\n\n");
            }
            AiRole::Tool => {
                let id = msg.tool_call_id.as_deref().unwrap_or("?");
                out.push_str(&format!("<tool_result id=\"{}\">\n", id));
                if let Some(c) = &msg.content {
                    out.push_str(c);
                }
                out.push_str("\n</tool_result>\n\n");
            }
        }
    }

    // Tool definitions and response instructions
    if let Some(tools) = &request.tools
        && !tools.is_empty()
    {
        out.push_str("<available_tools>\n");
        for tool in tools {
            out.push_str(&format!(
                "- name: {}\n  description: {}\n  parameters: {}\n\n",
                tool.name, tool.description, tool.parameters
            ));
        }
        out.push_str("</available_tools>\n\n");
        out.push_str(
            "RESPONSE FORMAT: You MUST respond with a SINGLE valid JSON object only (no markdown, no explanation).\n\
             To call tools: {\"tool_calls\": [{\"id\": \"c1\", \"function_name\": \"TOOL_NAME\", \"arguments\": {ARGS}}, {\"id\": \"c2\", \"function_name\": \"OTHER_TOOL\", \"arguments\": {ARGS2}}]}\n\
             Put ALL tool calls in ONE tool_calls array. Do NOT output multiple JSON objects.\n\
             For your final answer: {\"content\": \"YOUR RESPONSE\"}\n\
             Do not mix both. Output exactly one JSON object.\n\
             Do NOT wrap the JSON in XML-style tags (no <assistant>, <tool_calls> or similar tags); output the raw JSON object only.\n",
        );
    } else if let Some(instruction) = request
        .response_format
        .as_ref()
        .and_then(|f| f.format_json_schema_instruction())
    {
        out.push_str(&instruction);
        out.push('\n');
    }

    out
}

fn parse_usage(outer: &Value) -> Option<AiUsage> {
    let u = &outer["usage"];
    if u.is_null() {
        return None;
    }
    let input = u["input_tokens"].as_u64().unwrap_or(0) as usize;
    let output = u["output_tokens"].as_u64().unwrap_or(0) as usize;
    let cache_read = u["cache_read_input_tokens"].as_u64().unwrap_or(0) as usize;
    let cache_write = u["cache_creation_input_tokens"].as_u64().unwrap_or(0) as usize;
    // input_tokens arrives without the cached prefix, so the two cache
    // counts fold back in here.  cached_tokens is a breakdown of
    // prompt_tokens, and a consumer subtracts it to get uncached input.
    let total_input = input + cache_read + cache_write;
    Some(AiUsage {
        prompt_tokens: total_input,
        completion_tokens: output,
        total_tokens: total_input + output,
        cached_tokens: if cache_read > 0 {
            Some(cache_read)
        } else {
            None
        },
    })
}

pub fn parse_inner_response(text: &str, usage: Option<AiUsage>) -> Result<AiResponse> {
    // Try extracting JSON: the whole response, then markdown fence contents,
    // then the contents of XML-style wrapper tags. Models speaking the
    // flattened <user>/<assistant>/<tool_call> conversation format sometimes
    // echo that format back and wrap the payload in tags of their own.
    for candidate in json_candidates(text) {
        if let Ok(v) = serde_json::from_str::<Value>(&candidate) {
            return parse_single_json(&v, &candidate, usage);
        }
        // Maybe JSONL: several JSON objects on separate lines (the model
        // sometimes emits one tool_calls object per line)
        if let Some(response) = merge_jsonl_tool_calls(&candidate, usage.clone()) {
            return Ok(response);
        }
    }

    // Not parseable as JSON — return raw text
    warn!("claude-cli response not valid JSON, returning as raw content");
    Ok(AiResponse {
        content: Some(text.to_string()),
        thought: None,
        thought_signature: None,
        tool_calls: None,
        usage,
        truncated: false,
    })
}

/// Merges per-line JSON objects that carry tool calls into one tool-call
/// response; None when no line both parses and carries tool calls.
fn merge_jsonl_tool_calls(text: &str, usage: Option<AiUsage>) -> Option<AiResponse> {
    let mut merged_tool_calls: Vec<ToolCall> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(trimmed)
            && let Some(calls) = v["tool_calls"].as_array()
        {
            for c in calls {
                if let Some(tc) = parse_tool_call(c) {
                    merged_tool_calls.push(tc);
                }
            }
        }
    }

    (!merged_tool_calls.is_empty()).then_some(AiResponse {
        content: None,
        thought: None,
        thought_signature: None,
        tool_calls: Some(merged_tool_calls),
        usage,
        truncated: false,
    })
}

fn parse_tool_call(c: &Value) -> Option<ToolCall> {
    let id = c["id"].as_str().unwrap_or("c1").to_string();
    let name = c["function_name"].as_str()?.to_string();
    let args = c["arguments"].clone();
    Some(ToolCall {
        id,
        function_name: name,
        arguments: args,
        thought_signature: None,
    })
}

fn parse_single_json(v: &Value, json_str: &str, usage: Option<AiUsage>) -> Result<AiResponse> {
    // A top-level array of tool-call objects: a model that wraps its payload
    // in <tool_calls> tags tends to emit the bare array, not the protocol
    // object that carries it.
    if let Some(items) = v.as_array()
        && !items.is_empty()
        && items.iter().all(|item| item["function_name"].is_string())
    {
        let tool_calls: Vec<ToolCall> = items.iter().filter_map(parse_tool_call).collect();
        return Ok(AiResponse {
            content: None,
            thought: None,
            thought_signature: None,
            tool_calls: Some(tool_calls),
            usage,
            truncated: false,
        });
    }

    // Tool calls?
    if let Some(calls) = v["tool_calls"].as_array() {
        let tool_calls: Vec<ToolCall> = calls.iter().filter_map(parse_tool_call).collect();

        if !tool_calls.is_empty() {
            return Ok(AiResponse {
                content: None,
                thought: None,
                thought_signature: None,
                tool_calls: Some(tool_calls),
                usage,
                truncated: false,
            });
        }
    }

    // Content field?
    if let Some(content) = v["content"].as_str() {
        return Ok(AiResponse {
            content: Some(content.to_string()),
            thought: None,
            thought_signature: None,
            tool_calls: None,
            usage,
            truncated: false,
        });
    }

    // Any other JSON — return it as content string (e.g. {"concerns": [...]})
    Ok(AiResponse {
        content: Some(json_str.to_string()),
        thought: None,
        thought_signature: None,
        tool_calls: None,
        usage,
        truncated: false,
    })
}

/// How deep to recurse into nested wrapper tags. Observed wrappers nest at
/// most two levels (<assistant><tool_calls>...</tool_calls></assistant>).
const WRAPPER_DEPTH: usize = 3;

/// Collects every substring that could be the model's JSON payload, in
/// falling order of protocol compliance: the whole response, the contents of
/// markdown code fences, then the contents of XML-style wrapper tags.
///
/// Does NOT try to find outermost braces — that can silently produce invalid
/// JSON when the text contains multiple objects (e.g. JSONL), which the JSONL
/// fallback in parse_inner_response handles better.
fn json_candidates(text: &str) -> Vec<String> {
    let normalized = text.replace("\r\n", "\n");
    let mut candidates = vec![normalized.trim().to_string()];
    // Strip markdown fences — handle optional language tags, first fence only
    for fence_start in &["```json\n", "```JSON\n", "```\n"] {
        if let Some(start) = normalized.find(fence_start) {
            let after = &normalized[start + fence_start.len()..];
            if let Some(end) = after.find("\n```") {
                candidates.push(after[..end].trim().to_string());
            }
        }
    }
    let mut wrapped = Vec::new();
    wrapper_tag_contents(&normalized, &mut wrapped, WRAPPER_DEPTH);
    candidates.extend(wrapped);
    candidates
}

/// Collects the contents of XML-style wrapper tags (<name ...>...</name>) in
/// document order, recursing into each tag's content so nested wrappers are
/// offered both outside-in and inside-out.
///
/// A '<' that opens no valid tag (comparison operators, stray brackets) is
/// skipped; the scan never panics on malformed input.
fn wrapper_tag_contents(text: &str, out: &mut Vec<String>, depth: usize) {
    if depth == 0 {
        return;
    }
    let mut rest = text;
    while let Some(open) = rest.find('<') {
        let after_open = &rest[open + 1..];
        let Some(name_end) = after_open.find(|c: char| c == '>' || c.is_whitespace()) else {
            return;
        };
        let name = &after_open[..name_end];
        let valid_name = !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        if !valid_name {
            rest = after_open;
            continue;
        }
        let Some(open_end) = after_open.find('>') else {
            return;
        };
        let content_start = open + 1 + open_end + 1;
        let closing = format!("</{name}>");
        let Some(close_rel) = rest[content_start..].find(&closing) else {
            // An opening tag with no matching close: keep scanning after it.
            rest = after_open;
            continue;
        };
        let content_end = content_start + close_rel;
        let inner = rest[content_start..content_end].trim();
        out.push(inner.to_string());
        wrapper_tag_contents(inner, out, depth - 1);
        rest = &rest[content_end + closing.len()..];
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{AiMessage, AiRequest, AiResponseFormat, AiRole, AiTool};
    use serde_json::json;

    #[test]
    fn test_parse_error_preview_handles_multibyte_cutoff() {
        let raw = format!("{}🙂not-json", "a".repeat(199));

        let error = parse_cli_output(&raw).unwrap_err();

        assert!(matches!(&error, ClaudeCliError::Parse(_)));
        assert!(error.to_string().contains(&"a".repeat(199)));
    }

    fn make_request(messages: Vec<AiMessage>) -> AiRequest {
        AiRequest {
            system: None,
            messages,
            tools: None,
            temperature: None,
            response_format: None,
            context_tag: None,
        }
    }

    fn simple_user_msg() -> Vec<AiMessage> {
        vec![AiMessage {
            role: AiRole::User,
            content: Some("hi".to_string()),
            thought: None,
            thought_signature: None,
            tool_calls: None,
            tool_call_id: None,
        }]
    }

    #[test]
    fn test_build_prompt_json_format_without_tools() {
        let mut req = make_request(simple_user_msg());
        req.response_format = Some(AiResponseFormat::Json { schema: None });

        let prompt = build_prompt(&req);
        assert!(prompt.contains("RESPONSE FORMAT"));
        assert!(prompt.contains("ONLY a valid JSON object"));
        assert!(!prompt.contains("tool_calls"));
    }

    #[test]
    fn test_build_prompt_json_format_with_schema() {
        let mut req = make_request(simple_user_msg());
        req.response_format = Some(AiResponseFormat::Json {
            schema: Some(
                json!({"type": "object", "properties": {"selected_prompts": {"type": "array"}}}),
            ),
        });

        let prompt = build_prompt(&req);
        assert!(prompt.contains("RESPONSE FORMAT"));
        assert!(prompt.contains("selected_prompts"));
        assert!(prompt.contains("matching this schema"));
    }

    #[test]
    fn test_build_prompt_with_tools_includes_format() {
        let mut req = make_request(simple_user_msg());
        req.tools = Some(vec![AiTool {
            name: "git_log".to_string(),
            description: "Show git log".to_string(),
            parameters: json!({"type": "object"}),
        }]);

        let prompt = build_prompt(&req);
        assert!(prompt.contains("RESPONSE FORMAT"));
        assert!(prompt.contains("tool_calls"));
        assert!(prompt.contains("<available_tools>"));
    }

    #[test]
    fn test_build_prompt_text_format_no_instruction() {
        let mut req = make_request(simple_user_msg());
        req.response_format = Some(AiResponseFormat::Text);

        let prompt = build_prompt(&req);
        assert!(!prompt.contains("RESPONSE FORMAT"));
    }

    #[test]
    fn test_build_prompt_no_format_no_instruction() {
        let req = make_request(simple_user_msg());

        let prompt = build_prompt(&req);
        assert!(!prompt.contains("RESPONSE FORMAT"));
    }

    #[test]
    fn test_parse_usage_folds_cache_counts_into_prompt_tokens() {
        let outer = serde_json::json!({
            "usage": {
                "input_tokens": 500,
                "output_tokens": 20,
                "cache_read_input_tokens": 19500,
                "cache_creation_input_tokens": 1000,
            }
        });

        let usage = parse_usage(&outer).unwrap();

        // Uncached input is prompt_tokens less the cached breakdown, so it
        // covers the fresh input and the prefix the model had to write.
        assert_eq!(usage.prompt_tokens, 21000);
        assert_eq!(usage.cached_tokens, Some(19500));
        assert_eq!(usage.total_tokens, 21020);
    }

    #[test]
    fn test_parse_usage_without_a_cache_read_reports_no_cached_tokens() {
        let outer = serde_json::json!({
            "usage": {
                "input_tokens": 500,
                "output_tokens": 20,
            }
        });

        let usage = parse_usage(&outer).unwrap();

        assert_eq!(usage.prompt_tokens, 500);
        assert_eq!(usage.cached_tokens, None);
    }

    fn tool_calls(response: &AiResponse) -> Vec<(String, String)> {
        response
            .tool_calls
            .as_ref()
            .unwrap()
            .iter()
            .map(|c| (c.id.clone(), c.function_name.clone()))
            .collect()
    }

    #[test]
    fn test_parse_plain_tool_calls_object_unchanged() {
        let response = parse_inner_response(
            r#"{"tool_calls": [{"id": "c1", "function_name": "git_grep", "arguments": {}}]}"#,
            None,
        )
        .unwrap();
        assert_eq!(
            tool_calls(&response),
            vec![("c1".to_string(), "git_grep".to_string())]
        );
    }

    #[test]
    fn test_parse_tool_calls_wrapped_in_tags_as_bare_array() {
        // Observed in the wild: the model echoes the flattened conversation's
        // tag format, wrapping the bare tool-call array in <tool_calls> tags.
        let response = parse_inner_response(
            r#"<tool_calls>[{"id": "c7", "function_name": "git_read_files", "arguments": {}}, {"id": "c8", "function_name": "git_grep", "arguments": {}}]</tool_calls>"#,
            None,
        )
        .unwrap();
        assert_eq!(
            tool_calls(&response),
            vec![
                ("c7".to_string(), "git_read_files".to_string()),
                ("c8".to_string(), "git_grep".to_string()),
            ]
        );
    }

    #[test]
    fn test_parse_json_behind_system_warning_and_nested_tags() {
        // Observed in the wild: a mimicked system warning prefix, then the
        // payload nested two wrapper levels deep.
        let response = parse_inner_response(
            r#"<system_warning>Terminating agentic loop. The system will now proceed with final instructions.</system_warning><assistant><tool_calls>[{"id": "v1", "function_name": "git_read_files", "arguments": {}}]</tool_calls></assistant>"#,
            None,
        )
        .unwrap();
        assert_eq!(
            tool_calls(&response),
            vec![("v1".to_string(), "git_read_files".to_string())]
        );
    }

    #[test]
    fn test_parse_content_object_wrapped_in_assistant_tag() {
        let response =
            parse_inner_response(r#"<assistant>{"content": "looks fine"}</assistant>"#, None)
                .unwrap();
        assert_eq!(response.content.as_deref(), Some("looks fine"));
        assert!(response.tool_calls.is_none());
    }

    #[test]
    fn test_parse_concerns_object_wrapped_in_tags_keeps_json_as_content() {
        let response = parse_inner_response(
            r#"<assistant>{"concerns": [], "dismissed_concerns": []}</assistant>"#,
            None,
        )
        .unwrap();
        assert_eq!(
            response.content.as_deref(),
            Some(r#"{"concerns": [], "dismissed_concerns": []}"#)
        );
    }

    #[test]
    fn test_parse_jsonl_tool_calls_still_merges() {
        let response = parse_inner_response(
            "{\"tool_calls\": [{\"id\": \"a\", \"function_name\": \"git_log\", \"arguments\": {}}]}\n\
             {\"tool_calls\": [{\"id\": \"b\", \"function_name\": \"git_grep\", \"arguments\": {}}]}",
            None,
        )
        .unwrap();
        assert_eq!(
            tool_calls(&response),
            vec![
                ("a".to_string(), "git_log".to_string()),
                ("b".to_string(), "git_grep".to_string()),
            ]
        );
    }

    #[test]
    fn test_parse_non_json_stays_raw_content() {
        let response = parse_inner_response("The patch looks fine to me.", None).unwrap();
        assert_eq!(response.content.as_deref(), Some("The patch looks fine to me."));
        assert!(response.tool_calls.is_none());
    }

    #[test]
    fn test_json_candidates_ignores_stray_angle_brackets() {
        let text = r#"if (a < b) then {"content": "x"}"#;
        // The comparison must not be mistaken for a tag; the whole text stays
        // the first candidate and no wrapper candidate is produced.
        assert_eq!(
            json_candidates(text),
            vec![r#"if (a < b) then {"content": "x"}"#.to_string()]
        );
    }

    #[test]
    fn test_wrapper_scan_recurses_in_document_order() {
        let mut out = Vec::new();
        wrapper_tag_contents(
            "<a>one<b>two</b></a><c>three</c>",
            &mut out,
            WRAPPER_DEPTH,
        );
        assert_eq!(out, vec!["one<b>two</b>", "two", "three"]);
    }
}
