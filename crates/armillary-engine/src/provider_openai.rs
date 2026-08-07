//! The OpenAI chat-completions dialect, behind the same `ModelProvider` seam.
//!
//! One provider, endpoint-parameterized — built for OpenCode Zen
//! (`opencode.ai/zen/v1`) but carrying nothing Zen-specific beyond what
//! `main.rs` constructs it with. pi's provider-quirks-as-compat-data lesson is
//! deliberately deferred: a `Compat` struct with one consumer and zero
//! measured quirks is scaffolding theater. It is introduced by whichever lands
//! first — a second OpenAI-shape endpoint, or the first measured quirk needing
//! a switch. This struct's fields are the seam it will slot into.
//!
//! **Reasoning is not captured in this dialect** (design decision 4,
//! 2026-08-07): `reasoning_content` is documented response-only — there is no
//! replay contract to honor, and capturing it into `ContentBlock::Thinking`
//! would poison cross-provider replay (an empty-signature block persisted
//! from here would read as tampered if a workspace switches `--model` to an
//! Anthropic family). When a display consumer is ratified, the shape is a new
//! display-only variant no request builder ever emits.

use crate::projection::{ContentBlock, ProviderRole};
use crate::provider::{
    parse_sse_data_line, ModelProvider, ProviderError, ToolChoice, TurnOutcome, TurnRequest,
    MAX_TOKENS,
};
use futures_util::StreamExt;
use tokio::sync::{mpsc, watch};

/// Calls an OpenAI-compatible chat-completions API over SSE.
pub struct OpenAiCompatProvider {
    /// Through `/chat/completions` — no trailing slash.
    pub base_url: String,
    /// The bare slug the endpoint knows (`kimi-k3`), not the engine's
    /// prefixed spelling (`zen/kimi-k3`) — the prefix is provenance for the
    /// log, and this field is what crosses the wire.
    pub model: String,
    pub api_key: String,
}

/// Hand-written to redact `api_key` — same discipline as `AnthropicProvider`.
/// Never derive `Debug` here.
impl std::fmt::Debug for OpenAiCompatProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatProvider")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

/// Build the chat-completions request body for one turn.
///
/// The two structural departures from the Anthropic shape, both pinned by
/// tests: tool calls ride a `tool_calls` array on the assistant message with
/// JSON-*encoded* arguments, and tool results get their own `role: "tool"`
/// messages where the Anthropic dialect rides them inside user turns.
/// `Thinking`/`RedactedThinking` blocks are dropped at this boundary — the log
/// still carries them; this dialect has no replay slot (module doc).
pub(crate) fn build_openai_request_body(model: &str, req: &TurnRequest) -> serde_json::Value {
    let mut messages: Vec<serde_json::Value> = Vec::new();
    if let Some(system) = &req.turn.system {
        messages.push(serde_json::json!({ "role": "system", "content": system }));
    }
    for m in &req.turn.messages {
        push_message(&mut messages, m);
    }

    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": messages,
        "stream": true,
    });
    if !req.tools.is_empty() {
        let tools: Vec<serde_json::Value> =
            req.tools.iter().map(|d| d.openai_definition()).collect();
        body["tools"] = serde_json::json!(tools);
    }
    if let Some(choice) = &req.tool_choice {
        body["tool_choice"] = match choice {
            ToolChoice::ForceText => serde_json::json!("none"),
        };
    }
    body
}

/// Fold one flattened `ProviderMessage` into completions messages.
///
/// A user message holding only tool results emits no user message at all —
/// the `role: "tool"` messages carry everything. An assistant message emits
/// `content` only when it has text and `tool_calls` only when it has calls;
/// a message that ends up with neither emits nothing (possible only if it
/// held nothing but thinking blocks, which this boundary drops).
fn push_message(messages: &mut Vec<serde_json::Value>, m: &crate::projection::ProviderMessage) {
    let role = match m.role {
        ProviderRole::User => "user",
        ProviderRole::Assistant => "assistant",
    };

    let mut text = String::new();
    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    let mut tool_results: Vec<serde_json::Value> = Vec::new();

    for block in &m.content {
        match block {
            ContentBlock::Text(t) => text.push_str(t),
            ContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        // JSON-encoded, not nested — the dialect's shape.
                        "arguments": serde_json::to_string(input)
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                }));
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error: _,
            } => {
                // No `is_error` slot in this dialect: the rendered content
                // already carries the machine code (P-4's "as far toward the
                // boundary as the channel allows" — here the channel allows
                // less, and the content is all of it).
                tool_results.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": content,
                }));
            }
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
        }
    }

    if !text.is_empty() || !tool_calls.is_empty() {
        let mut msg = serde_json::json!({ "role": role });
        if !text.is_empty() {
            msg["content"] = serde_json::json!(text);
        }
        if !tool_calls.is_empty() {
            msg["tool_calls"] = serde_json::json!(tool_calls);
        }
        messages.push(msg);
    }
    messages.extend(tool_results);
}

/// A tool call still being streamed. `arguments` is the raw concatenation of
/// fragments — same rule as the Anthropic accumulator: no fragment is valid
/// JSON on its own, so parsing happens once, at materialization.
#[derive(Debug, Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

/// Folds a chat-completions SSE stream into text, calls, and a stop reason.
///
/// Frames are `choices[0].delta`: `content` appends to one text buffer (this
/// dialect has no block indexes for text), `tool_calls[i]` upserts by its own
/// `index` field, and `finish_reason` arrives on the closing frame. The
/// `data: [DONE]` sentinel is detected on the raw line by the transport loop,
/// before JSON parsing — it is not JSON.
///
/// `reasoning_content` deltas have no arm here and are ignored by
/// construction — see the module doc for why that is a decision, not a gap.
#[derive(Debug, Default)]
struct OpenAiStreamAccumulator {
    text: String,
    calls: std::collections::BTreeMap<u64, PartialCall>,
    stop_reason: Option<String>,
}

impl OpenAiStreamAccumulator {
    fn observe(&mut self, v: &serde_json::Value) {
        let Some(choice) = v.get("choices").and_then(|c| c.get(0)) else {
            return;
        };

        if let Some(delta) = choice.get("delta") {
            if let Some(t) = delta.get("content").and_then(|t| t.as_str()) {
                self.text.push_str(t);
            }
            if let Some(calls) = delta.get("tool_calls").and_then(|c| c.as_array()) {
                for frag in calls {
                    let Some(index) = frag.get("index").and_then(|i| i.as_u64()) else {
                        continue;
                    };
                    let call = self.calls.entry(index).or_default();
                    if let Some(id) = frag.get("id").and_then(|i| i.as_str()) {
                        call.id.push_str(id);
                    }
                    if let Some(f) = frag.get("function") {
                        if let Some(name) = f.get("name").and_then(|n| n.as_str()) {
                            call.name.push_str(name);
                        }
                        if let Some(args) = f.get("arguments").and_then(|a| a.as_str()) {
                            call.arguments.push_str(args);
                        }
                    }
                }
            }
        }

        if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
            // Normalized into the engine's vocabulary so the log speaks one
            // language regardless of pilot; an unknown value passes through
            // verbatim rather than being swallowed — normalized, never lied
            // about.
            self.stop_reason = Some(
                match reason {
                    "stop" => "end_turn",
                    "tool_calls" => "tool_use",
                    "length" => "max_tokens",
                    other => other,
                }
                .to_string(),
            );
        }
    }

    /// The turn's visible text — I-4's snapshot source, same as Anthropic's.
    fn text(&self) -> String {
        self.text.clone()
    }

    /// Materialize: text first (when any), then calls in index order.
    ///
    /// A call whose non-empty arguments do not parse was cut mid-JSON and is
    /// dropped, never salvaged; empty arguments are a complete call with an
    /// empty input. Both rules inherited from the Anthropic accumulator with
    /// their rationale (`provider.rs`, "two failures that look alike").
    fn blocks(&self) -> Vec<ContentBlock> {
        let mut out = Vec::new();
        if !self.text.is_empty() {
            out.push(ContentBlock::Text(self.text.clone()));
        }
        for call in self.calls.values() {
            let input = if call.arguments.trim().is_empty() {
                Some(serde_json::Value::Object(Default::default()))
            } else {
                serde_json::from_str(&call.arguments).ok()
            };
            if let Some(input) = input {
                out.push(ContentBlock::ToolUse {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    input,
                });
            }
        }
        out
    }
}

/// True for this dialect's end-of-stream sentinel. Checked on the raw line
/// BEFORE `parse_sse_data_line` — `[DONE]` is not JSON and never parses.
fn is_done_line(line: &str) -> bool {
    line.trim() == "data: [DONE]"
}

impl OpenAiCompatProvider {
    /// Fold the accumulator into an outcome — one place, mirroring
    /// `AnthropicProvider::outcome`, so the exit paths cannot disagree.
    fn outcome(&self, acc: OpenAiStreamAccumulator, stopped: bool) -> TurnOutcome {
        TurnOutcome {
            text: acc.text(),
            blocks: acc.blocks(),
            stop_reason: acc.stop_reason.clone(),
            stopped,
            model: self.model.clone(),
        }
    }
}

#[async_trait::async_trait]
impl ModelProvider for OpenAiCompatProvider {
    async fn run_turn(
        &self,
        req: TurnRequest,
        sink: mpsc::Sender<String>,
        mut cancel: watch::Receiver<bool>,
    ) -> Result<TurnOutcome, ProviderError> {
        if *cancel.borrow() {
            return Ok(TurnOutcome {
                text: String::new(),
                blocks: Vec::new(),
                stop_reason: None,
                stopped: true,
                model: self.model.clone(),
            });
        }

        let body = build_openai_request_body(&self.model, &req);

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/chat/completions", self.base_url))
            .header("authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body: body_text,
            });
        }

        let mut byte_stream = response.bytes_stream();
        let mut line_buffer = String::new();
        let mut acc = OpenAiStreamAccumulator::default();

        loop {
            tokio::select! {
                biased;

                changed = cancel.changed() => {
                    if changed.is_err() || *cancel.borrow() {
                        return Ok(self.outcome(acc, true));
                    }
                }

                chunk = byte_stream.next() => {
                    match chunk {
                        Some(Ok(bytes)) => {
                            line_buffer.push_str(&String::from_utf8_lossy(&bytes));
                            while let Some(idx) = line_buffer.find('\n') {
                                let line = line_buffer[..idx].trim_end_matches('\r').to_string();
                                line_buffer.drain(..=idx);

                                if is_done_line(&line) {
                                    return Ok(self.outcome(acc, false));
                                }
                                let Some(v) = parse_sse_data_line(&line) else { continue };

                                let had_text = v
                                    .get("choices")
                                    .and_then(|c| c.get(0))
                                    .and_then(|c| c.get("delta"))
                                    .and_then(|d| d.get("content"))
                                    .and_then(|t| t.as_str())
                                    .is_some_and(|t| !t.is_empty());
                                acc.observe(&v);

                                if had_text {
                                    // I-4: snapshot conversion at the source —
                                    // the sink never sees a raw delta.
                                    let _ = sink.send(acc.text()).await;
                                }
                            }
                        }
                        Some(Err(e)) => {
                            return Err(ProviderError::Http(e.to_string()));
                        }
                        None => {
                            return Ok(self.outcome(acc, false));
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{ModelTurn, ProviderMessage};

    /// One message whose content is exactly one text block — mirrors
    /// `provider.rs`'s test helper.
    fn text_msg(role: ProviderRole, s: &str) -> ProviderMessage {
        ProviderMessage {
            role,
            content: vec![ContentBlock::Text(s.to_string())],
        }
    }

    // --- the request body ---

    #[test]
    fn a_text_turn_builds_the_completions_shape_with_system_leading() {
        let turn = ModelTurn {
            system: Some("# boot".to_string()),
            messages: vec![
                text_msg(ProviderRole::User, "hi"),
                text_msg(ProviderRole::Assistant, "hello"),
                text_msg(ProviderRole::User, "bye"),
            ],
        };

        assert_eq!(
            build_openai_request_body("kimi-k3", &TurnRequest::bare(turn)),
            serde_json::json!({
                "model": "kimi-k3",
                "max_tokens": 64000,
                "stream": true,
                "messages": [
                    { "role": "system", "content": "# boot" },
                    { "role": "user", "content": "hi" },
                    { "role": "assistant", "content": "hello" },
                    { "role": "user", "content": "bye" },
                ],
            })
        );
    }

    #[test]
    fn a_tool_round_trip_splits_into_tool_calls_and_a_tool_role_message() {
        // The two structural departures from the Anthropic shape: calls ride
        // a `tool_calls` array with JSON-ENCODED arguments, and results get
        // their own `role: "tool"` message where Anthropic rides them in user
        // turns.
        let turn = ModelTurn {
            system: None,
            messages: vec![
                text_msg(ProviderRole::User, "read it"),
                ProviderMessage {
                    role: ProviderRole::Assistant,
                    content: vec![
                        ContentBlock::Text("let me look".to_string()),
                        ContentBlock::ToolUse {
                            id: "call_1".to_string(),
                            name: "read_file".to_string(),
                            input: serde_json::json!({"path": "a.md"}),
                        },
                    ],
                },
                ProviderMessage {
                    role: ProviderRole::User,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: "call_1".to_string(),
                        content: "a.md lines 1-1".to_string(),
                        is_error: false,
                    }],
                },
            ],
        };
        let body = build_openai_request_body("m", &TurnRequest::bare(turn));

        assert_eq!(
            body["messages"],
            serde_json::json!([
                { "role": "user", "content": "read it" },
                { "role": "assistant", "content": "let me look", "tool_calls": [
                    { "id": "call_1", "type": "function",
                      "function": { "name": "read_file", "arguments": "{\"path\":\"a.md\"}" } }
                ]},
                { "role": "tool", "tool_call_id": "call_1", "content": "a.md lines 1-1" },
            ])
        );
    }

    #[test]
    fn thinking_blocks_never_cross_this_dialects_boundary() {
        // Decision 4: this dialect documents reasoning as response-only. The
        // log still carries the blocks; THIS boundary drops them.
        let turn = ModelTurn {
            system: None,
            messages: vec![
                text_msg(ProviderRole::User, "hi"),
                ProviderMessage {
                    role: ProviderRole::Assistant,
                    content: vec![
                        ContentBlock::Thinking {
                            thinking: "hmm".to_string(),
                            signature: "sig".to_string(),
                        },
                        ContentBlock::Text("answer".to_string()),
                    ],
                },
            ],
        };
        let body = build_openai_request_body("m", &TurnRequest::bare(turn));

        assert_eq!(
            body["messages"][1],
            serde_json::json!({ "role": "assistant", "content": "answer" })
        );
    }

    #[test]
    fn tools_and_force_text_project_in_this_dialect() {
        let def = crate::tools::ToolDef {
            name: "specimen",
            description: "d".to_string(),
            schema: serde_json::json!({ "type": "object", "properties": {}, "required": [] }),
        };
        let turn = ModelTurn {
            system: None,
            messages: vec![text_msg(ProviderRole::User, "hi")],
        };
        let body = build_openai_request_body(
            "m",
            &TurnRequest {
                turn,
                tools: vec![def.clone()],
                tool_choice: Some(ToolChoice::ForceText),
            },
        );

        assert_eq!(body["tools"], serde_json::json!([def.openai_definition()]));
        assert_eq!(body["tool_choice"], serde_json::json!("none"));
    }

    // --- the stream accumulator ---

    #[test]
    fn deltas_accumulate_text_tool_fragments_and_a_normalized_stop() {
        let mut acc = OpenAiStreamAccumulator::default();
        for v in [
            serde_json::json!({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
            serde_json::json!({"choices": [{"delta": {"content": "on "}}]}),
            serde_json::json!({"choices": [{"delta": {"content": "it"}}]}),
            serde_json::json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": ""}}]}}]}),
            serde_json::json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "{\"pa"}}]}}]}),
            serde_json::json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "th\": \"a.md\"}"}}]}}]}),
            serde_json::json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        ] {
            acc.observe(&v);
        }

        assert_eq!(acc.text(), "on it");
        assert_eq!(
            acc.blocks(),
            vec![
                ContentBlock::Text("on it".to_string()),
                ContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "a.md"}),
                },
            ]
        );
        assert_eq!(acc.stop_reason.as_deref(), Some("tool_use"));
    }

    #[test]
    fn finish_reasons_normalize_into_the_engines_vocabulary() {
        // stop→end_turn, tool_calls→tool_use, length→max_tokens; an unknown
        // value passes through verbatim rather than being swallowed — the log
        // speaks one vocabulary regardless of pilot, but never lies.
        for (wire, ours) in [
            ("stop", "end_turn"),
            ("tool_calls", "tool_use"),
            ("length", "max_tokens"),
            ("content_filter", "content_filter"),
        ] {
            let mut acc = OpenAiStreamAccumulator::default();
            acc.observe(&serde_json::json!({"choices": [{"delta": {}, "finish_reason": wire}]}));
            assert_eq!(acc.stop_reason.as_deref(), Some(ours), "{wire}");
        }
    }

    #[test]
    fn truncated_tool_arguments_drop_the_call_and_reasoning_content_is_ignored() {
        // Both rules inherited with their rationale: a half-written path is a
        // read of the wrong file (provider.rs's two-failures doc), and
        // decision 4 keeps this dialect's response-only reasoning out of the
        // engine.
        let mut acc = OpenAiStreamAccumulator::default();
        for v in [
            serde_json::json!({"choices": [{"delta": {"reasoning_content": "thinking hard"}}]}),
            serde_json::json!({"choices": [{"delta": {"content": "text"}}]}),
            serde_json::json!({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": "{\"pa"}}]}}]}),
        ] {
            acc.observe(&v);
        }

        assert_eq!(acc.text(), "text");
        assert_eq!(acc.blocks(), vec![ContentBlock::Text("text".to_string())]);
    }

    #[test]
    fn the_done_sentinel_is_detected_on_the_raw_line() {
        // `[DONE]` is not JSON; the transport loop must catch it before the
        // parser would discard it as unparseable.
        assert!(is_done_line("data: [DONE]"));
        assert!(is_done_line("data: [DONE]\r".trim_end_matches('\r')));
        assert!(!is_done_line(r#"data: {"choices":[]}"#));
    }

    // --- live (needs a zen key and network) ---

    #[tokio::test]
    #[ignore] // needs OPENCODE_ZEN_API_KEY or ~/.config/armillary/zen-key, and network
    async fn a_live_zen_turn_answers_and_can_call_a_tool() {
        let key = std::env::var("OPENCODE_ZEN_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
            .or_else(|| {
                let home = std::env::var("HOME").ok()?;
                let s = std::fs::read_to_string(
                    std::path::Path::new(&home).join(".config/armillary/zen-key"),
                )
                .ok()?;
                let t = s.trim().to_string();
                (!t.is_empty()).then_some(t)
            })
            .expect("no zen key on this machine");
        let provider = OpenAiCompatProvider {
            base_url: "https://opencode.ai/zen/v1".to_string(),
            // Free tier; adjust to Zen's current slug if this 404s — the
            // catalog is the account's, not this repo's.
            model: "deepseek-v4-flash".to_string(),
            api_key: key,
        };
        let turn = ModelTurn {
            system: None,
            messages: vec![text_msg(
                ProviderRole::User,
                "Call the get_composition tool, then say done.",
            )],
        };
        let defs = vec![crate::tools::registry()[0].def.clone()];
        let (tx, mut rx) = mpsc::channel(32);
        tokio::spawn(async move { while rx.recv().await.is_some() {} });
        let (_c, cancel_rx) = watch::channel(false);

        let outcome = provider
            .run_turn(
                TurnRequest {
                    turn,
                    tools: defs,
                    tool_choice: None,
                },
                tx,
                cancel_rx,
            )
            .await
            .expect("live zen turn failed");

        assert!(
            outcome.blocks.iter().any(
                |b| matches!(b, ContentBlock::ToolUse { name, .. } if name == "get_composition")
            ) || !outcome.text.is_empty(),
            "neither a tool call nor text came back: {outcome:?}"
        );
    }
}
