//! The model provider — Anthropic streaming behind a trait.
//!
//! P-4: this layer consumes the already-flattened `ModelTurn` produced by
//! `projection::project_context`; it never sees a typed log event, and it
//! never re-derives the message shape from anything but the `ModelTurn` it
//! is handed. I-4: the sink this trait writes to receives the FULL
//! text-so-far on every call — a snapshot, not an increment — because I-4
//! says a transient's payload must be a snapshot, and the cheapest place to
//! guarantee that is at the one source that ever produces these strings,
//! not at every consumer downstream of it.
//!
//! `ModelProvider` is the testable seam Task 11's loop is written against:
//! it stores `Arc<dyn ModelProvider>`, which is why the trait goes through
//! `async_trait` rather than return-position `impl Trait` — RPITIT is not
//! dyn-compatible, and a boxed trait object is the whole point here.

use crate::projection::{ContentBlock, ModelTurn, ProviderRole};
use futures_util::StreamExt;
use std::time::Duration;
use tokio::sync::{mpsc, watch};

/// `max_tokens: 4096` below is a v0 default this task picked, not a rule
/// this codebase or Anthropic's API imposes — a later task is free to make
/// it configurable per `ModelConfig` without this module's shape changing.
const MAX_TOKENS: u32 = 4096;

/// What one turn produced. `stopped` is true only when `cancel` fired before
/// the model finished on its own — a normal end-of-stream (or a scripted
/// provider exhausting its fragments) is `stopped: false` even if the text
/// is short.
///
/// `text` is retained as the turn's visible prose (it is what the durable
/// `assistant_message` records and what the transient sink streamed), but it is
/// no longer the whole outcome: `blocks` carries every content block in wire
/// order, and `stop_reason` is the provider's own account of why generation
/// ended. A tool call is expressible in neither of the first two fields, which
/// is why a loop written against the old shape could not see one.
///
/// No `Eq`: `blocks` may contain a `serde_json::Value`.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnOutcome {
    pub text: String,
    pub blocks: Vec<ContentBlock>,
    /// `end_turn` | `tool_use` | `max_tokens` | `refusal` | … Read from the
    /// `message_delta` frame; `None` when the stream ended without one (a
    /// transport cut, or a cancel before the first frame).
    pub stop_reason: Option<String>,
    pub stopped: bool,
    pub model: String,
}

/// Failure modes this layer can surface. Never wraps anything that could
/// print `api_key` — see `AnthropicProvider`'s hand-written `Debug` for the
/// other half of that guarantee.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderError {
    /// Transport-level failure (connect, TLS, body read) — the `String` is
    /// `reqwest::Error`'s `Display`, which does not include request headers.
    Http(String),
    /// A non-2xx response, with the status and body Anthropic sent back.
    Api { status: u16, body: String },
    /// No credential to call out with at all (`KeylessProvider`) — the
    /// Explorer keeps working with no model wired in; this is how a caller
    /// finds that out at first `send`, not at boot.
    NoApiKey,
}

/// Streams one turn's model output. `sink` receives the full accumulated
/// text on every emission (a snapshot — see the module doc); `cancel`
/// carries the loop's stop signal (Task 11 flips it to interrupt a
/// generation in flight). Object-safe so callers can hold `Arc<dyn
/// ModelProvider>`.
#[async_trait::async_trait]
pub trait ModelProvider: Send + Sync + 'static {
    async fn run_turn(
        &self,
        turn: ModelTurn,
        sink: mpsc::Sender<String>,
        cancel: watch::Receiver<bool>,
    ) -> Result<TurnOutcome, ProviderError>;
}

/// Calls the real Anthropic Messages API over SSE.
pub struct AnthropicProvider {
    pub model: String,
    pub api_key: String,
}

/// Hand-written to redact `api_key` — mirrors `state::ModelConfig`'s
/// `Debug` impl. Never derive `Debug` here; a derive would print the key
/// verbatim the first time this struct lands in a log line or an `unwrap`
/// panic message.
impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

/// Parses one line of an Anthropic SSE stream. Pure: no I/O, so the
/// fixture-driven tests below exercise the real wire shape with no network.
/// Returns `None` for anything that isn't a `data:` line (an `event:` line,
/// a blank keep-alive line, or a `[DONE]` sentinel) or whose payload isn't
/// valid JSON.
fn parse_sse_data_line(line: &str) -> Option<serde_json::Value> {
    let payload = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:"))?;
    let payload = payload.trim();
    if payload.is_empty() || payload == "[DONE]" {
        return None;
    }
    serde_json::from_str(payload).ok()
}

/// Pulls the incremental text out of a `content_block_delta` frame whose
/// `delta.type` is `text_delta`. Every other frame shape (`message_start`,
/// `ping`, `content_block_stop`, `message_delta`, `message_stop`, …) yields
/// `None` — this function contributes text, nothing else.
fn extract_text_delta(v: &serde_json::Value) -> Option<&str> {
    if v.get("type").and_then(|t| t.as_str()) != Some("content_block_delta") {
        return None;
    }
    let delta = v.get("delta")?;
    if delta.get("type").and_then(|t| t.as_str()) != Some("text_delta") {
        return None;
    }
    delta.get("text").and_then(|t| t.as_str())
}

fn is_message_stop(v: &serde_json::Value) -> bool {
    v.get("type").and_then(|t| t.as_str()) == Some("message_stop")
}

/// A block still being streamed. `ToolUse.json` is the raw concatenation of
/// `input_json_delta` fragments — no fragment is valid JSON on its own, so
/// parsing happens once, at materialization.
#[derive(Debug)]
enum PartialBlock {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        json: String,
    },
}

/// Folds an Anthropic SSE stream into content blocks and a stop reason.
///
/// Three things this exists to get right, each measured against a live stream:
///
/// 1. **`content_block_start` for a `tool_use` carries `input: {}`** — empty,
///    never the arguments. The arguments arrive only as `input_json_delta`
///    fragments and must be concatenated in arrival order.
/// 2. **The first fragment is frequently the empty string**, and no fragment is
///    independently parseable (`""`, `{"`, `pa`, `th": ` …). Parse once, at the
///    end, over the concatenation.
/// 3. **`stop_reason` arrives on `message_delta`, not `message_stop`.** The
///    pre-tool parser only inspected `message_stop`, so a turn that ended in
///    order to call a tool looked exactly like one that finished talking.
///
/// Keyed by the frame's `index` so blocks materialize in wire order regardless
/// of frame interleaving.
#[derive(Debug, Default)]
struct StreamAccumulator {
    open: std::collections::BTreeMap<u64, PartialBlock>,
    stop_reason: Option<String>,
}

impl StreamAccumulator {
    fn observe(&mut self, v: &serde_json::Value) {
        let frame = v.get("type").and_then(|t| t.as_str()).unwrap_or_default();
        let index = v.get("index").and_then(|i| i.as_u64());

        match frame {
            "content_block_start" => {
                let (Some(index), Some(block)) = (index, v.get("content_block")) else {
                    return;
                };
                let kind = block.get("type").and_then(|t| t.as_str()).unwrap_or_default();
                match kind {
                    "text" => {
                        let seed = block.get("text").and_then(|t| t.as_str()).unwrap_or_default();
                        self.open.insert(index, PartialBlock::Text(seed.to_string()));
                    }
                    "tool_use" => {
                        self.open.insert(
                            index,
                            PartialBlock::ToolUse {
                                id: block
                                    .get("id")
                                    .and_then(|i| i.as_str())
                                    .unwrap_or_default()
                                    .to_string(),
                                name: block
                                    .get("name")
                                    .and_then(|n| n.as_str())
                                    .unwrap_or_default()
                                    .to_string(),
                                // NOT block["input"] — it is always `{}` here.
                                json: String::new(),
                            },
                        );
                    }
                    // A block kind this engine does not model (thinking,
                    // server_tool_use, …). Deliberately not opened: an unknown
                    // block we cannot faithfully echo back is worse than one we
                    // never claim to have.
                    _ => {}
                }
            }

            "content_block_delta" => {
                let (Some(index), Some(delta)) = (index, v.get("delta")) else {
                    return;
                };
                let kind = delta.get("type").and_then(|t| t.as_str()).unwrap_or_default();
                match (self.open.get_mut(&index), kind) {
                    (Some(PartialBlock::Text(buf)), "text_delta") => {
                        if let Some(t) = delta.get("text").and_then(|t| t.as_str()) {
                            buf.push_str(t);
                        }
                    }
                    (Some(PartialBlock::ToolUse { json, .. }), "input_json_delta") => {
                        if let Some(fragment) = delta.get("partial_json").and_then(|p| p.as_str()) {
                            json.push_str(fragment);
                        }
                    }
                    _ => {}
                }
            }

            "message_delta" => {
                if let Some(reason) = v
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|r| r.as_str())
                {
                    self.stop_reason = Some(reason.to_string());
                }
            }

            _ => {}
        }
    }

    /// The turn's visible text — every `Text` block concatenated. This is what
    /// the transient sink has always carried (I-4: a snapshot, not an
    /// increment), unchanged by the arrival of tool blocks.
    fn text(&self) -> String {
        self.open
            .values()
            .filter_map(|b| match b {
                PartialBlock::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Materialize in wire order.
    ///
    /// A `tool_use` whose accumulated fragments do not parse is **dropped**, not
    /// salvaged. That happens when `max_tokens` cuts the stream mid-arguments,
    /// and a partially-parsed argument set is the dangerous outcome: a
    /// `read_file` with a truncated path is a read of the wrong file, and
    /// `guard::resolve` will allow it if the truncation is still inside root.
    /// The turn's `stop_reason` is already `max_tokens`, so the caller can tell
    /// what happened without being handed a fabricated call.
    fn blocks(&self) -> Vec<ContentBlock> {
        self.open
            .values()
            .filter_map(|b| match b {
                PartialBlock::Text(t) => Some(ContentBlock::Text(t.clone())),
                PartialBlock::ToolUse { id, name, json } => serde_json::from_str(json)
                    .ok()
                    .map(|input| ContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input,
                    }),
            })
            .collect()
    }
}

fn role_str(role: ProviderRole) -> &'static str {
    match role {
        ProviderRole::User => "user",
        ProviderRole::Assistant => "assistant",
    }
}

/// Encode one block in Anthropic's wire shape.
///
/// `ToolResult` emits exactly `{type, tool_use_id, content, is_error}`. It must
/// never gain a `status` key — measured, that is a 400 ("Extra inputs are not
/// permitted"). See `projection::ContentBlock`'s doc for where the typed status
/// actually lives.
fn block_json(block: &ContentBlock) -> serde_json::Value {
    match block {
        ContentBlock::Text(text) => serde_json::json!({ "type": "text", "text": text }),
        ContentBlock::ToolUse { id, name, input } => serde_json::json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": input,
        }),
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => serde_json::json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
        }),
    }
}

/// Materialize a block list into the `content` field.
///
/// A lone `Text` block becomes a bare JSON string rather than a one-element
/// array. Both are legal, and choosing the string is what makes this migration
/// inert: every turn that uses no tools produces the exact bytes the
/// pre-migration build produced, which the request goldens pin.
fn flatten_content(blocks: &[ContentBlock]) -> serde_json::Value {
    match blocks {
        [ContentBlock::Text(text)] => serde_json::json!(text),
        _ => serde_json::Value::Array(blocks.iter().map(block_json).collect()),
    }
}

/// Build the `/v1/messages` request body for one turn.
///
/// Extracted from `AnthropicProvider::run_turn` so the body is inspectable
/// without a network call. It is the seam the golden tests pin: the block-list
/// migration has to leave this function's output byte-identical for a
/// text-only turn, and there is no way to check that while the body is built
/// inline and handed straight to `reqwest`.
fn build_request_body(model: &str, turn: &ModelTurn) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = turn
        .messages
        .iter()
        .map(|m| {
            serde_json::json!({
                "role": role_str(m.role),
                "content": flatten_content(&m.content),
            })
        })
        .collect();

    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": messages,
        "stream": true,
    });
    if let Some(system) = &turn.system {
        body["system"] = serde_json::json!(system);
    }
    body
}

impl AnthropicProvider {
    /// Fold the accumulator into an outcome. One place so the four exit paths
    /// through the stream loop cannot disagree about the shape.
    fn outcome(&self, acc: StreamAccumulator, stopped: bool) -> TurnOutcome {
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
impl ModelProvider for AnthropicProvider {
    async fn run_turn(
        &self,
        turn: ModelTurn,
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

        let body = build_request_body(&self.model, &turn);

        let client = reqwest::Client::new();
        let response = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
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
        let mut acc = StreamAccumulator::default();

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

                                let Some(v) = parse_sse_data_line(&line) else { continue };

                                let had_text = extract_text_delta(&v).is_some();
                                acc.observe(&v);

                                if had_text {
                                    // I-4: snapshot conversion happens here, at
                                    // the source — the sink never sees a raw
                                    // delta, only the whole string so far.
                                    // Ignored send error: no receiver just
                                    // means nobody is painting this turn.
                                    let _ = sink.send(acc.text()).await;
                                }

                                if is_message_stop(&v) {
                                    return Ok(self.outcome(acc, false));
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

/// No credential composed at all. Always fails — the Explorer keeps working
/// keyless (routes that don't need a model are unaffected); a caller only
/// discovers there's no model wired in when it actually tries to run a
/// turn, not at boot.
pub struct KeylessProvider;

#[async_trait::async_trait]
impl ModelProvider for KeylessProvider {
    async fn run_turn(
        &self,
        _turn: ModelTurn,
        _sink: mpsc::Sender<String>,
        _cancel: watch::Receiver<bool>,
    ) -> Result<TurnOutcome, ProviderError> {
        Err(ProviderError::NoApiKey)
    }
}

/// A scripted turn's block list: one `Text` block, or none when the script
/// produced nothing. Mirrors what the accumulator would build from a text-only
/// stream, so a test written against the double exercises the same shape the
/// real provider returns.
fn scripted_blocks(text: &str) -> Vec<ContentBlock> {
    if text.is_empty() {
        Vec::new()
    } else {
        vec![ContentBlock::Text(text.to_string())]
    }
}

/// Test double: emits a scripted sequence of cumulative snapshots, then
/// stops. `fragments` are the FULL text-so-far at each step (e.g. `["The",
/// "The engine", "The engine hums"]`), not deltas — matching exactly what a
/// real provider's sink receives, so a test written against this double
/// exercises the same contract Task 11's loop consumes from either
/// provider.
pub struct ScriptedProvider {
    fragments: Vec<&'static str>,
    pause: Option<Duration>,
}

impl ScriptedProvider {
    pub fn new(fragments: Vec<&'static str>) -> Self {
        ScriptedProvider {
            fragments,
            pause: None,
        }
    }

    /// Sleep this long between fragments so a test can flip `cancel` mid-
    /// script and observe the interrupt take effect before the script
    /// finishes (Task 11's interrupt test needs exactly this knob).
    pub fn with_pause(mut self, pause: Duration) -> Self {
        self.pause = Some(pause);
        self
    }
}

#[async_trait::async_trait]
impl ModelProvider for ScriptedProvider {
    async fn run_turn(
        &self,
        _turn: ModelTurn,
        sink: mpsc::Sender<String>,
        cancel: watch::Receiver<bool>,
    ) -> Result<TurnOutcome, ProviderError> {
        let mut last_sent = String::new();

        for (i, fragment) in self.fragments.iter().enumerate() {
            if *cancel.borrow() {
                return Ok(TurnOutcome {
                    blocks: scripted_blocks(&last_sent),
                    text: last_sent,
                    stop_reason: None,
                    stopped: true,
                    model: "scripted".to_string(),
                });
            }

            let _ = sink.send(fragment.to_string()).await;
            last_sent = fragment.to_string();

            let is_last = i + 1 == self.fragments.len();
            if !is_last {
                if let Some(pause) = self.pause {
                    tokio::time::sleep(pause).await;
                }
            }
        }

        Ok(TurnOutcome {
            blocks: scripted_blocks(&last_sent),
            text: last_sent,
            stop_reason: Some("end_turn".to_string()),
            stopped: false,
            model: "scripted".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::ProviderMessage;

    fn empty_turn() -> ModelTurn {
        ModelTurn {
            system: None,
            messages: vec![],
        }
    }

    // --- ScriptedProvider ---

    #[tokio::test]
    async fn scripted_provider_emits_cumulative_snapshots_in_order() {
        let provider = ScriptedProvider::new(vec!["The", "The engine", "The engine hums"]);
        let (tx, mut rx) = mpsc::channel(8);
        let (_cancel_tx, cancel_rx) = watch::channel(false);

        let outcome = provider.run_turn(empty_turn(), tx, cancel_rx).await.unwrap();

        assert_eq!(outcome.text, "The engine hums");
        assert!(!outcome.stopped);
        assert_eq!(outcome.model, "scripted");

        let mut received = Vec::new();
        while let Ok(s) = rx.try_recv() {
            received.push(s);
        }
        assert_eq!(received, vec!["The", "The engine", "The engine hums"]);
    }

    #[tokio::test(start_paused = true)]
    async fn scripted_provider_cancel_mid_script_stops_with_last_sent_snapshot() {
        let provider =
            ScriptedProvider::new(vec!["a", "ab", "abc"]).with_pause(Duration::from_millis(50));
        let (tx, mut rx) = mpsc::channel(8);
        let (cancel_tx, cancel_rx) = watch::channel(false);

        let handle = tokio::spawn(async move { provider.run_turn(empty_turn(), tx, cancel_rx).await });

        assert_eq!(rx.recv().await.unwrap(), "a");
        cancel_tx.send(true).unwrap();
        tokio::time::advance(Duration::from_millis(60)).await;

        let outcome = handle.await.unwrap().unwrap();
        assert!(outcome.stopped);
        assert_eq!(outcome.text, "a");
    }

    #[tokio::test]
    async fn scripted_provider_single_fragment_completes_unstopped() {
        let provider = ScriptedProvider::new(vec!["done"]);
        let (tx, _rx) = mpsc::channel(8);
        let (_cancel_tx, cancel_rx) = watch::channel(false);

        let outcome = provider.run_turn(empty_turn(), tx, cancel_rx).await.unwrap();

        assert_eq!(outcome.text, "done");
        assert!(!outcome.stopped);
    }

    // --- SSE parsing (pure, no network) ---
    //
    // Fixture lines below are copied shapes from a real Anthropic
    // `stream: true` response to `/v1/messages`.

    #[test]
    fn parse_sse_data_line_ignores_event_lines() {
        assert_eq!(parse_sse_data_line("event: content_block_delta"), None);
    }

    #[test]
    fn parse_sse_data_line_ignores_blank_lines() {
        assert_eq!(parse_sse_data_line(""), None);
    }

    #[test]
    fn parse_sse_data_line_ignores_done_sentinel() {
        assert_eq!(parse_sse_data_line("data: [DONE]"), None);
    }

    #[test]
    fn parse_sse_data_line_parses_content_block_delta() {
        let line =
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let v = parse_sse_data_line(line).expect("valid data line parses");
        assert_eq!(v["type"], "content_block_delta");
        assert_eq!(v["delta"]["text"], "Hello");
    }

    #[test]
    fn parse_sse_data_line_parses_message_start() {
        let line = r#"data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}"#;
        let v = parse_sse_data_line(line).expect("valid data line parses");
        assert_eq!(v["type"], "message_start");
    }

    #[test]
    fn extract_text_delta_pulls_text_from_content_block_delta() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
        )
        .unwrap();
        assert_eq!(extract_text_delta(&v), Some("Hello"));
    }

    #[test]
    fn extract_text_delta_ignores_ping() {
        let v: serde_json::Value = serde_json::from_str(r#"{"type": "ping"}"#).unwrap();
        assert_eq!(extract_text_delta(&v), None);
    }

    #[test]
    fn extract_text_delta_ignores_content_block_stop() {
        let v: serde_json::Value =
            serde_json::from_str(r#"{"type":"content_block_stop","index":0}"#).unwrap();
        assert_eq!(extract_text_delta(&v), None);
    }

    #[test]
    fn extract_text_delta_ignores_message_delta_stop_reason() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}"#,
        )
        .unwrap();
        assert_eq!(extract_text_delta(&v), None);
    }

    #[test]
    fn is_message_stop_detects_the_terminal_frame() {
        let v: serde_json::Value = serde_json::from_str(r#"{"type":"message_stop"}"#).unwrap();
        assert!(is_message_stop(&v));

        let other: serde_json::Value = serde_json::from_str(r#"{"type":"ping"}"#).unwrap();
        assert!(!is_message_stop(&other));
    }

    #[test]
    fn a_full_fixture_stream_accumulates_to_the_concatenated_text() {
        // Simulates what run_turn's line-buffer loop does, without any
        // network: feed every line of a realistic SSE stream through the
        // two pure functions and check the accumulated result.
        let lines = [
            "event: message_start",
            r#"data: {"type":"message_start","message":{"id":"msg_01","role":"assistant","content":[]}}"#,
            "",
            "event: content_block_start",
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            "",
            "event: ping",
            r#"data: {"type": "ping"}"#,
            "",
            "event: content_block_delta",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            "",
            "event: content_block_delta",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}"#,
            "",
            "event: content_block_stop",
            r#"data: {"type":"content_block_stop","index":0}"#,
            "",
            "event: message_delta",
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}"#,
            "",
            "event: message_stop",
            r#"data: {"type":"message_stop"}"#,
        ];

        let mut accumulated = String::new();
        let mut saw_stop = false;
        for line in lines {
            if let Some(v) = parse_sse_data_line(line) {
                if let Some(delta) = extract_text_delta(&v) {
                    accumulated.push_str(delta);
                }
                if is_message_stop(&v) {
                    saw_stop = true;
                }
            }
        }

        assert_eq!(accumulated, "Hello!");
        assert!(saw_stop);
    }

    // --- streaming a tool call ---
    //
    // Every frame below is copied from a real `claude-sonnet-5` stream captured
    // against the live API, including the fragment boundaries. They are ugly on
    // purpose: the first `partial_json` fragment is the empty string, and no
    // fragment is valid JSON on its own.

    fn frames(lines: &[&str]) -> StreamAccumulator {
        let mut acc = StreamAccumulator::default();
        for line in lines {
            if let Some(v) = parse_sse_data_line(line) {
                acc.observe(&v);
            }
        }
        acc
    }

    #[test]
    fn a_tool_use_stream_accumulates_partial_json_into_one_parsed_input() {
        let acc = frames(&[
            r#"data: {"type":"message_start","message":{"id":"msg_01","role":"assistant","content":[]}}"#,
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01Na9x","name":"read_file","input":{}}}"#,
            r#"data: {"type":"ping"}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"pa"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"th\": "}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"x.md\""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":", \"of"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"fset\""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":": 40}"}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":15}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);

        assert_eq!(acc.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(
            acc.blocks(),
            vec![ContentBlock::ToolUse {
                id: "toolu_01Na9x".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({ "path": "x.md", "offset": 40 }),
            }]
        );
    }

    #[test]
    fn the_tool_use_start_frames_empty_input_is_not_mistaken_for_the_arguments() {
        // Measured: `content_block_start` carries `input: {}` — empty, never the
        // real arguments. Reading it instead of the deltas yields a tool call
        // with no parameters, which for `read_file` is a read of nothing.
        let acc = frames(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"read_file","input":{}}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"a\"}"}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
        ]);

        let blocks = acc.blocks();
        match &blocks[0] {
            ContentBlock::ToolUse { input, .. } => {
                assert_eq!(*input, serde_json::json!({ "path": "a" }));
            }
            other => panic!("expected a tool_use block, got {other:?}"),
        }
    }

    #[test]
    fn stop_reason_is_read_from_message_delta_not_message_stop() {
        // The loop's termination signal lives on `message_delta`. `message_stop`
        // carries nothing, and the pre-tool parser only ever looked at that —
        // so a turn that ended to call a tool was indistinguishable from one
        // that finished talking.
        let ended = frames(&[
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            r#"data: {"type":"message_stop"}"#,
        ]);
        assert_eq!(ended.stop_reason.as_deref(), Some("end_turn"));

        let truncated = frames(&[
            r#"data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"}}"#,
        ]);
        assert_eq!(truncated.stop_reason.as_deref(), Some("max_tokens"));
    }

    #[test]
    fn a_text_only_stream_still_yields_exactly_one_text_block() {
        let acc = frames(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
        ]);

        assert_eq!(acc.text(), "Hello!");
        assert_eq!(acc.blocks(), vec![ContentBlock::Text("Hello!".to_string())]);
    }

    #[test]
    fn text_and_a_tool_call_in_one_turn_keep_their_order() {
        let acc = frames(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"reading it"}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"read_file","input":{}}}"#,
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"a\"}"}}"#,
            r#"data: {"type":"content_block_stop","index":1}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
        ]);

        assert_eq!(acc.text(), "reading it");
        assert_eq!(acc.blocks().len(), 2);
        assert!(matches!(acc.blocks()[0], ContentBlock::Text(_)));
        assert!(matches!(acc.blocks()[1], ContentBlock::ToolUse { .. }));
    }

    #[test]
    fn unparseable_tool_input_yields_no_tool_block_rather_than_a_fabricated_one() {
        // A `max_tokens` cut mid-`input_json_delta` leaves fragments that do not
        // parse. Executing a tool with salvaged arguments is worse than not
        // executing it: a `read_file` with a truncated path reads the wrong
        // file, and the guard will happily allow it if the truncation is still
        // inside root. Drop the block; the turn's `stop_reason` already says
        // `max_tokens`, so the loop can see what happened.
        let acc = frames(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"read_file","input":{}}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\": \"oper"}}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"}}"#,
        ]);

        assert_eq!(acc.stop_reason.as_deref(), Some("max_tokens"));
        assert!(
            acc.blocks().is_empty(),
            "a half-parsed tool call must not become a block: {:?}",
            acc.blocks()
        );
    }

    #[tokio::test]
    async fn a_turn_outcome_carries_the_blocks_and_the_stop_reason_not_just_text() {
        // `text` alone cannot express a tool call, so the loop had no way to
        // learn the model asked for one — it only ever saw a string. These two
        // fields are what make a second round decidable.
        let provider = ScriptedProvider::new(vec!["The", "The engine hums"]);
        let (tx, _rx) = mpsc::channel(8);
        let (_cancel_tx, cancel_rx) = watch::channel(false);

        let outcome = provider.run_turn(empty_turn(), tx, cancel_rx).await.unwrap();

        assert_eq!(outcome.text, "The engine hums");
        assert_eq!(outcome.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(
            outcome.blocks,
            vec![ContentBlock::Text("The engine hums".to_string())]
        );
    }

    // --- KeylessProvider ---

    #[tokio::test]
    async fn keyless_provider_always_returns_no_api_key() {
        let provider = KeylessProvider;
        let (tx, _rx) = mpsc::channel(8);
        let (_cancel_tx, cancel_rx) = watch::channel(false);

        let err = provider.run_turn(empty_turn(), tx, cancel_rx).await.unwrap_err();

        assert_eq!(err, ProviderError::NoApiKey);
    }

    // --- the request body, pinned ---
    //
    // These two goldens are captured against the PRE-migration build and must
    // not be regenerated afterwards. Their whole job is to let the block-list
    // migration prove it is inert rather than assert it: a golden written after
    // the change would pass by construction and prove nothing, which is the
    // "regression test that passes identically with the bug present" defect this
    // repo has already shipped once.

    #[test]
    fn request_body_for_a_text_only_turn_is_the_pinned_shape() {
        let turn = ModelTurn {
            system: Some("# boot".to_string()),
            messages: vec![
                ProviderMessage {
                    role: ProviderRole::User,
                    content: vec![ContentBlock::Text("hi".to_string())],
                },
                ProviderMessage {
                    role: ProviderRole::Assistant,
                    content: vec![ContentBlock::Text("hello".to_string())],
                },
                ProviderMessage {
                    role: ProviderRole::User,
                    content: vec![ContentBlock::Text("bye".to_string())],
                },
            ],
        };

        assert_eq!(
            build_request_body("claude-sonnet-5", &turn),
            serde_json::json!({
                "model": "claude-sonnet-5",
                "max_tokens": 4096,
                "stream": true,
                "system": "# boot",
                "messages": [
                    { "role": "user", "content": "hi" },
                    { "role": "assistant", "content": "hello" },
                    { "role": "user", "content": "bye" },
                ],
            })
        );
    }

    #[test]
    fn request_body_omits_system_entirely_when_no_boot_event_stands() {
        // Not `"system": null` — the key is absent. A bare clone declares no
        // boot file, and this is the shape that has been in production since
        // the boot channel shipped.
        let turn = ModelTurn {
            system: None,
            messages: vec![ProviderMessage {
                role: ProviderRole::User,
                content: vec![ContentBlock::Text("hi".to_string())],
            }],
        };

        let body = build_request_body("claude-sonnet-5", &turn);

        assert!(body.get("system").is_none(), "system must be absent, got: {body}");
        assert_eq!(
            body,
            serde_json::json!({
                "model": "claude-sonnet-5",
                "max_tokens": 4096,
                "stream": true,
                "messages": [{ "role": "user", "content": "hi" }],
            })
        );
    }

    // --- block flattening (P-4's materialization edge) ---

    fn text_msg(role: ProviderRole, s: &str) -> ProviderMessage {
        ProviderMessage {
            role,
            content: vec![ContentBlock::Text(s.to_string())],
        }
    }

    #[test]
    fn a_lone_text_block_flattens_to_a_bare_string_not_a_block_list() {
        // The wire accepts both forms. Choosing the string keeps every
        // text-only turn byte-identical to the pre-migration build, so the
        // goldens above stay valid and the migration's blast radius is only
        // turns that actually use tools.
        let turn = ModelTurn {
            system: None,
            messages: vec![text_msg(ProviderRole::User, "hi")],
        };

        assert_eq!(
            build_request_body("m", &turn)["messages"][0]["content"],
            serde_json::json!("hi")
        );
    }

    #[test]
    fn a_message_carrying_a_tool_use_flattens_to_a_block_list() {
        let turn = ModelTurn {
            system: None,
            messages: vec![ProviderMessage {
                role: ProviderRole::Assistant,
                content: vec![
                    ContentBlock::Text("reading it".to_string()),
                    ContentBlock::ToolUse {
                        id: "toolu_01AAA".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({ "path": "x.md" }),
                    },
                ],
            }],
        };

        assert_eq!(
            build_request_body("m", &turn)["messages"][0]["content"],
            serde_json::json!([
                { "type": "text", "text": "reading it" },
                {
                    "type": "tool_use",
                    "id": "toolu_01AAA",
                    "name": "read_file",
                    "input": { "path": "x.md" },
                },
            ])
        );
    }

    #[test]
    fn a_tool_result_block_never_emits_the_typed_status_to_the_wire() {
        // MEASURED against the live API: a `status` key inside a `tool_result`
        // block returns 400 `invalid_request_error` — "Extra inputs are not
        // permitted". The typed status is sovereign in the LOG; at the wire it
        // survives only as `is_error` plus whatever the projection rendered
        // into `content`. That is P-4's "as far as the provider channel
        // allows", and this test is what stops someone re-adding the field.
        let turn = ModelTurn {
            system: None,
            messages: vec![ProviderMessage {
                role: ProviderRole::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "toolu_01AAA".to_string(),
                    content: "denied_credential: refused; try a path under operators/".to_string(),
                    is_error: true,
                }],
            }],
        };

        let block = &build_request_body("m", &turn)["messages"][0]["content"][0];

        assert!(
            block.get("status").is_none(),
            "a status key on the wire is a 400: {block}"
        );
        assert_eq!(
            *block,
            serde_json::json!({
                "type": "tool_result",
                "tool_use_id": "toolu_01AAA",
                "content": "denied_credential: refused; try a path under operators/",
                "is_error": true,
            })
        );
    }

    // --- AnthropicProvider ---

    #[test]
    fn anthropic_provider_debug_redacts_api_key() {
        let provider = AnthropicProvider {
            model: "claude-3-5-sonnet-20241022".to_string(),
            api_key: "sk-ant-super-secret-do-not-print".to_string(),
        };
        let rendered = format!("{provider:?}");
        assert!(!rendered.contains("sk-ant-super-secret-do-not-print"));
        assert!(rendered.contains("<redacted>"));
        assert!(rendered.contains("claude-3-5-sonnet-20241022"));
    }

    #[tokio::test]
    #[ignore = "hits the real Anthropic API; run by hand with ANTHROPIC_API_KEY set"]
    async fn anthropic_live_turn() {
        let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") else {
            eprintln!("skipping anthropic_live_turn: ANTHROPIC_API_KEY not set");
            return;
        };

        let provider = AnthropicProvider {
            model: "claude-3-5-haiku-20241022".to_string(),
            api_key,
        };
        let turn = ModelTurn {
            system: None,
            messages: vec![ProviderMessage {
                role: ProviderRole::User,
                content: vec![ContentBlock::Text("Reply with exactly one word: hello".to_string())],
            }],
        };
        let (tx, mut rx) = mpsc::channel(32);
        let (_cancel_tx, cancel_rx) = watch::channel(false);

        let drain = tokio::spawn(async move { while rx.recv().await.is_some() {} });

        let outcome = provider
            .run_turn(turn, tx, cancel_rx)
            .await
            .expect("live call should succeed with a valid key");

        drain.await.unwrap();

        assert!(!outcome.text.is_empty());
        assert!(!outcome.stopped);
    }
}
