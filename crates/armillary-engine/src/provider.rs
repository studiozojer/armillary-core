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
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch};

/// The per-response output ceiling.
///
/// Was 4096, which was a live defect rather than a conservative default.
/// `max_tokens` caps **thinking plus response text together**, and this engine
/// sends no `thinking` parameter — on `claude-sonnet-5`, omitting it runs
/// *adaptive thinking*. So every session has been sharing one 4096-token
/// ceiling between the model's reasoning and its answer, and a turn that
/// thought hard arrived truncated or empty. Nothing observed it: the stream
/// parser did not read `stop_reason`, so a `max_tokens` cut was recorded as an
/// ordinary completed turn.
///
/// 64000 is the streaming-request default this codebase's model family
/// documents. It is a ceiling, not a reservation — an ordinary turn costs what
/// it costs.
///
/// **Deliberately NOT changed here: the `thinking` parameter.** Making it
/// explicit would be honest, but its meaning is model-dependent — on
/// `claude-sonnet-5` omitting it and sending `{"type":"adaptive"}` are
/// identical, while on `claude-haiku-4-5` the first means no thinking and the
/// second means thinking. `--model` is configurable, so declaring it is a
/// behaviour change for some workspaces and a no-op for others. That is a call
/// for the workspace owner, not a fix to slip into a truncation patch.
///
/// **Thinking blocks are captured opaquely and replayed verbatim** — the
/// documented contract, honored rather than leaned on (the 2026-08-07 probe
/// measured the stripped replay *tolerated* on `claude-sonnet-5`; tolerance is
/// not a guarantee, and always-thinking families may refuse the shape). One
/// deliberate exception: a thinking block cut before its `signature_delta` is
/// dropped at materialization — unsigned, it is unreplayable.
pub(crate) const MAX_TOKENS: u32 = 64_000;

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

/// A constraint on what the model may do this round, dialect-neutral: each
/// provider projects it to its own wire shape. One variant today — the loop's
/// at-bound "answer in prose" — and adding one later forces every projection
/// to choose, which is the point of the enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoice {
    ForceText,
}

/// One request to the provider: what the model sees, plus what it may do.
///
/// `turn` is the projection's output and nothing else — P-4's flattened shape.
/// `tools` and `tool_choice` are **not** projections of the log; the loop
/// attaches them per round, which is why they live beside the turn rather than
/// inside it.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnRequest {
    pub turn: ModelTurn,
    /// Sent only when non-empty, and dialect-neutral: each provider projects
    /// these to its own wire shape in its request builder. Tool definitions
    /// render at the very front of the prompt, so a set that changes between
    /// requests invalidates the whole cached prefix — keep the order fixed.
    pub tools: Vec<crate::tools::ToolDef>,
    /// `ToolChoice::ForceText` at a bound to force prose. Dialect-neutral —
    /// the Anthropic projection is `{"type":"none"}`, measured accepted on
    /// `claude-sonnet-5` with adaptive thinking on, which is this engine's
    /// default.
    pub tool_choice: Option<ToolChoice>,
}

impl TurnRequest {
    /// A turn with no tools offered — the pre-tool shape, and what the request
    /// goldens pin.
    pub fn bare(turn: ModelTurn) -> Self {
        TurnRequest {
            turn,
            tools: Vec::new(),
            tool_choice: None,
        }
    }
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
        req: TurnRequest,
        sink: mpsc::Sender<String>,
        cancel: watch::Receiver<bool>,
    ) -> Result<TurnOutcome, ProviderError>;

    /// A redaction-safe identity for this provider — `"anthropic:<model>"`,
    /// `"opencode-zen:<slug>"`, `"keyless"`. Exists so selection is
    /// assertable without a network call and without any impl growing a
    /// `Debug` that could print a key. Never include the credential.
    fn describe(&self) -> String {
        "unknown".to_string()
    }
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
pub(crate) fn parse_sse_data_line(line: &str) -> Option<serde_json::Value> {
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
    Thinking {
        thinking: String,
        signature: String,
    },
    Redacted {
        data: String,
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
                    "thinking" => {
                        let seed = block
                            .get("thinking")
                            .and_then(|t| t.as_str())
                            .unwrap_or_default();
                        self.open.insert(
                            index,
                            PartialBlock::Thinking {
                                thinking: seed.to_string(),
                                signature: String::new(),
                            },
                        );
                    }
                    // Arrives complete in the start frame — no deltas follow.
                    "redacted_thinking" => {
                        let data = block
                            .get("data")
                            .and_then(|d| d.as_str())
                            .unwrap_or_default();
                        self.open.insert(index, PartialBlock::Redacted { data: data.to_string() });
                    }
                    // A block kind this engine does not model (server_tool_use,
                    // …). Deliberately not opened: an unknown block we cannot
                    // faithfully echo back is worse than one we never claim to
                    // have.
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
                    (Some(PartialBlock::Thinking { thinking, .. }), "thinking_delta") => {
                        if let Some(t) = delta.get("thinking").and_then(|t| t.as_str()) {
                            thinking.push_str(t);
                        }
                    }
                    (Some(PartialBlock::Thinking { signature, .. }), "signature_delta") => {
                        if let Some(s) = delta.get("signature").and_then(|s| s.as_str()) {
                            signature.push_str(s);
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
    /// Two failures that look alike and are not:
    ///
    /// - **No arguments.** A tool whose schema takes none streams either no
    ///   `input_json_delta` frames at all, or exactly one whose `partial_json`
    ///   is the empty string. Both accumulate to `""`, which does not parse.
    ///   That is a complete call with an empty input, and dropping it loses a
    ///   perfectly good tool use — observed live against `get_composition`
    ///   before this distinction existed.
    /// - **Truncated arguments.** A non-empty accumulation that does not parse
    ///   means `max_tokens` cut the stream mid-JSON. That call is **dropped**,
    ///   never salvaged: a `read_file` with a half-written path is a read of
    ///   the wrong file, and `guard::resolve` will allow it if the truncation
    ///   happens to land inside root. The turn's `stop_reason` already says
    ///   `max_tokens`, so the caller learns what happened without being handed
    ///   a fabricated call.
    fn blocks(&self) -> Vec<ContentBlock> {
        self.open
            .values()
            .filter_map(|b| match b {
                PartialBlock::Text(t) => Some(ContentBlock::Text(t.clone())),
                PartialBlock::ToolUse { id, name, json } => {
                    let input = if json.trim().is_empty() {
                        Some(serde_json::Value::Object(Default::default()))
                    } else {
                        serde_json::from_str(json).ok()
                    };
                    input.map(|input| ContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input,
                    })
                }
                PartialBlock::Thinking { thinking, signature } => {
                    // Cut before signature_delta: unreplayable, and an
                    // unsigned block sent back is a tamper-shaped 400 risk.
                    // Dropped, never salvaged — same rule as the truncated
                    // tool-call JSON above.
                    (!signature.is_empty()).then(|| ContentBlock::Thinking {
                        thinking: thinking.clone(),
                        signature: signature.clone(),
                    })
                }
                PartialBlock::Redacted { data } => {
                    Some(ContentBlock::RedactedThinking { data: data.clone() })
                }
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
pub(crate) fn block_json(block: &ContentBlock) -> serde_json::Value {
    match block {
        ContentBlock::Text(text) => serde_json::json!({ "type": "text", "text": text }),
        ContentBlock::Thinking { thinking, signature } => serde_json::json!({
            "type": "thinking",
            "thinking": thinking,
            "signature": signature,
        }),
        ContentBlock::RedactedThinking { data } => serde_json::json!({
            "type": "redacted_thinking",
            "data": data,
        }),
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
fn build_request_body(model: &str, req: &TurnRequest) -> serde_json::Value {
    let turn = &req.turn;
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
    // Omitted entirely when empty, not sent as `[]`. That is what keeps a
    // no-tools turn byte-identical to the pre-tool build, which the goldens
    // pin — the claim is about the flattening, not about the engine never
    // offering tools.
    if !req.tools.is_empty() {
        let tools: Vec<serde_json::Value> =
            req.tools.iter().map(|d| d.anthropic_definition()).collect();
        body["tools"] = serde_json::json!(tools);
    }
    if let Some(choice) = &req.tool_choice {
        body["tool_choice"] = match choice {
            ToolChoice::ForceText => serde_json::json!({ "type": "none" }),
        };
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

        let body = build_request_body(&self.model, &req);

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

    fn describe(&self) -> String {
        format!("anthropic:{}", self.model)
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
        _req: TurnRequest,
        _sink: mpsc::Sender<String>,
        _cancel: watch::Receiver<bool>,
    ) -> Result<TurnOutcome, ProviderError> {
        Err(ProviderError::NoApiKey)
    }

    fn describe(&self) -> String {
        "keyless".to_string()
    }
}

/// Which provider a model string selects, and with what residue. Pure so the
/// selection rule is testable without booting a server. Moved here from
/// `main.rs` — `KeyedProviders::provider_for` below is now the only caller
/// that matters at runtime, and it belongs beside the providers it chooses
/// between.
pub enum ProviderChoice {
    Anthropic,
    Zen { slug: String },
}

pub fn choose_provider(model: &str) -> ProviderChoice {
    match model.strip_prefix("zen/") {
        Some(slug) => ProviderChoice::Zen {
            slug: slug.to_string(),
        },
        None => ProviderChoice::Anthropic,
    }
}

/// Maps a model string to the provider that can pilot it.
///
/// This replaces a single `Arc<dyn ModelProvider>` in `AppState` because
/// the model is now a property of the INSTANCE (design decision 1), not of
/// the process — two instances in one engine may want different providers,
/// so the choice cannot be made once at boot. Both providers are plain
/// structs holding a model string and a key, so constructing one per turn
/// costs nothing: there is no client to pool.
pub trait ProviderFor: Send + Sync + 'static {
    fn provider_for(&self, model: &str) -> Arc<dyn ModelProvider>;
}

/// The real one: holds every credential this host resolved at boot, and
/// applies `choose_provider`'s rule per call.
///
/// A model whose provider has no key resolves to `KeylessProvider`, which
/// is what makes design decision 3 (accept at create, fail at the turn)
/// need no new error path — the existing `no_api_key` turn failure is the
/// report, and it names the instance's own model.
pub struct KeyedProviders {
    pub anthropic_key: Option<String>,
    pub zen_key: Option<String>,
}

/// Hand-written to redact both keys. NEVER derive — a derive prints them
/// verbatim the first time this lands in a log line or a panic message.
/// Same discipline as `AnthropicProvider` and `state::ModelConfig`.
impl std::fmt::Debug for KeyedProviders {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyedProviders")
            .field("anthropic_key", &self.anthropic_key.as_ref().map(|_| "<redacted>"))
            .field("zen_key", &self.zen_key.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

impl ProviderFor for KeyedProviders {
    fn provider_for(&self, model: &str) -> Arc<dyn ModelProvider> {
        match choose_provider(model) {
            ProviderChoice::Anthropic => match &self.anthropic_key {
                Some(key) => Arc::new(AnthropicProvider {
                    model: model.to_string(),
                    api_key: key.clone(),
                }),
                None => Arc::new(KeylessProvider),
            },
            ProviderChoice::Zen { slug } => match &self.zen_key {
                Some(key) => Arc::new(crate::provider_openai::OpenAiCompatProvider {
                    base_url: "https://opencode.ai/zen/v1".to_string(),
                    // The BARE slug crosses the wire; the prefixed spelling
                    // stays in the log. core#19's rule, unchanged.
                    model: slug,
                    api_key: key.clone(),
                }),
                None => Arc::new(KeylessProvider),
            },
        }
    }
}

/// One provider for every model — the test seam. Wraps a single provider
/// and ignores the model string, so a `ScriptedProvider` still injects
/// exactly as it did before this trait existed.
pub fn fixed(provider: Arc<dyn ModelProvider>) -> Arc<dyn ProviderFor> {
    struct Fixed(Arc<dyn ModelProvider>);
    impl ProviderFor for Fixed {
        fn provider_for(&self, _model: &str) -> Arc<dyn ModelProvider> {
            self.0.clone()
        }
    }
    Arc::new(Fixed(provider))
}

/// A request shape the Anthropic API rejects with a 400.
///
/// Every variant here was **measured**, not inferred — each corresponds to a
/// probe against the live API that came back `invalid_request_error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireViolation {
    /// A `tool_use` with no `tool_result` immediately after it.
    UnansweredToolUse { id: String },
    /// A `tool_result` whose `tool_use_id` matches no `tool_use` in the
    /// preceding assistant message.
    OrphanToolResult { tool_use_id: String },
    /// `tool_result` blocks must form an uninterrupted *leading* sequence in
    /// the turn that answers a `tool_use`. Trailing text is fine; text before
    /// or between results is not.
    ToolResultsNotLeading,
    /// `text content blocks must be non-empty`.
    EmptyTextBlock,
    /// `content cannot be empty if is_error is true`. Empty content alone is
    /// legal; `is_error` with content is legal; the pair is not.
    EmptyErrorResult { tool_use_id: String },
}

impl std::fmt::Display for WireViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireViolation::UnansweredToolUse { id } => {
                write!(f, "`tool_use` ids were found without `tool_result` blocks immediately after: {id}")
            }
            WireViolation::OrphanToolResult { tool_use_id } => write!(
                f,
                "unexpected `tool_use_id` found in `tool_result` blocks: {tool_use_id}"
            ),
            WireViolation::ToolResultsNotLeading => write!(
                f,
                "`tool_result` blocks must lead the turn, uninterrupted"
            ),
            WireViolation::EmptyTextBlock => write!(f, "text content blocks must be non-empty"),
            WireViolation::EmptyErrorResult { tool_use_id } => write!(
                f,
                "content cannot be empty if `is_error` is true: {tool_use_id}"
            ),
        }
    }
}

/// Check a turn against the request-validity rules the API enforces.
///
/// This is the piece that makes a scripted test mean something. A test double
/// that answers anything will happily answer a message list the real API would
/// refuse — so an assertion like "the stream still projects and takes a turn"
/// passes over a broken projection. That is the same defect as a regression
/// test that passes identically with the bug present, and this repo has shipped
/// one already.
pub fn validate_turn(turn: &ModelTurn) -> Result<(), WireViolation> {
    for (i, message) in turn.messages.iter().enumerate() {
        // Empty text is rejected wherever it appears.
        if message
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Text(t) if t.is_empty()))
        {
            return Err(WireViolation::EmptyTextBlock);
        }

        // Results must lead their turn, uninterrupted: once a non-result block
        // is seen, no further result may appear in this message.
        let mut seen_non_result = false;
        for block in &message.content {
            match block {
                ContentBlock::ToolResult { .. } if seen_non_result => {
                    return Err(WireViolation::ToolResultsNotLeading)
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    if *is_error && content.is_empty() {
                        return Err(WireViolation::EmptyErrorResult {
                            tool_use_id: tool_use_id.clone(),
                        });
                    }
                    // Every result must answer a call in the message before it.
                    let answered = i
                        .checked_sub(1)
                        .and_then(|p| turn.messages.get(p))
                        .is_some_and(|prev| {
                            prev.content.iter().any(|b| {
                                matches!(b, ContentBlock::ToolUse { id, .. } if id == tool_use_id)
                            })
                        });
                    if !answered {
                        return Err(WireViolation::OrphanToolResult {
                            tool_use_id: tool_use_id.clone(),
                        });
                    }
                }
                _ => seen_non_result = true,
            }
        }

        // Every call must be answered by the message immediately after.
        for block in &message.content {
            let ContentBlock::ToolUse { id, .. } = block else {
                continue;
            };
            let answered = turn.messages.get(i + 1).is_some_and(|next| {
                next.content.iter().any(|b| {
                    matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == id)
                })
            });
            if !answered {
                return Err(WireViolation::UnansweredToolUse { id: id.clone() });
            }
        }
    }
    Ok(())
}

/// Wraps another provider and refuses, exactly as the API would, any turn that
/// violates a measured request-validity rule.
///
/// A sibling of `ScriptedProvider` — a test double that ships in the library
/// because the integration tests in `tests/` need it, same as that one.
#[derive(Debug)]
pub struct ValidatingProvider<P> {
    inner: P,
}

impl<P> ValidatingProvider<P> {
    pub fn new(inner: P) -> Self {
        ValidatingProvider { inner }
    }
}

#[async_trait::async_trait]
impl<P: ModelProvider> ModelProvider for ValidatingProvider<P> {
    async fn run_turn(
        &self,
        req: TurnRequest,
        sink: mpsc::Sender<String>,
        cancel: watch::Receiver<bool>,
    ) -> Result<TurnOutcome, ProviderError> {
        if let Err(violation) = validate_turn(&req.turn) {
            // Shaped like the real refusal so a caller's error handling is
            // exercised by the double rather than bypassed.
            return Err(ProviderError::Api {
                status: 400,
                body: format!(
                    r#"{{"type":"error","error":{{"type":"invalid_request_error","message":"{violation}"}}}}"#
                ),
            });
        }
        self.inner.run_turn(req, sink, cancel).await
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
        _req: TurnRequest,
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

    fn empty_turn() -> TurnRequest {
        TurnRequest::bare(ModelTurn {
            system: None,
            messages: vec![],
        })
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
    fn a_no_argument_tool_call_survives_its_single_empty_fragment() {
        // Captured verbatim from a live `claude-sonnet-5` stream calling
        // `get_composition`, which takes no arguments: ONE `input_json_delta`
        // whose `partial_json` is the empty string. The accumulation is
        // therefore `""`, which does not parse — and an earlier version of the
        // drop rule below threw the call away because of it.
        //
        // No arguments and TRUNCATED arguments are different failures and must
        // not share a branch. Every other test here used a tool that takes
        // arguments, so none of them could see this.
        let acc = frames(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01J6MQ","name":"get_composition","input":{},"caller":{"type":"direct"}}}"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
        ]);

        assert_eq!(acc.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(
            acc.blocks(),
            vec![ContentBlock::ToolUse {
                id: "toolu_01J6MQ".to_string(),
                name: "get_composition".to_string(),
                input: serde_json::json!({}),
            }]
        );
    }

    #[test]
    fn a_tool_call_with_no_delta_frames_at_all_is_also_a_call() {
        // The same case one step further: no `input_json_delta` frames at all.
        let acc = frames(&[
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"get_composition","input":{}}}"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
        ]);

        assert_eq!(acc.blocks().len(), 1, "{:?}", acc.blocks());
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

    // --- the validating double ---
    //
    // Each case below is a shape MEASURED against the live API as a 400. A
    // scripted provider validates nothing, so a test asserting "the stream
    // still projects and takes a turn" passes over a projection the real API
    // would refuse. These are what stop that.

    fn user(blocks: Vec<ContentBlock>) -> ProviderMessage {
        ProviderMessage {
            role: ProviderRole::User,
            content: blocks,
        }
    }
    fn assistant(blocks: Vec<ContentBlock>) -> ProviderMessage {
        ProviderMessage {
            role: ProviderRole::Assistant,
            content: blocks,
        }
    }
    fn tool_use(id: &str) -> ContentBlock {
        ContentBlock::ToolUse {
            id: id.to_string(),
            name: "read_file".to_string(),
            input: serde_json::json!({ "path": "x.md" }),
        }
    }
    fn tool_result(id: &str) -> ContentBlock {
        ContentBlock::ToolResult {
            tool_use_id: id.to_string(),
            content: "contents".to_string(),
            is_error: false,
        }
    }
    fn turn_of(messages: Vec<ProviderMessage>) -> ModelTurn {
        ModelTurn {
            system: None,
            messages,
        }
    }

    #[test]
    fn a_well_formed_tool_round_validates() {
        let turn = turn_of(vec![
            user(vec![ContentBlock::text("read x")]),
            assistant(vec![ContentBlock::text("on it"), tool_use("t1")]),
            user(vec![tool_result("t1")]),
        ]);
        assert_eq!(validate_turn(&turn), Ok(()));
    }

    #[test]
    fn an_unanswered_tool_use_is_refused() {
        let turn = turn_of(vec![
            user(vec![ContentBlock::text("read x")]),
            assistant(vec![tool_use("t1")]),
            user(vec![ContentBlock::text("never mind")]),
        ]);
        assert_eq!(
            validate_turn(&turn),
            Err(WireViolation::UnansweredToolUse { id: "t1".into() })
        );
    }

    #[test]
    fn a_tool_result_with_no_preceding_tool_use_is_refused() {
        // This is the shape a boot-time composition pushed as a tool_result
        // would have had. The API rejects it outright.
        let turn = turn_of(vec![user(vec![
            tool_result("t-nonexistent"),
            ContentBlock::text("what repos exist?"),
        ])]);
        assert_eq!(
            validate_turn(&turn),
            Err(WireViolation::OrphanToolResult {
                tool_use_id: "t-nonexistent".into()
            })
        );
    }

    #[test]
    fn tool_results_must_lead_their_turn_uninterrupted() {
        let turn = turn_of(vec![
            user(vec![ContentBlock::text("read x")]),
            assistant(vec![tool_use("t1"), tool_use("t2")]),
            user(vec![
                tool_result("t1"),
                ContentBlock::text("also, hello"),
                tool_result("t2"),
            ]),
        ]);
        assert_eq!(
            validate_turn(&turn),
            Err(WireViolation::ToolResultsNotLeading)
        );
    }

    #[test]
    fn an_empty_text_block_is_refused() {
        let turn = turn_of(vec![
            user(vec![ContentBlock::text("hi")]),
            assistant(vec![ContentBlock::text(""), tool_use("t1")]),
            user(vec![tool_result("t1")]),
        ]);
        assert_eq!(validate_turn(&turn), Err(WireViolation::EmptyTextBlock));
    }

    #[test]
    fn an_error_result_with_empty_content_is_refused() {
        // The combination is the illegal one: empty content alone is fine, and
        // is_error with content is fine. Every error path the loop can take has
        // to render something, which is why "name the recovery action" is a
        // wire requirement and not a style preference.
        let turn = turn_of(vec![
            user(vec![ContentBlock::text("read x")]),
            assistant(vec![tool_use("t1")]),
            user(vec![ContentBlock::ToolResult {
                tool_use_id: "t1".to_string(),
                content: String::new(),
                is_error: true,
            }]),
        ]);
        assert_eq!(
            validate_turn(&turn),
            Err(WireViolation::EmptyErrorResult {
                tool_use_id: "t1".into()
            })
        );
    }

    #[tokio::test]
    async fn the_validating_provider_refuses_rather_than_answering() {
        let provider = ValidatingProvider::new(ScriptedProvider::new(vec!["ok"]));
        let bad = turn_of(vec![
            user(vec![ContentBlock::text("read x")]),
            assistant(vec![tool_use("t1")]),
            user(vec![ContentBlock::text("never mind")]),
        ]);
        let (tx, _rx) = mpsc::channel(8);
        let (_c, cancel_rx) = watch::channel(false);

        let err = provider.run_turn(TurnRequest::bare(bad), tx, cancel_rx).await.unwrap_err();

        match err {
            ProviderError::Api { status, body } => {
                assert_eq!(status, 400);
                assert!(body.contains("t1"), "the violation should name it: {body}");
            }
            other => panic!("expected a 400 like the real API, got {other:?}"),
        }
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

    // --- choose_provider (moved from main.rs) ---

    #[test]
    fn a_zen_prefix_selects_the_compat_provider_and_bare_names_do_not() {
        assert!(matches!(
            choose_provider("zen/kimi-k3"),
            ProviderChoice::Zen { slug } if slug == "kimi-k3"
        ));
        assert!(matches!(
            choose_provider("claude-sonnet-5"),
            ProviderChoice::Anthropic
        ));
        // A slug containing further slashes passes through whole — Zen's
        // catalog naming is not ours to constrain.
        assert!(matches!(
            choose_provider("zen/deepseek/v4-flash"),
            ProviderChoice::Zen { slug } if slug == "deepseek/v4-flash"
        ));
    }

    // --- KeyedProviders::provider_for ---

    #[test]
    fn keyed_providers_selects_by_prefix_and_falls_keyless_without_a_key() {
        let both = KeyedProviders {
            anthropic_key: Some("a".to_string()),
            zen_key: Some("z".to_string()),
        };
        // A bare name is Anthropic; a zen/ prefix crosses the wire as the BARE
        // slug while the prefixed spelling stays in the log (core#19's rule).
        assert_eq!(both.provider_for("claude-sonnet-5").describe(), "anthropic:claude-sonnet-5");
        assert_eq!(both.provider_for("zen/kimi-k3").describe(), "opencode-zen:kimi-k3");

        // Decision 3's posture, and the whole reason it costs no new error
        // path: an unpilotable model resolves to the provider that already
        // fails every turn with no_api_key.
        let neither = KeyedProviders { anthropic_key: None, zen_key: None };
        assert_eq!(neither.provider_for("claude-sonnet-5").describe(), "keyless");
        assert_eq!(neither.provider_for("zen/kimi-k3").describe(), "keyless");

        // Each key covers only its own provider.
        let anthropic_only = KeyedProviders {
            anthropic_key: Some("a".to_string()),
            zen_key: None,
        };
        assert_eq!(anthropic_only.provider_for("zen/kimi-k3").describe(), "keyless");
        assert_eq!(anthropic_only.provider_for("claude-sonnet-5").describe(), "anthropic:claude-sonnet-5");
    }

    #[test]
    fn force_text_projects_to_the_anthropic_none_shape() {
        // The neutral enum must land on the wire as the exact bytes the loop
        // used to hand-build — measured accepted on claude-sonnet-5 with
        // adaptive thinking on.
        let turn = ModelTurn {
            system: None,
            messages: vec![text_msg(ProviderRole::User, "hi")],
        };
        let body = build_request_body(
            "m",
            &TurnRequest { turn, tools: Vec::new(), tool_choice: Some(ToolChoice::ForceText) },
        );

        assert_eq!(body["tool_choice"], serde_json::json!({ "type": "none" }));
    }

    #[test]
    fn a_replayed_turn_carries_its_thinking_blocks_verbatim() {
        let turn = ModelTurn {
            system: None,
            messages: vec![
                text_msg(ProviderRole::User, "hi"),
                ProviderMessage {
                    role: ProviderRole::Assistant,
                    content: vec![
                        ContentBlock::Thinking {
                            thinking: "let me look".to_string(),
                            signature: "sig-1".to_string(),
                        },
                        ContentBlock::Text("checking".to_string()),
                        ContentBlock::ToolUse {
                            id: "tu_1".to_string(),
                            name: "read_file".to_string(),
                            input: serde_json::json!({"path": "a.md"}),
                        },
                    ],
                },
            ],
        };

        assert_eq!(
            build_request_body("m", &TurnRequest::bare(turn))["messages"][1]["content"],
            serde_json::json!([
                {"type": "thinking", "thinking": "let me look", "signature": "sig-1"},
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "a.md"}},
            ])
        );
    }

    // --- streaming thinking blocks ---

    #[test]
    fn thinking_deltas_accumulate_and_materialize_in_wire_order() {
        let mut acc = StreamAccumulator::default();
        for v in [
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "thinking", "thinking": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "let me "}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "check"}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig-abc"}}),
            serde_json::json!({"type": "content_block_start", "index": 1,
                "content_block": {"type": "text", "text": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 1,
                "delta": {"type": "text_delta", "text": "on it"}}),
            serde_json::json!({"type": "content_block_start", "index": 2,
                "content_block": {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {}}}),
            serde_json::json!({"type": "content_block_delta", "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": "{\"path\": \"a.md\"}"}}),
        ] {
            acc.observe(&v);
        }

        assert_eq!(
            acc.blocks(),
            vec![
                ContentBlock::Thinking {
                    thinking: "let me check".to_string(),
                    signature: "sig-abc".to_string(),
                },
                ContentBlock::Text("on it".to_string()),
                ContentBlock::ToolUse {
                    id: "tu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "a.md"}),
                },
            ]
        );
        // Thinking never enters the visible prose.
        assert_eq!(acc.text(), "on it");
    }

    #[test]
    fn a_thinking_block_cut_before_its_signature_is_dropped_not_salvaged() {
        // An unsigned thinking block on replay is a tamper-shaped 400 risk —
        // the same rule blocks() applies to truncated tool-call JSON.
        let mut acc = StreamAccumulator::default();
        for v in [
            serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "thinking", "thinking": ""}}),
            serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "cut mid-"}}),
            serde_json::json!({"type": "content_block_start", "index": 1,
                "content_block": {"type": "text", "text": "still here"}}),
        ] {
            acc.observe(&v);
        }

        assert_eq!(acc.blocks(), vec![ContentBlock::Text("still here".to_string())]);
    }

    #[test]
    fn redacted_thinking_arrives_whole_and_survives_whole() {
        let mut acc = StreamAccumulator::default();
        acc.observe(&serde_json::json!({"type": "content_block_start", "index": 0,
            "content_block": {"type": "redacted_thinking", "data": "opaque-bytes"}}));

        assert_eq!(
            acc.blocks(),
            vec![ContentBlock::RedactedThinking { data: "opaque-bytes".to_string() }]
        );
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
            build_request_body("claude-sonnet-5", &TurnRequest::bare(turn)),
            serde_json::json!({
                "model": "claude-sonnet-5",
                "max_tokens": 64000,
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

        let body = build_request_body("claude-sonnet-5", &TurnRequest::bare(turn));

        assert!(body.get("system").is_none(), "system must be absent, got: {body}");
        assert_eq!(
            body,
            serde_json::json!({
                "model": "claude-sonnet-5",
                "max_tokens": 64000,
                "stream": true,
                "messages": [{ "role": "user", "content": "hi" }],
            })
        );
    }

    #[test]
    fn a_request_with_tools_carries_the_anthropic_dialect() {
        // TurnRequest is dialect-neutral; THIS provider chooses the Anthropic
        // projection at the boundary. The text-only goldens above are untouched
        // by construction: empty tools still omits the key entirely.
        let def = crate::tools::ToolDef {
            name: "specimen",
            description: "d".to_string(),
            schema: serde_json::json!({ "type": "object", "properties": {}, "required": [] }),
        };
        let turn = ModelTurn {
            system: None,
            messages: vec![text_msg(ProviderRole::User, "hi")],
        };
        let body = build_request_body(
            "m",
            &TurnRequest { turn, tools: vec![def], tool_choice: None },
        );

        assert_eq!(
            body["tools"],
            serde_json::json!([{
                "name": "specimen",
                "description": "d",
                "input_schema": { "type": "object", "properties": {}, "required": [] },
            }])
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
    fn thinking_blocks_encode_in_their_exact_wire_shapes() {
        // Opaque capture-and-echo: the engine never interprets these fields, so
        // the encoding must be exactly what the stream delivered.
        assert_eq!(
            block_json(&ContentBlock::Thinking {
                thinking: "let me check".to_string(),
                signature: "sig-abc".to_string(),
            }),
            serde_json::json!({
                "type": "thinking",
                "thinking": "let me check",
                "signature": "sig-abc",
            })
        );
        assert_eq!(
            block_json(&ContentBlock::RedactedThinking { data: "opaque-bytes".to_string() }),
            serde_json::json!({ "type": "redacted_thinking", "data": "opaque-bytes" })
        );
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
            build_request_body("m", &TurnRequest::bare(turn))["messages"][0]["content"],
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
            build_request_body("m", &TurnRequest::bare(turn))["messages"][0]["content"],
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

        let block = &build_request_body("m", &TurnRequest::bare(turn))["messages"][0]["content"][0];

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
            .run_turn(TurnRequest::bare(turn), tx, cancel_rx)
            .await
            .expect("live call should succeed with a valid key");

        drain.await.unwrap();

        assert!(!outcome.text.is_empty());
        assert!(!outcome.stopped);
    }
}
