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

use crate::projection::{ModelTurn, ProviderRole};
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnOutcome {
    pub text: String,
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

fn role_str(role: ProviderRole) -> &'static str {
    match role {
        ProviderRole::User => "user",
        ProviderRole::Assistant => "assistant",
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
                stopped: true,
                model: self.model.clone(),
            });
        }

        let messages: Vec<serde_json::Value> = turn
            .messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": role_str(m.role),
                    "content": m.content,
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": messages,
            "stream": true,
        });
        if let Some(system) = &turn.system {
            body["system"] = serde_json::json!(system);
        }

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
        let mut accumulated = String::new();

        loop {
            tokio::select! {
                biased;

                changed = cancel.changed() => {
                    if changed.is_err() || *cancel.borrow() {
                        return Ok(TurnOutcome {
                            text: accumulated,
                            stopped: true,
                            model: self.model.clone(),
                        });
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

                                if let Some(delta) = extract_text_delta(&v) {
                                    accumulated.push_str(delta);
                                    // I-4: snapshot conversion happens here, at
                                    // the source — the sink never sees a raw
                                    // delta, only the whole string so far.
                                    // Ignored send error: no receiver just
                                    // means nobody is painting this turn.
                                    let _ = sink.send(accumulated.clone()).await;
                                }

                                if is_message_stop(&v) {
                                    return Ok(TurnOutcome {
                                        text: accumulated,
                                        stopped: false,
                                        model: self.model.clone(),
                                    });
                                }
                            }
                        }
                        Some(Err(e)) => {
                            return Err(ProviderError::Http(e.to_string()));
                        }
                        None => {
                            return Ok(TurnOutcome {
                                text: accumulated,
                                stopped: false,
                                model: self.model.clone(),
                            });
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
                    text: last_sent,
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
            text: last_sent,
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

    // --- KeylessProvider ---

    #[tokio::test]
    async fn keyless_provider_always_returns_no_api_key() {
        let provider = KeylessProvider;
        let (tx, _rx) = mpsc::channel(8);
        let (_cancel_tx, cancel_rx) = watch::channel(false);

        let err = provider.run_turn(empty_turn(), tx, cancel_rx).await.unwrap_err();

        assert_eq!(err, ProviderError::NoApiKey);
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
                content: "Reply with exactly one word: hello".to_string(),
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
