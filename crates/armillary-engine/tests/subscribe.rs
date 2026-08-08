//! `GET /streams/{stream}/events` against a REAL listener.
//!
//! `tests/routes.rs`'s `oneshot` harness proves status codes and bodies, but
//! it can't assert a stream: a `oneshot` response is fully materialized
//! before the test ever sees it, so there is no way to observe "these three
//! frames arrived, then a fourth arrived later, in this order" — which is
//! the entire thing this endpoint has to get right (constitution A-1/A-2).
//! So this file binds a real `TcpListener`, drives `axum::serve` in a
//! spawned task, and reads the response with a hand-rolled HTTP/1.1 client:
//! parse the header block, de-chunk the (chunked) SSE body incrementally,
//! and split it into `event: .. / data: ..` frames as they arrive — each
//! socket read guarded by a short timeout so a hung stream fails the test
//! instead of the suite.

use armillary_engine::{
    app,
    log::envelope::{Actor, EventEnvelope, Role},
    log::store::LogStore,
    provider::{self, KeylessProvider},
    sessions::{NewEvent, Sessions},
    state::{AppState, ModelConfig},
};
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const READ_TIMEOUT: Duration = Duration::from_secs(2);

fn model_config() -> ModelConfig {
    ModelConfig {
        model: "claude-sonnet-5".to_string(),
    }
}

fn actor() -> Actor {
    Actor {
        role: Role::User,
        instance: None,
        principal: None,
    }
}

fn append(sessions: &Sessions, stream: &str, text: &str) {
    sessions
        .append(
            stream,
            NewEvent {
                actor: actor(),
                event_type: "user_message".to_string(),
                data: serde_json::json!({ "text": text }),
            },
        )
        .unwrap();
}

/// Spawns a real listener on an OS-assigned loopback port, serving `app`
/// over the given data dir. Returns the address and a handle to the same
/// `Sessions` the server holds, so a test can append events out-of-band
/// (racing the connection deliberately, or building gap/transient fixtures).
async fn spawn(data_dir: &std::path::Path) -> (SocketAddr, Arc<Sessions>) {
    let store = LogStore::open(data_dir).unwrap();
    let sessions = Arc::new(Sessions::new(store));
    let root = tempfile::tempdir().unwrap().keep();

    let state = AppState {
        root: root.canonicalize().unwrap(),
        sessions: sessions.clone(),
        model: model_config(),
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        registry_dir: std::path::PathBuf::from("/nonexistent/registry"),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: None,
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app(state)).await;
    });

    (addr, sessions)
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Decodes HTTP/1.1 chunked-transfer framing incrementally: feed it raw
/// socket bytes as they arrive, and it hands back whatever complete chunks'
/// worth of body content that unblocks, holding a partial chunk until more
/// bytes arrive.
#[derive(Default)]
struct Dechunker {
    buf: Vec<u8>,
}

impl Dechunker {
    fn feed(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    fn drain_complete_chunks(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        while let Some(pos) = find(&self.buf, b"\r\n") {
            let size_line = String::from_utf8_lossy(&self.buf[..pos]).trim().to_string();
            let size_str = size_line.split(';').next().unwrap_or("");
            let Ok(size) = usize::from_str_radix(size_str, 16) else {
                break;
            };
            let data_start = pos + 2;
            let needed = data_start + size + 2; // chunk data + trailing CRLF
            if self.buf.len() < needed {
                break;
            }
            if size == 0 {
                // Terminal chunk. Not expected in these tests (the SSE body
                // never completes), but handled rather than looping forever.
                self.buf.clear();
                break;
            }
            out.extend_from_slice(&self.buf[data_start..data_start + size]);
            self.buf.drain(..needed);
        }
        out
    }
}

/// Splits decoded SSE body text into `(event, data)` frames on the blank
/// line that terminates each one.
#[derive(Default)]
struct SseFramer {
    text: String,
}

impl SseFramer {
    fn feed(&mut self, bytes: &[u8]) {
        self.text.push_str(&String::from_utf8_lossy(bytes));
    }

    fn drain_frames(&mut self) -> Vec<(String, String)> {
        let mut frames = Vec::new();
        while let Some(pos) = self.text.find("\n\n") {
            let frame = self.text[..pos].to_string();
            self.text.drain(..pos + 2);

            let mut event = "message".to_string();
            let mut data = String::new();
            for line in frame.lines() {
                if let Some(rest) = line.strip_prefix("event: ") {
                    event = rest.to_string();
                } else if let Some(rest) = line.strip_prefix("data: ") {
                    data = rest.to_string();
                }
            }
            frames.push((event, data));
        }
        frames
    }
}

/// A raw HTTP/1.1 client for one SSE subscription: sends the GET, exposes
/// the status line, then yields `(event, data)` frames one at a time —
/// reading and de-chunking more of the socket only when the queue runs dry.
struct SseClient {
    stream: TcpStream,
    dechunker: Dechunker,
    framer: SseFramer,
    queue: VecDeque<(String, String)>,
    pub status_line: String,
}

impl SseClient {
    async fn connect(addr: SocketAddr, path: &str) -> Self {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        let request = format!(
            "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAccept: text/event-stream\r\n\r\n"
        );
        stream.write_all(request.as_bytes()).await.unwrap();

        let mut header_buf = Vec::new();
        let body_start = loop {
            let mut chunk = [0u8; 4096];
            let n = tokio::time::timeout(READ_TIMEOUT, stream.read(&mut chunk))
                .await
                .expect("timed out reading response headers")
                .unwrap();
            assert!(n > 0, "connection closed before headers completed");
            header_buf.extend_from_slice(&chunk[..n]);
            if let Some(pos) = find(&header_buf, b"\r\n\r\n") {
                let body_start = header_buf[pos + 4..].to_vec();
                header_buf.truncate(pos);
                break body_start;
            }
        };

        let status_line = String::from_utf8_lossy(&header_buf)
            .lines()
            .next()
            .unwrap_or("")
            .to_string();

        let mut dechunker = Dechunker::default();
        dechunker.feed(&body_start);

        SseClient {
            stream,
            dechunker,
            framer: SseFramer::default(),
            queue: VecDeque::new(),
            status_line,
        }
    }

    async fn next_frame(&mut self) -> (String, String) {
        loop {
            if let Some(frame) = self.queue.pop_front() {
                return frame;
            }
            let decoded = self.dechunker.drain_complete_chunks();
            if !decoded.is_empty() {
                self.framer.feed(&decoded);
                for frame in self.framer.drain_frames() {
                    self.queue.push_back(frame);
                }
                continue;
            }
            let mut buf = [0u8; 4096];
            let n = tokio::time::timeout(READ_TIMEOUT, self.stream.read(&mut buf))
                .await
                .expect("timed out waiting for an SSE frame")
                .unwrap();
            assert!(n > 0, "connection closed before the expected frame arrived");
            self.dechunker.feed(&buf[..n]);
        }
    }

    /// Reads the whole response as plain text (no chunk framing) — for
    /// asserting a non-streaming error response like a 404.
    async fn read_body_text(&mut self) -> String {
        let mut body = self.dechunker.buf.clone();
        // A short read is fine here: error responses are small, and any
        // remaining bytes are drained best-effort within the timeout.
        let mut buf = [0u8; 4096];
        if let Ok(Ok(n)) = tokio::time::timeout(Duration::from_millis(200), self.stream.read(&mut buf)).await {
            body.extend_from_slice(&buf[..n]);
        }
        String::from_utf8_lossy(&body).into_owned()
    }
}

#[tokio::test]
async fn from_zero_replays_three_then_caught_up() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let (addr, sessions) = spawn(&data_dir).await;
    append(&sessions, "s1", "one");
    append(&sessions, "s1", "two");
    append(&sessions, "s1", "three");

    let mut client = SseClient::connect(addr, "/streams/s1/events?from=0").await;
    assert!(client.status_line.contains("200"), "{}", client.status_line);

    for expected_seq in 1..=3u64 {
        let (event, data) = client.next_frame().await;
        assert_eq!(event, "envelope");
        let json: serde_json::Value = serde_json::from_str(&data).unwrap();
        assert_eq!(json["seq"], expected_seq);
    }

    let (event, data) = client.next_frame().await;
    assert_eq!(event, "caught-up");
    let json: serde_json::Value = serde_json::from_str(&data).unwrap();
    assert_eq!(json["headSeq"], 3);
}

/// NOT a test of the replay/tail race itself — `subscribe` is one
/// straight-line `async fn` that fully completes `subscribe_live`, the
/// blocking replay read, and building the whole replay+`caught-up` prefix
/// BEFORE the SSE response's headers are ever written. So by the time this
/// test observes `client.status_line` (which only arrives once the header
/// block is on the wire), `subscribe_live` and the durable read have both
/// unconditionally already run — appending event 4 "during" this test can
/// never land inside the durable read's window, only after it. This test
/// therefore only pins the tail-after-replay path: event 4 arrives exactly
/// once, on the tail, after `caught-up`. It intentionally does NOT exercise
/// the dedup rule (`seq <= last_replayed`) — that an event landing in BOTH
/// the replay batch and the broadcast channel is delivered exactly once —
/// which is covered instead by the unit test
/// `routes::subscribe::tests::skips_durable_duplicates_already_covered_by_replay_but_passes_new_ones_and_transients`
/// in `src/routes/subscribe.rs`, where the replay/broadcast overlap can
/// actually be constructed directly rather than raced over a real socket.
#[tokio::test]
async fn a_fourth_event_appended_after_the_request_lands_arrives_on_the_tail_exactly_once() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let (addr, sessions) = spawn(&data_dir).await;
    append(&sessions, "s1", "one");
    append(&sessions, "s1", "two");
    append(&sessions, "s1", "three");

    let mut client = SseClient::connect(addr, "/streams/s1/events?from=0").await;
    assert!(client.status_line.contains("200"));

    // By this point `subscribe_live` and the durable replay read have both
    // already completed (see the doc comment above) — this append can only
    // ever reach the tail, never the replay batch.
    append(&sessions, "s1", "four");

    let mut seen_seqs = Vec::new();
    loop {
        let (event, data) = client.next_frame().await;
        let json: serde_json::Value = serde_json::from_str(&data).unwrap();
        match event.as_str() {
            "envelope" => seen_seqs.push(json["seq"].as_u64().unwrap()),
            "caught-up" => break,
            other => panic!("unexpected frame: {other}"),
        }
    }
    assert_eq!(seen_seqs, vec![1, 2, 3], "seq 1..=3 replay in order, seq 4 not among them");

    let (event, data) = client.next_frame().await;
    assert_eq!(event, "envelope", "seq 4 arrives on the tail, after caught-up");
    let json: serde_json::Value = serde_json::from_str(&data).unwrap();
    assert_eq!(json["seq"], 4);
}

#[tokio::test]
async fn from_two_replays_only_seq_three() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let (addr, sessions) = spawn(&data_dir).await;
    append(&sessions, "s1", "one");
    append(&sessions, "s1", "two");
    append(&sessions, "s1", "three");

    let mut client = SseClient::connect(addr, "/streams/s1/events?from=2").await;
    assert!(client.status_line.contains("200"));

    let (event, data) = client.next_frame().await;
    assert_eq!(event, "envelope");
    let json: serde_json::Value = serde_json::from_str(&data).unwrap();
    assert_eq!(json["seq"], 3);

    let (event, data) = client.next_frame().await;
    assert_eq!(event, "caught-up");
    let json: serde_json::Value = serde_json::from_str(&data).unwrap();
    assert_eq!(json["headSeq"], 3);
}

#[tokio::test]
async fn a_stream_whose_earliest_is_ahead_of_the_request_emits_gap_first() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    // Hand-write a JSONL file whose first (and only) event is seq 5 —
    // simulating a stream that has been truncated/compacted past seq 1.
    // `LogStore::read_from`/`earliest_seq` read this fine; nothing about it
    // requires the file to start at seq 1.
    let streams_dir = data_dir.join("streams");
    std::fs::create_dir_all(&streams_dir).unwrap();
    let line = serde_json::json!({
        "stream": "s1",
        "id": "id-5",
        "seq": 5,
        "ts": "2026-07-27T00:00:00Z",
        "actor": { "role": "user" },
        "type": "user_message",
        "version": 1,
        "data": { "text": "five" }
    });
    std::fs::write(streams_dir.join("s1.jsonl"), format!("{line}\n")).unwrap();

    let (addr, _sessions) = spawn(&data_dir).await;

    let mut client = SseClient::connect(addr, "/streams/s1/events?from=0").await;
    assert!(client.status_line.contains("200"));

    let (event, data) = client.next_frame().await;
    assert_eq!(event, "gap");
    let json: serde_json::Value = serde_json::from_str(&data).unwrap();
    assert_eq!(json["requestedFrom"], 0);
    assert_eq!(json["earliestAvailable"], 5);

    let (event, data) = client.next_frame().await;
    assert_eq!(event, "envelope");
    let json: serde_json::Value = serde_json::from_str(&data).unwrap();
    assert_eq!(json["seq"], 5);

    let (event, data) = client.next_frame().await;
    assert_eq!(event, "caught-up");
    let json: serde_json::Value = serde_json::from_str(&data).unwrap();
    assert_eq!(json["headSeq"], 5);
}

#[tokio::test]
async fn a_transient_broadcast_after_caught_up_arrives_as_an_envelope_frame_with_seq_zero() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let (addr, sessions) = spawn(&data_dir).await;
    append(&sessions, "s1", "one");

    let mut client = SseClient::connect(addr, "/streams/s1/events?from=0").await;
    assert!(client.status_line.contains("200"));

    let (event, _) = client.next_frame().await;
    assert_eq!(event, "envelope");
    let (event, _) = client.next_frame().await;
    assert_eq!(event, "caught-up");

    let hint = EventEnvelope {
        stream: "s1".to_string(),
        id: "hint-1".to_string(),
        seq: 0,
        ts: "2026-07-27T00:00:00Z".to_string(),
        actor: actor(),
        event_type: "typing".to_string(),
        thread: None,
        parent: None,
        version: 1,
        cost: None,
        data: serde_json::json!({ "who": "tycho" }),
    };
    sessions.broadcast_transient("s1", hint);

    let (event, data) = client.next_frame().await;
    assert_eq!(event, "envelope");
    let json: serde_json::Value = serde_json::from_str(&data).unwrap();
    assert_eq!(json["seq"], 0);
    assert_eq!(json["type"], "typing");
}

#[tokio::test]
async fn unknown_stream_is_404() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let (addr, _sessions) = spawn(&data_dir).await;

    let mut client = SseClient::connect(addr, "/streams/does-not-exist/events?from=0").await;
    assert!(client.status_line.contains("404"), "{}", client.status_line);
    let body = client.read_body_text().await;
    assert_eq!(body, "unknown_stream");
}

#[tokio::test]
async fn a_non_numeric_from_is_400_bad_from() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let (addr, sessions) = spawn(&data_dir).await;
    append(&sessions, "s1", "one");

    let mut client = SseClient::connect(addr, "/streams/s1/events?from=nope").await;
    assert!(client.status_line.contains("400"), "{}", client.status_line);
    let body = client.read_body_text().await;
    assert_eq!(body, "bad_from");
}

#[tokio::test]
async fn from_u64_max_falls_out_as_an_ordinary_empty_replay_not_a_panic() {
    // `from + 1` overflows at this boundary; the handler must not let an
    // adversarial cursor value panic the request task.
    let data_dir = tempfile::tempdir().unwrap().keep();
    let (addr, sessions) = spawn(&data_dir).await;
    append(&sessions, "s1", "one");

    let mut client =
        SseClient::connect(addr, &format!("/streams/s1/events?from={}", u64::MAX)).await;
    assert!(client.status_line.contains("200"), "{}", client.status_line);

    let (event, data) = client.next_frame().await;
    assert_eq!(event, "caught-up");
    let json: serde_json::Value = serde_json::from_str(&data).unwrap();
    assert_eq!(json["headSeq"], 1);
}
