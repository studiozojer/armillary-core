//! The loop, end to end, over a REAL listener — send/interrupt/evict wired
//! to `ScriptedProvider` (Task 10) via a swapped-in `AppState.provider`.
//!
//! Unlike `tests/subscribe.rs`'s hand-rolled `TcpStream` + chunk-decoder,
//! this file drives everything through `reqwest` (already a normal
//! dependency of the crate — see `Cargo.toml` — so it's available to
//! integration tests too, not just `src/provider.rs`): `reqwest` already
//! de-chunks HTTP chunked transfer-encoding for us, so the SSE half only
//! needs a text framer splitting on the blank line between frames, not a
//! second decoder underneath it.

use armillary_engine::{
    app,
    log::envelope::EventEnvelope,
    log::store::LogStore,
    projection::ModelTurn,
    provider::{ModelProvider, ProviderError, ScriptedProvider, TurnOutcome},
    sessions::Sessions,
    state::{AppState, ModelConfig},
};
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, watch};

const READ_TIMEOUT: Duration = Duration::from_secs(2);

fn model_config() -> ModelConfig {
    ModelConfig {
        model: "scripted".to_string(),
        api_key: None,
    }
}

/// Spawns a real listener on an OS-assigned loopback port, serving `app`
/// over `data_dir` with `provider` swapped in. Returns the address, a handle
/// to the same `Sessions` the server holds (for out-of-band assertions), and
/// the workspace root (empty — this test file never exercises `boot`).
async fn spawn(data_dir: &Path, provider: Arc<dyn ModelProvider>) -> (SocketAddr, Arc<Sessions>) {
    let store = LogStore::open(data_dir).unwrap();
    let sessions = Arc::new(Sessions::new(store));
    let root = tempfile::tempdir().unwrap().keep();

    let state = AppState {
        root: root.canonicalize().unwrap(),
        sessions: sessions.clone(),
        model: model_config(),
        provider,
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app(state)).await;
    });

    (addr, sessions)
}

/// Splits decoded SSE body text into `(event, data)` frames on the blank
/// line that terminates each one — the same shape `tests/subscribe.rs`'s
/// `SseFramer` uses, just fed from `reqwest`'s already-de-chunked bytes
/// instead of a raw dechunker.
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

struct SseClient {
    response: reqwest::Response,
    framer: SseFramer,
    queue: VecDeque<(String, String)>,
}

impl SseClient {
    async fn connect(client: &reqwest::Client, url: &str) -> Self {
        let response = client.get(url).send().await.unwrap();
        assert!(response.status().is_success(), "{}", response.status());
        SseClient {
            response,
            framer: SseFramer::default(),
            queue: VecDeque::new(),
        }
    }

    async fn next_frame(&mut self) -> (String, String) {
        loop {
            if let Some(frame) = self.queue.pop_front() {
                return frame;
            }
            let chunk = tokio::time::timeout(READ_TIMEOUT, self.response.chunk())
                .await
                .expect("timed out waiting for an SSE frame")
                .unwrap()
                .expect("connection closed before the expected frame arrived");
            self.framer.feed(&chunk);
            for frame in self.framer.drain_frames() {
                self.queue.push_back(frame);
            }
        }
    }
}

async fn create_instance(client: &reqwest::Client, addr: SocketAddr) -> String {
    let created: serde_json::Value = client
        .post(format!("http://{addr}/instances"))
        .json(&serde_json::json!({ "operator": null }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    created["id"].as_str().unwrap().to_string()
}

async fn attach(client: &reqwest::Client, addr: SocketAddr, id: &str) -> serde_json::Value {
    client
        .get(format!("http://{addr}/instances/{id}"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

/// Test double for scenario (d): wraps a `ScriptedProvider` (so a turn still
/// actually completes and the send flow's normal machinery runs) while
/// recording the `ModelTurn` it was actually handed — the only way to
/// observe, from outside `loop_.rs`, what `project_context` produced for a
/// given turn.
struct RecordingProvider {
    inner: ScriptedProvider,
    last_turn: Mutex<Option<ModelTurn>>,
}

impl RecordingProvider {
    fn new(inner: ScriptedProvider) -> Self {
        RecordingProvider {
            inner,
            last_turn: Mutex::new(None),
        }
    }
}

#[async_trait::async_trait]
impl ModelProvider for RecordingProvider {
    async fn run_turn(
        &self,
        turn: ModelTurn,
        sink: mpsc::Sender<String>,
        cancel: watch::Receiver<bool>,
    ) -> Result<TurnOutcome, ProviderError> {
        *self.last_turn.lock().unwrap() = Some(turn.clone());
        self.inner.run_turn(turn, sink, cancel).await
    }
}

// --- (a) send returns 201 receipt whose seq is the user event's ---

#[tokio::test]
async fn send_returns_the_user_events_id_and_seq() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(ScriptedProvider::new(vec!["hi"]));
    let (addr, sessions) = spawn(&data_dir, provider).await;
    let client = reqwest::Client::new();
    let id = create_instance(&client, addr).await;

    let response = client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "hello", "clientKey": "c1" }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::CREATED);
    let receipt: serde_json::Value = response.json().await.unwrap();

    // seq 1 is instance_created; seq 2 is this user_message.
    assert_eq!(receipt["seq"], 2);
    assert!(receipt["id"].as_str().is_some_and(|s| !s.is_empty()));

    // Proven over the log directly, not just the response: the receipt's id
    // really is the user event's id.
    let events = sessions.store().read_from(&id, 0).unwrap();
    let user_ev = events.iter().find(|e| e.event_type == "user_message").unwrap();
    assert_eq!(user_ev.id, receipt["id"]);
    assert_eq!(user_ev.seq, receipt["seq"].as_u64().unwrap());
    assert_eq!(user_ev.data["clientKey"], "c1");
}

// --- (b) echo -> >=2 transient snapshots -> durable assistant_message ---

#[tokio::test]
async fn subscriber_sees_echo_then_transients_then_the_durable_assistant_message() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(
        ScriptedProvider::new(vec!["Hel", "Hello", "Hello there"])
            .with_pause(Duration::from_millis(20)),
    );
    let (addr, _sessions) = spawn(&data_dir, provider).await;
    let client = reqwest::Client::new();
    let id = create_instance(&client, addr).await;

    let mut sub = SseClient::connect(&client, &format!("http://{addr}/streams/{id}/events?from=0")).await;
    // Drain the replay of instance_created (seq 1) + caught-up before send.
    loop {
        let (event, _) = sub.next_frame().await;
        if event == "caught-up" {
            break;
        }
    }

    let response = client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "hi", "clientKey": "c1" }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::CREATED);

    // The user echo arrives first, clientKey intact.
    let (event, data) = sub.next_frame().await;
    assert_eq!(event, "envelope");
    let echo: serde_json::Value = serde_json::from_str(&data).unwrap();
    assert_eq!(echo["type"], "user_message");
    assert_eq!(echo["data"]["clientKey"], "c1");

    // At least 2 transient assistant_delta snapshots before the durable
    // assistant_message.
    let mut transient_count = 0;
    let mut generation_seen: Option<String> = None;
    let assistant_message = loop {
        let (event, data) = sub.next_frame().await;
        assert_eq!(event, "envelope");
        let json: serde_json::Value = serde_json::from_str(&data).unwrap();
        if json["seq"] == 0 {
            assert_eq!(json["type"], "assistant_delta");
            transient_count += 1;
            generation_seen = Some(json["data"]["generation"].as_str().unwrap().to_string());
            continue;
        }
        assert_eq!(json["type"], "assistant_message", "no other durable type expected here");
        break json;
    };

    assert!(transient_count >= 2, "expected >=2 transient snapshots, got {transient_count}");
    assert_eq!(assistant_message["data"]["text"], "Hello there");
    assert_eq!(assistant_message["data"]["interrupted"], false);
    assert_eq!(assistant_message["data"]["model"], "scripted");
    assert_eq!(
        assistant_message["data"]["generation"],
        generation_seen.expect("at least one transient should have carried a generation")
    );
}

// --- (c) interrupt mid-script -> durable interrupt, then partial assistant_message ---

#[tokio::test]
async fn interrupt_mid_script_records_interrupt_then_a_partial_assistant_message() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(
        ScriptedProvider::new(vec!["a", "ab", "abc", "abcd"]).with_pause(Duration::from_millis(50)),
    );
    let (addr, sessions) = spawn(&data_dir, provider).await;
    let client = reqwest::Client::new();
    let id = create_instance(&client, addr).await;

    client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "hi", "clientKey": "c1" }))
        .send()
        .await
        .unwrap();

    // Give the turn time to emit its first snapshot, then interrupt while
    // still mid-script (pauses are 50ms; fragment 4 of 4 is well past this).
    tokio::time::sleep(Duration::from_millis(30)).await;
    let response = client
        .post(format!("http://{addr}/instances/{id}/interrupt"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);

    // Poll until the turn finishes (assistant_message lands) — bounded by
    // this test's own timeout, not a fixed sleep.
    let events = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let events = sessions.store().read_from(&id, 0).unwrap();
            if events.iter().any(|e| e.event_type == "assistant_message") {
                return events;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("turn did not finish after interrupt");

    let interrupt_idx = events.iter().position(|e| e.event_type == "interrupt");
    let assistant_idx = events.iter().position(|e| e.event_type == "assistant_message");
    let (interrupt_idx, assistant_idx) = (
        interrupt_idx.expect("an interrupt event must be recorded"),
        assistant_idx.expect("an assistant_message must be recorded"),
    );
    assert!(interrupt_idx < assistant_idx, "interrupt must be recorded BEFORE the partial assistant_message");

    let assistant = &events[assistant_idx];
    assert_eq!(assistant.data["interrupted"], true);
    // The scripted provider only notices `cancel` between fragments (after a
    // pause), so the recorded text is whichever fragment was in flight —
    // never empty, never the full un-cancelled script.
    assert_ne!(assistant.data["text"], "abcd");
    assert!(!assistant.data["text"].as_str().unwrap().is_empty());
}

// --- (d) evict, then send: the projection fed to the provider excludes it ---

#[tokio::test]
async fn evicted_message_is_absent_from_the_next_turns_projection() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let recorder = Arc::new(RecordingProvider::new(ScriptedProvider::new(vec!["ok"])));
    let (addr, sessions) = spawn(&data_dir, recorder.clone()).await;
    let client = reqwest::Client::new();
    let id = create_instance(&client, addr).await;

    let first: serde_json::Value = client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "evict me", "clientKey": "c1" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let evicted_event_id = first["id"].as_str().unwrap().to_string();

    wait_for_assistant_message(&sessions, &id, 1).await;

    let response = client
        .post(format!("http://{addr}/instances/{id}/evict"))
        .json(&serde_json::json!({ "eventId": evicted_event_id }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);

    client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "second turn", "clientKey": "c2" }))
        .send()
        .await
        .unwrap();

    wait_for_assistant_message(&sessions, &id, 2).await;

    let last_turn = recorder.last_turn.lock().unwrap().clone().expect("a turn was recorded");
    assert!(
        !last_turn.messages.iter().any(|m| m.content.contains("evict me")),
        "the evicted message must not reach the provider: {last_turn:?}"
    );
    assert!(last_turn.messages.iter().any(|m| m.content.contains("second turn")));
}

async fn wait_for_assistant_message(sessions: &Arc<Sessions>, id: &str, count: usize) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let events = sessions.store().read_from(id, 0).unwrap();
            if events.iter().filter(|e| e.event_type == "assistant_message").count() >= count {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("turn did not finish in time");
}

// --- (e) second send mid-turn -> 409 ---

#[tokio::test]
async fn a_second_send_while_a_turn_runs_is_409_turn_in_progress() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(ScriptedProvider::new(vec!["a", "b"]).with_pause(Duration::from_millis(200)));
    let (addr, _sessions) = spawn(&data_dir, provider).await;
    let client = reqwest::Client::new();
    let id = create_instance(&client, addr).await;

    let first = client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "first", "clientKey": "c1" }))
        .send()
        .await
        .unwrap();
    assert_eq!(first.status(), reqwest::StatusCode::CREATED);

    let second = client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "second", "clientKey": "c2" }))
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), reqwest::StatusCode::CONFLICT);
    let body = second.text().await.unwrap();
    assert_eq!(body, "turn_in_progress");
}

// --- (f) the crash-resume proof ---

#[tokio::test]
async fn crash_resume_the_log_survives_dropping_and_rebuilding_the_whole_process_state() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(ScriptedProvider::new(vec!["done"]));
    let (addr, sessions) = spawn(&data_dir, provider).await;
    let client = reqwest::Client::new();
    let id = create_instance(&client, addr).await;

    client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "hello", "clientKey": "c1" }))
        .send()
        .await
        .unwrap();
    wait_for_assistant_message(&sessions, &id, 1).await;

    let head_before = sessions.store().head_seq(&id).unwrap();
    assert_eq!(head_before, 3, "instance_created, user_message, assistant_message");

    // "Crash": drop every in-process handle this test holds onto the first
    // `Sessions`/`AppState` — the server task itself is left running (there
    // is no clean way to un-spawn it from here), but nothing below reads
    // from it again; only a FRESH `Sessions` built from the same `data_dir`
    // is consulted from this point on, so this proves the log — not the
    // process — is the source of truth.
    drop(sessions);

    let fresh_store = LogStore::open(&data_dir).unwrap();
    let fresh_sessions = Arc::new(Sessions::new(fresh_store));

    assert_eq!(fresh_sessions.store().head_seq(&id).unwrap(), head_before, "headSeq survives the rebuild");

    let replayed = fresh_sessions.store().read_from(&id, 0).unwrap();
    assert_eq!(replayed.len(), 3);
    assert_eq!(replayed[0].event_type, "instance_created");
    assert_eq!(replayed[1].event_type, "user_message");
    assert_eq!(replayed[1].data["text"], "hello");
    assert_eq!(replayed[2].event_type, "assistant_message");
    assert_eq!(replayed[2].data["text"], "done");

    // And over HTTP, via a brand-new server built on the SAME data dir and a
    // brand-new AppState — attach reports the identical headSeq.
    let (fresh_addr, _fresh_sessions_2) = spawn(&data_dir, Arc::new(ScriptedProvider::new(vec!["unused"]))).await;
    let attach_info = attach(&client, fresh_addr, &id).await;
    assert_eq!(attach_info["headSeq"], head_before);
    assert_eq!(attach_info["earliestSeq"], 1);
}

// --- bonus: BootDrift re-record path (not in the brief's lettered list, but
// cited by the turn's step 3 and cheap to pin directly) ---

#[tokio::test]
async fn a_drifted_boot_event_is_rerecorded_fresh_before_the_turn_runs() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(ScriptedProvider::new(vec!["ok"]));
    let root_dir = tempfile::tempdir().unwrap();
    std::fs::write(root_dir.path().join("boot.md"), "# current boot content").unwrap();

    let store = LogStore::open(&data_dir).unwrap();
    let sessions = Arc::new(Sessions::new(store));
    let state = AppState {
        root: root_dir.path().canonicalize().unwrap(),
        sessions: sessions.clone(),
        model: model_config(),
        provider,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app(state)).await;
    });

    let client = reqwest::Client::new();
    let id = create_instance(&client, addr).await;

    // Hand-append a `boot` event recording the WRONG hash — simulating
    // drift (the file on disk changed since this event was written).
    sessions
        .append(
            &id,
            armillary_engine::sessions::NewEvent {
                actor: armillary_engine::log::envelope::Actor {
                    role: armillary_engine::log::envelope::Role::System,
                    instance: None,
                },
                event_type: "boot".to_string(),
                data: serde_json::json!({ "path": "boot.md", "sha256": "0".repeat(64) }),
            },
        )
        .unwrap();

    client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "hi", "clientKey": "c1" }))
        .send()
        .await
        .unwrap();

    wait_for_assistant_message(&sessions, &id, 1).await;

    let events = sessions.store().read_from(&id, 0).unwrap();
    let boot_events: Vec<&EventEnvelope> = events.iter().filter(|e| e.event_type == "boot").collect();
    assert_eq!(boot_events.len(), 2, "the drifted boot must be followed by a fresh re-record");
    let correct_sha = armillary_engine::hash::sha256_hex(b"# current boot content");
    assert_eq!(boot_events[1].data["sha256"], correct_sha);
    assert_eq!(boot_events[1].actor.role, armillary_engine::log::envelope::Role::System);

    let assistant = events.iter().find(|e| e.event_type == "assistant_message").unwrap();
    assert_eq!(assistant.data["interrupted"], false, "the turn recovers and completes normally");
}

// --- 404 unknown_instance / unknown_event: every mutating endpoint, both
// the "the stream doesn't exist at all" shape and evict's own "the stream
// exists but this eventId doesn't" shape. Unverified-by-inspection was the
// exact gap a review found here — these four pin it directly. ---

#[tokio::test]
async fn send_to_an_unknown_instance_is_404_unknown_instance() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(ScriptedProvider::new(vec!["unused"]));
    let (addr, _sessions) = spawn(&data_dir, provider).await;
    let client = reqwest::Client::new();

    let response = client
        .post(format!("http://{addr}/instances/does-not-exist/send"))
        .json(&serde_json::json!({ "text": "hi", "clientKey": "c1" }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
    assert_eq!(response.text().await.unwrap(), "unknown_instance");
}

#[tokio::test]
async fn interrupt_on_an_unknown_instance_is_404_unknown_instance() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(ScriptedProvider::new(vec!["unused"]));
    let (addr, _sessions) = spawn(&data_dir, provider).await;
    let client = reqwest::Client::new();

    let response = client
        .post(format!("http://{addr}/instances/does-not-exist/interrupt"))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
    assert_eq!(response.text().await.unwrap(), "unknown_instance");
}

#[tokio::test]
async fn evict_on_an_unknown_instance_is_404_unknown_instance() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(ScriptedProvider::new(vec!["unused"]));
    let (addr, _sessions) = spawn(&data_dir, provider).await;
    let client = reqwest::Client::new();

    let response = client
        .post(format!("http://{addr}/instances/does-not-exist/evict"))
        .json(&serde_json::json!({ "eventId": "whatever" }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
    assert_eq!(response.text().await.unwrap(), "unknown_instance");
}

#[tokio::test]
async fn evict_with_an_event_id_not_in_the_stream_is_404_unknown_event() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(ScriptedProvider::new(vec!["ok"]));
    let (addr, _sessions) = spawn(&data_dir, provider).await;
    let client = reqwest::Client::new();
    let id = create_instance(&client, addr).await;

    let response = client
        .post(format!("http://{addr}/instances/{id}/evict"))
        .json(&serde_json::json!({ "eventId": "not-a-real-event-id" }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
    assert_eq!(response.text().await.unwrap(), "unknown_event");
}
