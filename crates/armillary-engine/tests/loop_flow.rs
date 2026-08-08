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
    provider::{self, ModelProvider, ProviderError, ScriptedProvider, TurnOutcome},
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
        providers: provider::fixed(provider),
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

/// Like `spawn`, but the workspace root holds a boot file at `boot_rel`, and
/// `AppState.boot` carries `declared`.
///
/// Returns the root so a test can mutate the boot file mid-stream — the only way
/// to make the two `boot` writers (`routes::instances::append_boot_event` at
/// create, `loop_::rerecord_boot` on drift) both fire on one real stream.
async fn spawn_with_boot(
    data_dir: &Path,
    provider: Arc<dyn ModelProvider>,
    boot_rel: &str,
    contents: &str,
    declared: Option<&str>,
) -> (SocketAddr, Arc<Sessions>, std::path::PathBuf) {
    let store = LogStore::open(data_dir).unwrap();
    let sessions = Arc::new(Sessions::new(store));
    let root = tempfile::tempdir().unwrap().keep();
    std::fs::write(root.join(boot_rel), contents).unwrap();
    let root = root.canonicalize().unwrap();

    let state = AppState {
        root: root.clone(),
        sessions: sessions.clone(),
        model: model_config(),
        providers: provider::fixed(provider),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        registry_dir: std::path::PathBuf::from("/nonexistent/registry"),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: declared.map(|s| s.to_string()),
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app(state)).await;
    });

    (addr, sessions, root)
}

/// Like `spawn`, but wires in the REAL `KeyedProviders` with neither
/// credential present, rather than `provider::fixed`'s single test double —
/// so provider SELECTION (`choose_provider`, keyed off the model string)
/// actually runs, and every turn fails with `no_api_key` regardless of which
/// model an instance asks for. `provider::fixed` ignores the model string
/// entirely, which would leave this file with no way to prove a turn asked
/// for the INSTANCE's provider rather than the engine's configured one.
async fn spawn_keyless(data_dir: &Path) -> (SocketAddr, Arc<Sessions>) {
    let store = LogStore::open(data_dir).unwrap();
    let sessions = Arc::new(Sessions::new(store));
    let root = tempfile::tempdir().unwrap().keep();

    let state = AppState {
        root: root.canonicalize().unwrap(),
        sessions: sessions.clone(),
        model: model_config(),
        providers: Arc::new(provider::KeyedProviders {
            anthropic_key: None,
            zen_key: None,
        }),
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
    create_instance_with(client, addr, None, None).await
}

/// Like `create_instance`, but lets a test pin the operator and/or the model
/// `routes::instances::CreateRequest` accepts at creation — every other test
/// in this file only ever needed the operator-less default until the model
/// became a per-instance fact too.
async fn create_instance_with(
    client: &reqwest::Client,
    addr: SocketAddr,
    operator: Option<&str>,
    model: Option<&str>,
) -> String {
    let created: serde_json::Value = client
        .post(format!("http://{addr}/instances"))
        .json(&serde_json::json!({ "operator": operator, "model": model }))
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
        req: armillary_engine::provider::TurnRequest,
        sink: mpsc::Sender<String>,
        cancel: watch::Receiver<bool>,
    ) -> Result<TurnOutcome, ProviderError> {
        *self.last_turn.lock().unwrap() = Some(req.turn.clone());
        self.inner.run_turn(req, sink, cancel).await
    }
}

// --- (a) send returns 201 receipt whose seq is the user event's ---

/// True when any text block in the message contains `needle`.
///
/// `ProviderMessage.content` became a block list with the tool-use migration;
/// these assertions are about what the model can read, which is the text.
fn message_contains(m: &armillary_engine::projection::ProviderMessage, needle: &str) -> bool {
    m.content.iter().any(|b| match b {
        armillary_engine::projection::ContentBlock::Text(t) => t.contains(needle),
        _ => false,
    })
}

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

    // seq 1 is instance_created, seq 2 the composition event DD-1 records at
    // creation, and seq 3 is this user_message.
    assert_eq!(receipt["seq"], 3);
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
        !last_turn.messages.iter().any(|m| message_contains(m, "evict me")),
        "the evicted message must not reach the provider: {last_turn:?}"
    );
    assert!(last_turn.messages.iter().any(|m| message_contains(m, "second turn")));
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
    assert_eq!(
        head_before, 4,
        "instance_created, composition, user_message, assistant_message"
    );

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
    assert_eq!(replayed.len(), 4);
    assert_eq!(replayed[0].event_type, "instance_created");
    assert_eq!(replayed[1].event_type, "composition");
    assert_eq!(replayed[2].event_type, "user_message");
    assert_eq!(replayed[2].data["text"], "hello");
    assert_eq!(replayed[3].event_type, "assistant_message");
    assert_eq!(replayed[3].data["text"], "done");

    // And over HTTP, via a brand-new server built on the SAME data dir and a
    // brand-new AppState — attach reports the identical headSeq.
    let (fresh_addr, _fresh_sessions_2) = spawn(&data_dir, Arc::new(ScriptedProvider::new(vec!["unused"]))).await;
    let attach_info = attach(&client, fresh_addr, &id).await;
    assert_eq!(attach_info["headSeq"], head_before);
    assert_eq!(attach_info["earliestSeq"], 1);
}

// --- bonus: BootDrift re-record path (not in the brief's lettered list, but
// cited by the turn's step 3 and cheap to pin directly) ---

/// The first declared file of a boot event, whichever shape it was written in.
///
/// B-2 turned the payload from `{path, sha256}` into `{files: [{path, sha256,
/// present}]}`. These tests are about the two WRITERS agreeing on a stream, not
/// about the payload's shape, so they read through this rather than restating
/// the shape six times.
fn boot_file(e: &EventEnvelope, key: &str) -> serde_json::Value {
    e.data["files"][0][key].clone()
}

#[tokio::test]
async fn a_drifted_boot_event_is_rerecorded_fresh_before_the_turn_runs() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(ScriptedProvider::new(vec!["ok"]));
    // `boot: None` deliberately: this test isolates the RECOVERY path, so the
    // stream's only boot event is the hand-written drifted one below. The two
    // writers meeting on one stream is the test that follows.
    let (addr, sessions, _root) =
        spawn_with_boot(&data_dir, provider, "boot.md", "# current boot content", None).await;

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
                    principal: None,
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
    assert_eq!(boot_file(boot_events[1], "sha256"), correct_sha);
    assert_eq!(boot_events[1].actor.role, armillary_engine::log::envelope::Role::System);

    let assistant = events.iter().find(|e| e.event_type == "assistant_message").unwrap();
    assert_eq!(assistant.data["interrupted"], false, "the turn recovers and completes normally");
}

#[tokio::test]
async fn a_manifest_edited_mid_session_is_re_derived_before_the_next_turn() {
    // DD-1's drift half, end to end, on a real stream with both writers: the
    // event `create` wrote, then the repair `run_turn` writes when the file
    // moves under it. Without this the session goes on describing a workspace
    // that no longer exists — which is the failure D3 shipped silently,
    // because a forged tool result had no hash to check.
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(ScriptedProvider::new(vec!["ok"]));

    let store = LogStore::open(&data_dir).unwrap();
    let sessions = Arc::new(Sessions::new(store));
    let root = tempfile::tempdir().unwrap().keep();
    std::fs::write(root.join("modules.toml"), "[[repos]]\nname='before'\npath='p'\n").unwrap();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = AppState {
        root: root.canonicalize().unwrap(),
        sessions: sessions.clone(),
        model: model_config(),
        providers: provider::fixed(provider),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        registry_dir: std::path::PathBuf::from("/nonexistent/registry"),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: None,
    };
    tokio::spawn(async move {
        let _ = axum::serve(listener, app(state)).await;
    });

    let client = reqwest::Client::new();
    let id = create_instance(&client, addr).await;

    // The workspace is recomposed while the session is open — a repo added to
    // `modules.toml` is the ordinary case, not an exotic one.
    std::fs::write(
        root.join("modules.toml"),
        "[[repos]]\nname='before'\npath='p'\n[[repos]]\nname='after'\npath='q'\n",
    )
    .unwrap();

    client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "what is composed?", "clientKey": "c1" }))
        .send()
        .await
        .unwrap();
    wait_for_assistant_message(&sessions, &id, 1).await;

    let events = sessions.store().read_from(&id, 0).unwrap();
    let compositions: Vec<&EventEnvelope> = events
        .iter()
        .filter(|e| e.event_type == "composition")
        .collect();
    assert_eq!(
        compositions.len(),
        2,
        "I-1: the correction is a NEW event, never an edit of the first"
    );

    // The fresh one describes the workspace as it now is, and the projection
    // lands on it rather than on the stale predecessor.
    let fresh = format!("{}", compositions[1].data);
    assert!(fresh.contains("after"), "{fresh}");
    let turn = armillary_engine::projection::project_context(&events, &root.canonicalize().unwrap())
        .expect("the superseded event must not fail the projection");
    let rendered = format!("{:?}", turn.messages);
    assert!(rendered.contains("after"), "{rendered}");

    let assistant = events
        .iter()
        .find(|e| e.event_type == "assistant_message")
        .unwrap();
    assert_eq!(
        assistant.data["interrupted"], false,
        "the turn recovers and completes normally"
    );
}

// --- the two `boot` writers on ONE real stream ---

/// There are two producers of `boot` events — `routes::instances::append_boot_event`
/// at instance creation, and `loop_::rerecord_boot` on `BootDrift` — and until this
/// test each was only ever proven in isolation (the drift test above hand-appends
/// its own first boot event and runs with `boot: None`). Nothing showed the two
/// agreeing on a real stream: same `path` spelling, advancing `sha256`, and a
/// projection that ends up on the CURRENT file rather than the recorded one.
///
/// It also retires a manual device-probe step: "edit the boot file, then send from
/// the phone and check the session picked up the edit" is now CI.
#[tokio::test]
async fn both_boot_writers_land_on_one_stream_and_the_turn_reads_the_current_file() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let recorder = Arc::new(RecordingProvider::new(ScriptedProvider::new(vec!["ok"])));
    let (addr, sessions, root) = spawn_with_boot(
        &data_dir,
        recorder.clone(),
        "boot.md",
        "# first boot content",
        Some("boot.md"),
    )
    .await;

    let client = reqwest::Client::new();
    let id = create_instance(&client, addr).await;

    // Writer one: the create path, no hand-appending anywhere in this test.
    let at_create = sessions.store().read_from(&id, 0).unwrap();
    let boots: Vec<&EventEnvelope> = at_create.iter().filter(|e| e.event_type == "boot").collect();
    assert_eq!(boots.len(), 1, "create appends exactly one boot event: {at_create:?}");
    assert_eq!(boot_file(boots[0], "path"), "boot.md");
    assert_eq!(
        boot_file(boots[0], "sha256"),
        armillary_engine::hash::sha256_hex(b"# first boot content")
    );

    // The workspace edits its boot file, the way a real one does between sessions.
    std::fs::write(root.join("boot.md"), "# second boot content").unwrap();

    client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "hi", "clientKey": "c1" }))
        .send()
        .await
        .unwrap();
    wait_for_assistant_message(&sessions, &id, 1).await;

    // Writer two: drift recovery, on the same stream, appending rather than
    // rewriting — P-1's append-only truth, with the projection's last-boot-wins
    // fold deciding which one is current.
    let events = sessions.store().read_from(&id, 0).unwrap();
    let boots: Vec<&EventEnvelope> = events.iter().filter(|e| e.event_type == "boot").collect();
    assert_eq!(boots.len(), 2, "both writers must be on the stream: {events:?}");
    assert_eq!(
        boot_file(boots[0], "path"), boot_file(boots[1], "path"),
        "both writers record the same declared path spelling"
    );
    assert_ne!(
        boot_file(boots[0], "sha256"), boot_file(boots[1], "sha256"),
        "the re-record must carry a DIFFERENT sha — otherwise it is not re-reading disk"
    );
    assert_eq!(
        boot_file(boots[1], "sha256"),
        armillary_engine::hash::sha256_hex(b"# second boot content")
    );

    // And the system prompt the provider was actually handed is the current file.
    let last_turn = recorder.last_turn.lock().unwrap().clone().expect("a turn was recorded");
    assert_eq!(last_turn.system.as_deref(), Some("# second boot content"));

    let assistant = events.iter().find(|e| e.event_type == "assistant_message").unwrap();
    assert_eq!(assistant.data["interrupted"], false, "the turn completes normally");
}

// --- the failure event names the INSTANCE's model, not the engine's ---

/// The assertion that catches a resolver wired only halfway: a failure event
/// must name the model the INSTANCE asked for, not the one the engine booted
/// with. Without this, a turn that fails under a per-instance model reports
/// the engine default and the log lies about which pilot failed — the exact
/// case decision 3 makes routine, since an unpilotable model's whole report
/// IS its failure event.
#[tokio::test]
async fn a_failed_turn_names_the_instances_model_not_the_engines() {
    // Engine default is "scripted" (see `model_config()`); the instance asks
    // for something else and has no key for it — real `KeyedProviders`, both
    // keys absent, so every turn fails with `no_api_key`.
    let data_dir = tempfile::tempdir().unwrap().keep();
    let (addr, sessions) = spawn_keyless(&data_dir).await;
    let client = reqwest::Client::new();

    let id = create_instance_with(&client, addr, Some("tycho"), Some("zen/deepseek-v4-flash")).await;

    client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "hello", "clientKey": "c1" }))
        .send()
        .await
        .unwrap();

    wait_for_assistant_message(&sessions, &id, 1).await;
    let events = sessions.store().read_from(&id, 0).unwrap();

    let failure = events
        .iter()
        .rev()
        .find(|e| e.event_type == "assistant_message")
        .expect("the turn must have produced a failure-shaped assistant_message");
    assert_eq!(failure.data["error"], "no_api_key");
    assert_eq!(failure.data["model"], "zen/deepseek-v4-flash");
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
