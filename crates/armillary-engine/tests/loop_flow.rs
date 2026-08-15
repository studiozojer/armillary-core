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
    principals::{hash_token, write_principal, Grant, Principal},
    projection::{ContentBlock, ModelTurn},
    provider::{self, ModelProvider, ProviderError, ScriptedProvider, TurnOutcome},
    sessions::Sessions,
    state::{AppState, ModelConfig},
};
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, watch};

const READ_TIMEOUT: Duration = Duration::from_secs(2);

fn model_config() -> ModelConfig {
    ModelConfig {
        model: "scripted".to_string(),
    }
}

/// The bearer token every `reqwest::Client` built by `authed_client` presents.
/// Fixed, mirroring `tests/repos.rs`'s `TEST_TOKEN` and `tests/routes.rs`'s
/// own copy: nothing in this file asserts anything about the token itself,
/// only that instance creation and the loop's own POSTs now need SOME valid
/// credential (2026-08-07, Task 8: instance lifecycle gated on
/// authentication).
const TEST_TOKEN: &str = "test-fixed-token-for-tests-loop-flow-rs-2026-08-07";

/// A fresh registry directory holding one principal that authenticates as
/// `TEST_TOKEN`. Mirrors `tests/repos.rs`'s `full_grant_registry` — this file
/// never calls `auth::require`, so the grants themselves are inert, but a
/// full set keeps this fixture identical in shape to the other two suites
/// rather than inventing a "why does this one have fewer grants" question.
fn full_grant_registry() -> PathBuf {
    let dir = tempfile::tempdir().unwrap().keep();
    write_principal(
        &dir,
        &Principal {
            name: "test-client".to_string(),
            token_hash: hash_token(TEST_TOKEN),
            grants: vec![Grant::Sync, Grant::Push],
            minted: "2026-08-07T00:00:00Z".to_string(),
        },
    )
    .unwrap();
    dir
}

/// A `reqwest::Client` that presents `TEST_TOKEN` on every request it sends,
/// GET included (harmless — reads stay open, R1). Centralizing the header
/// here, rather than adding it call-by-call to this file's ~30 `.post(...)`
/// sites, is the same shape Task 7 used for `tests/repos.rs`'s shared POST
/// helpers: one place carries the credential, no assertion anywhere changes.
fn authed_client() -> reqwest::Client {
    client_presenting(TEST_TOKEN)
}

/// `authed_client`, for some token other than this file's default — the two
/// enrolled devices the gate tests need are two tokens, not two clients built
/// two different ways.
fn client_presenting(token: &str) -> reqwest::Client {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::AUTHORIZATION,
        reqwest::header::HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
    );
    reqwest::Client::builder().default_headers(headers).build().unwrap()
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
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
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
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
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
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
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
    let client = authed_client();
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

// --- Task 2: `agentTools` on the wire ---

/// (a) A valid `agentTools` list is accepted and the 200-path is unchanged
/// otherwise: same 201, same receipt shape, same seq numbering as the plain
/// send above. And — the point of routing consent through the spawned
/// turn's own arguments rather than the append — the durable `user_message`
/// carries `text`/`clientKey` only; consent never reaches the log.
#[tokio::test]
async fn send_with_a_valid_agent_tools_list_is_accepted_and_the_200_path_is_unchanged() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(ScriptedProvider::new(vec!["hi"]));
    let (addr, sessions) = spawn(&data_dir, provider).await;
    let client = authed_client();
    let id = create_instance(&client, addr).await;

    let response = client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "hello", "clientKey": "c1", "agentTools": ["commit"] }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::CREATED);
    let receipt: serde_json::Value = response.json().await.unwrap();
    assert_eq!(receipt["seq"], 3);
    assert!(receipt["id"].as_str().is_some_and(|s| !s.is_empty()));

    let events = sessions.store().read_from(&id, 0).unwrap();
    let user_ev = events.iter().find(|e| e.event_type == "user_message").unwrap();
    assert_eq!(user_ev.data["clientKey"], "c1");
    // Consent is per-request state, never a durable fact (brief, Task 2):
    // the append this route owns must not gain a new key just because the
    // request carried one.
    let keys = user_ev.data.as_object().unwrap();
    assert_eq!(keys.len(), 2, "{:?}", user_ev.data);
    assert!(!keys.contains_key("agentTools"), "{:?}", user_ev.data);
    assert!(!keys.contains_key("agent_tools"), "{:?}", user_ev.data);
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
    let client = authed_client();
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

    // `begin_turn` broadcasts a `turn_started` transient before
    // `session_ops::send` appends the durable echo (Task 2) — filter for the
    // type under test rather than assume it is the very next frame.
    let echo = loop {
        let (event, data) = sub.next_frame().await;
        assert_eq!(event, "envelope");
        let json: serde_json::Value = serde_json::from_str(&data).unwrap();
        if json["type"] == "turn_started" {
            assert_eq!(json["seq"], 0, "I-4: transients are seq 0");
            continue;
        }
        break json;
    };
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
        // The title daemon's durable events (thread "daemon-*") ride the
        // same stream before the operator's rounds — they are not the type
        // under test here, and the SSE contract deliberately carries them.
        if json["thread"].as_str().is_some_and(|t| t.starts_with("daemon-")) {
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

// --- (b') the whole-path claim: a live subscriber hears the turn's
// open/close bracket, and `attach` reports the claim while it is genuinely
// held ---

/// Task 4, test 1 (brief step 1): proves `begin_turn`/`end_turn`'s
/// transients (Tasks 1-2) actually reach a subscriber wired through the real
/// `Sessions` a running server holds — `turn_started` first, the turn's own
/// work between, `turn_ended` last. Unlike (b) above this reads straight off
/// `Sessions::subscribe_live` rather than the SSE wire, because the claim is
/// about `Sessions` broadcasting correctly to any subscriber, and the SSE
/// framing is already covered by (b).
#[tokio::test]
async fn a_subscriber_sees_the_turn_open_and_close_around_its_work() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(
        ScriptedProvider::new(vec!["Hel", "Hello", "Hello there"])
            .with_pause(Duration::from_millis(20)),
    );
    let (addr, sessions) = spawn(&data_dir, provider).await;
    let client = authed_client();
    let id = create_instance(&client, addr).await;

    let mut rx = sessions.subscribe_live(&id);

    let response = client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "hi", "clientKey": "c1" }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::CREATED);

    let mut seen: Vec<String> = Vec::new();
    // Bounded: collect until turn_ended or the budget runs out, so a
    // regression fails as a wrong sequence rather than hanging forever.
    for _ in 0..200 {
        match tokio::time::timeout(READ_TIMEOUT, rx.recv()).await {
            Ok(Ok(ev)) => {
                let done = ev.event_type == "turn_ended";
                seen.push(ev.event_type.clone());
                if done {
                    break;
                }
            }
            _ => break,
        }
    }

    assert_eq!(
        seen.first().map(String::as_str),
        Some("turn_started"),
        "the claim must be the first thing a subscriber hears, got {seen:?}"
    );
    // This only proves nothing *collected* comes after `turn_ended` — the
    // loop above breaks the moment it sees one, so it stops looking rather
    // than keeps watching and finding nothing follows. The claim that the
    // release is genuinely last is true (`_end_turn_guard` is declared first
    // in `run_turn`, so it drops last), just not established by this assertion.
    assert_eq!(
        seen.last().map(String::as_str),
        Some("turn_ended"),
        "the release must be the last, got {seen:?}"
    );
    assert!(
        seen.iter().any(|t| t == "assistant_message"),
        "the turn's actual work must sit between them, got {seen:?}"
    );
}

/// Task 4, test 2 (scope amendment): the load-bearing claim the design
/// exists for — a client that attaches mid-turn, never having seen
/// `turn_started`, still learns the turn is running from `turnInProgress` on
/// the attach payload; and once the turn ends, a fresh attach reports it
/// cleared. Goes through the real HTTP `attach` handler (`GET
/// /instances/{id}`, this file's existing `attach` helper) and asserts on
/// the wire's own camelCase field name, so the serialization is pinned
/// end-to-end and not just at `instance_from_first_event`'s unit level.
///
/// Held open deterministically: rather than sleeping and hoping the turn is
/// still running, this subscribes to the same live channel test 1 uses and
/// waits for the `turn_started` transient before calling `attach` — that
/// transient is broadcast by `Sessions::begin_turn` under the very lock that
/// fills the turn slot (see `sessions.rs`), so observing it is proof the
/// slot is occupied at that instant, not a guess about scheduling. The
/// scripted provider's multi-fragment pause then gives a wide, deterministic
/// margin (at least 3 * 40ms of remaining script) for the attach round-trip
/// to land before the turn can finish. The "after" attach is gated the same
/// way, off `turn_ended`, so it can only run once the slot has actually
/// cleared.
#[tokio::test]
async fn attach_reports_turn_in_progress_while_a_turn_is_mid_flight_then_clears_it() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(
        ScriptedProvider::new(vec!["a", "ab", "abc", "abcd"])
            .with_pause(Duration::from_millis(40)),
    );
    let (addr, sessions) = spawn(&data_dir, provider).await;
    let client = authed_client();
    let id = create_instance(&client, addr).await;

    let mut rx = sessions.subscribe_live(&id);

    let response = client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "hi", "clientKey": "c1" }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::CREATED);

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let ev = rx.recv().await.unwrap();
            if ev.event_type == "turn_started" {
                return;
            }
        }
    })
    .await
    .expect("timed out waiting for turn_started");

    let mid_flight = attach(&client, addr, &id).await;
    assert_eq!(
        mid_flight["instance"]["turnInProgress"], true,
        "attach must report turnInProgress while the turn is genuinely mid-flight: {mid_flight:?}"
    );

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let ev = rx.recv().await.unwrap();
            if ev.event_type == "turn_ended" {
                return;
            }
        }
    })
    .await
    .expect("timed out waiting for turn_ended");

    let after = attach(&client, addr, &id).await;
    assert_eq!(
        after["instance"]["turnInProgress"], false,
        "attach must report the turn cleared once it has ended: {after:?}"
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
    let client = authed_client();
    let id = create_instance(&client, addr).await;

    client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "hi", "clientKey": "c1" }))
        .send()
        .await
        .unwrap();

    // The title daemon's own provider call runs the SAME paused script to
    // completion before the operator's round begins, and its cancel channel
    // is private — an interrupt landing during the daemon's run would leave
    // the operator's round cancelled before its first fragment, recording an
    // EMPTY partial. Wait for the daemon's pulse (its always-written last
    // event) so the sleep below lands mid-way through the operator's script.
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let events = sessions.store().read_from(&id, 0).unwrap();
            if events.iter().any(|e| e.event_type == "daemon_pulse") {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the daemon's pulse never landed");

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
    let client = authed_client();
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
    let client = authed_client();
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
    let client = authed_client();
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
        head_before, 6,
        "instance_created, composition, user_message, instance_renamed, \
         daemon_pulse, assistant_message — the title daemon's two events \
         are part of the log this test proves durable"
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
    assert_eq!(replayed.len(), 6);
    assert_eq!(replayed[0].event_type, "instance_created");
    assert_eq!(replayed[1].event_type, "composition");
    assert_eq!(replayed[2].event_type, "user_message");
    assert_eq!(replayed[2].data["text"], "hello");
    assert_eq!(replayed[3].event_type, "instance_renamed");
    assert_eq!(replayed[4].event_type, "daemon_pulse");
    assert_eq!(replayed[4].data["disposition"], "updated");
    assert_eq!(replayed[5].event_type, "assistant_message");
    assert_eq!(replayed[5].data["text"], "done");

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

    let client = authed_client();
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
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: None,
    };
    tokio::spawn(async move {
        let _ = axum::serve(listener, app(state)).await;
    });

    let client = authed_client();
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

    let client = authed_client();
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
    let client = authed_client();

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
    let client = authed_client();

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
    let client = authed_client();

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
    let client = authed_client();

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
async fn archive_on_an_unknown_instance_is_404_unknown_instance() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(ScriptedProvider::new(vec!["unused"]));
    let (addr, _sessions) = spawn(&data_dir, provider).await;
    let client = authed_client();

    let response = client
        .post(format!("http://{addr}/instances/does-not-exist/archive"))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
    assert_eq!(response.text().await.unwrap(), "unknown_instance");
}

// --- D1 pinned end to end: archive flips the listing flag and bars nothing ---

#[tokio::test]
async fn archive_flips_the_listing_flag_and_bars_nothing() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(ScriptedProvider::new(vec!["still here"]));
    let (addr, _sessions) = spawn(&data_dir, provider).await;
    let client = authed_client();

    let created: serde_json::Value = client
        .post(format!("http://{addr}/instances"))
        .json(&serde_json::json!({}))
        .send().await.unwrap()
        .json().await.unwrap();
    let id = created["id"].as_str().unwrap().to_string();
    assert_eq!(created["archived"], serde_json::json!(false));

    let response = client
        .post(format!("http://{addr}/instances/{id}/archive"))
        .send().await.unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);

    let listed: Vec<serde_json::Value> = client
        .get(format!("http://{addr}/instances"))
        .send().await.unwrap().json().await.unwrap();
    let mine = listed.iter().find(|i| i["id"] == created["id"]).unwrap();
    assert_eq!(mine["archived"], serde_json::json!(true));

    // D1 / A-3: an archived instance still takes a send — 201, not a refusal.
    let send = client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "still alive", "clientKey": "c1" }))
        .send().await.unwrap();
    assert_eq!(send.status(), reqwest::StatusCode::CREATED);

    // Unarchive restores the default listing state.
    let response = client
        .post(format!("http://{addr}/instances/{id}/unarchive"))
        .send().await.unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);

    let listed: Vec<serde_json::Value> = client
        .get(format!("http://{addr}/instances"))
        .send().await.unwrap().json().await.unwrap();
    let mine = listed.iter().find(|i| i["id"] == created["id"]).unwrap();
    assert_eq!(mine["archived"], serde_json::json!(false));
}

// --- Task 4: the gate, at the door. `offered = the base seven + (agent_tools
// ∩ caller-grants ∩ manifest)`, resolved fresh per turn. These four drive the
// WHOLE path a phone drives — `agentTools` over the wire, a principal recorded
// at creation, a registry and a manifest on disk — and assert on the tool
// definitions that actually reached the provider, which is the only place the
// offer is observable from outside the loop. ---

/// The base set every turn is handed regardless of consent — `tools::registry()`
/// in its offer order. Spelled out rather than counted so a test failure names
/// which one went missing, and so a new base tool has to be admitted here on
/// purpose.
const BASE_TOOLS: [&str; 8] = [
    "get_composition",
    "list_directory",
    "read_file",
    "find_files",
    "search",
    "write_file",
    "edit_file",
    "inspect_daemons",
];

/// A registry directory holding one principal — the same `test-client` every
/// request in this file authenticates as — with exactly `grants`. The whole
/// point of the gate is that this set and the manifest disagree sometimes, so
/// unlike `full_grant_registry` this one is a knob.
fn registry_granting(grants: Vec<Grant>) -> PathBuf {
    let dir = tempfile::tempdir().unwrap().keep();
    write_principal(
        &dir,
        &Principal {
            name: "test-client".to_string(),
            token_hash: hash_token(TEST_TOKEN),
            grants,
            minted: "2026-08-09T00:00:00Z".to_string(),
        },
    )
    .unwrap();
    dir
}

/// `spawn`, with the two things the gate reads made explicit: the registry the
/// caller's grants come from, and a workspace manifest whose `[router]` keys
/// are the ceiling.
async fn spawn_gated(
    data_dir: &Path,
    provider: Arc<dyn ModelProvider>,
    registry_dir: PathBuf,
    manifest: &str,
) -> (SocketAddr, Arc<Sessions>) {
    let store = LogStore::open(data_dir).unwrap();
    let sessions = Arc::new(Sessions::new(store));
    let root = tempfile::tempdir().unwrap().keep();
    std::fs::write(root.join("modules.toml"), manifest).unwrap();

    let state = AppState {
        root: root.canonicalize().unwrap(),
        sessions: sessions.clone(),
        model: model_config(),
        providers: provider::fixed(provider),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
        registry_dir,
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

/// Records the tool NAMES each round was offered, then answers with prose so
/// the turn ends in one round. `RecordingProvider` above captures the turn;
/// this captures the other half of the same `TurnRequest` — what the model was
/// allowed to reach for.
struct OfferProbe {
    offered: Mutex<Vec<Vec<String>>>,
}

impl OfferProbe {
    fn new() -> Arc<Self> {
        Arc::new(OfferProbe {
            offered: Mutex::new(Vec::new()),
        })
    }

    /// The first round's offer — the one assembled from the send's own
    /// consent.
    fn first_round(&self) -> Vec<String> {
        self.offered
            .lock()
            .unwrap()
            .first()
            .cloned()
            .expect("the provider was never called")
    }
}

#[async_trait::async_trait]
impl ModelProvider for OfferProbe {
    async fn run_turn(
        &self,
        req: armillary_engine::provider::TurnRequest,
        _sink: mpsc::Sender<String>,
        _cancel: watch::Receiver<bool>,
    ) -> Result<TurnOutcome, ProviderError> {
        // The title daemon's pre-round offers no tools at all — it is a
        // model call but not an OFFER, so recording it would shift every
        // operator round's index by one per turn. An operator round always
        // carries at least the base seven, so empty-tools is an unambiguous
        // discriminator.
        if !req.tools.is_empty() {
            self.offered
                .lock()
                .unwrap()
                .push(req.tools.iter().map(|t| t.name.to_string()).collect());
        }
        Ok(TurnOutcome {
            text: "noted".to_string(),
            blocks: vec![ContentBlock::text("noted")],
            stop_reason: Some("end_turn".to_string()),
            stopped: false,
            model: "offer-probe".to_string(),
        })
    }
}

/// One send carrying `agent_tools`, run to completion; returns what the first
/// round was offered.
async fn offered_for(registry_dir: PathBuf, manifest: &str, agent_tools: serde_json::Value) -> Vec<String> {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let probe = OfferProbe::new();
    let (addr, sessions) = spawn_gated(&data_dir, probe.clone(), registry_dir, manifest).await;
    let client = authed_client();
    let id = create_instance(&client, addr).await;

    client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({ "text": "commit that for me", "clientKey": "c1", "agentTools": agent_tools }))
        .send()
        .await
        .unwrap();
    wait_for_assistant_message(&sessions, &id, 1).await;

    probe.first_round()
}

const ALL_THREE_GRANTED: &str = "[router]\nsync = true\npush = true\ncommit = true\n";

#[tokio::test]
async fn consent_a_grant_and_the_manifest_agreeing_puts_the_verb_on_the_turn() {
    // (a) The whole law in its permitting direction: the device consented to
    // `commit`, its principal holds `commit`, and the workspace grants it — so
    // the turn is offered `commit_repo`. And ONLY that one: consent is an
    // intersection, not a switch, so the two verbs the send never named stay
    // off even though grant and manifest would both have allowed them.
    let offered = offered_for(
        registry_granting(vec![Grant::Sync, Grant::Push, Grant::Commit]),
        ALL_THREE_GRANTED,
        serde_json::json!(["commit"]),
    )
    .await;

    assert!(offered.contains(&"commit_repo".to_string()), "{offered:?}");
    assert!(!offered.contains(&"sync_repo".to_string()), "{offered:?}");
    assert!(!offered.contains(&"push_repo".to_string()), "{offered:?}");
    for base in BASE_TOOLS {
        assert!(offered.contains(&base.to_string()), "{base} left the base set: {offered:?}");
    }
    // E-4: the git verb lands AFTER the base set, never interleaved, so the
    // cached prompt prefix is the same bytes whether or not a turn is granted.
    assert_eq!(
        offered[..BASE_TOOLS.len()],
        BASE_TOOLS.map(str::to_string),
        "{offered:?}"
    );
}

#[tokio::test]
async fn a_device_consenting_to_what_it_was_never_granted_is_offered_nothing() {
    // (b) The client can narrow itself and never widen itself (D3). This send
    // asks for `commit` with a principal holding only sync and push — an
    // inflated consent, which is what a stale or lying client sends.
    let offered = offered_for(
        registry_granting(vec![Grant::Sync, Grant::Push]),
        ALL_THREE_GRANTED,
        serde_json::json!(["commit"]),
    )
    .await;

    assert!(!offered.contains(&"commit_repo".to_string()), "{offered:?}");
    assert_eq!(offered, BASE_TOOLS.map(str::to_string), "{offered:?}");
}

#[tokio::test]
async fn a_manifest_that_withholds_commit_withholds_it_from_a_fully_granted_device() {
    // (c) The ceiling, one edit away from denying — the property that keeps
    // the manifest keys from being decorative (`routes::repos`'s own argument
    // for them). Consent and grant both say yes; the workspace says no.
    let offered = offered_for(
        registry_granting(vec![Grant::Sync, Grant::Push, Grant::Commit]),
        "[router]\nsync = true\npush = true\ncommit = false\n",
        serde_json::json!(["commit"]),
    )
    .await;

    assert!(!offered.contains(&"commit_repo".to_string()), "{offered:?}");
    assert_eq!(offered, BASE_TOOLS.map(str::to_string), "{offered:?}");
}

#[tokio::test]
async fn consenting_to_sync_puts_sync_repo_on_the_turn_and_neither_of_the_others() {
    // The rows themselves, pinned. Every other test here consents to `commit`
    // and asks after `commit_repo` — so `sync` and `push` were only ever
    // observed being WITHHELD, and a table that swapped their two rows
    // (`sync_repo → Push`, `push_repo → Sync`) passed the lot. A grant that
    // opens the wrong verb is not a smaller defect than a grant that opens
    // nothing.
    //
    // Sync-only in all three terms, so the two verbs left off are left off by
    // the grant and by the consent both.
    let offered = offered_for(
        registry_granting(vec![Grant::Sync]),
        ALL_THREE_GRANTED,
        serde_json::json!(["sync"]),
    )
    .await;

    assert!(offered.contains(&"sync_repo".to_string()), "{offered:?}");
    assert!(!offered.contains(&"push_repo".to_string()), "{offered:?}");
    assert!(!offered.contains(&"commit_repo".to_string()), "{offered:?}");
}

#[tokio::test]
async fn a_send_carrying_no_consent_is_offered_none_of_the_three() {
    // (d) Fail-closed, and the case every OTHER test in this file exercises
    // by accident: absent consent is not "whatever the device could do", it is
    // nothing. A grant and a manifest that both permit all three still put no
    // verb on a turn nobody consented to.
    let offered = offered_for(
        registry_granting(vec![Grant::Sync, Grant::Push, Grant::Commit]),
        ALL_THREE_GRANTED,
        serde_json::Value::Null,
    )
    .await;

    assert_eq!(offered, BASE_TOOLS.map(str::to_string), "{offered:?}");
}

// --- The gate spends the SENDER's authority. Nothing scopes an instance to
// the device that created it — every enrolled device can send into every
// window — so which device the gate looks up is the whole of whether consent
// means anything. These two drive it over the wire with two real tokens. ---

const TOKEN_A: &str = "test-token-device-a-loop-flow-rs-2026-08-09";
const TOKEN_B: &str = "test-token-device-b-loop-flow-rs-2026-08-09";

/// A registry holding two enrolled devices with different tokens and
/// different grants. `registry_granting`'s single principal cannot express
/// "one device opened the window, another one sent into it", which is the
/// only shape these two tests are about.
fn registry_of_two(device_a: Vec<Grant>, device_b: Vec<Grant>) -> PathBuf {
    let dir = tempfile::tempdir().unwrap().keep();
    for (name, token, grants) in [
        ("device-a", TOKEN_A, device_a),
        ("device-b", TOKEN_B, device_b),
    ] {
        write_principal(
            &dir,
            &Principal {
                name: name.to_string(),
                token_hash: hash_token(token),
                grants,
                minted: "2026-08-09T00:00:00Z".to_string(),
            },
        )
        .unwrap();
    }
    dir
}

/// Calls `write_file` once, then speaks. The smallest turn that produces a
/// `file_changed` — the one effect event observable from here without a git
/// fixture, and the one that carries the acting principal.
struct WriteThenSpeak {
    calls: Mutex<usize>,
}

#[async_trait::async_trait]
impl ModelProvider for WriteThenSpeak {
    async fn run_turn(
        &self,
        _req: armillary_engine::provider::TurnRequest,
        _sink: mpsc::Sender<String>,
        _cancel: watch::Receiver<bool>,
    ) -> Result<TurnOutcome, ProviderError> {
        // The title daemon's pre-round offers no tools — a model can only
        // call what a round offers, so answer it with prose and keep it out
        // of the round count. Without this, the daemon consumes round 1 and
        // the scripted tool_use never reaches the operator's turn.
        if _req.tools.is_empty() {
            return Ok(TurnOutcome {
                text: "a title".to_string(),
                blocks: vec![ContentBlock::text("a title")],
                stop_reason: Some("end_turn".to_string()),
                stopped: false,
                model: "write-then-speak".to_string(),
            });
        }
        let round = {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            *calls
        };
        if round == 1 {
            return Ok(TurnOutcome {
                text: String::new(),
                blocks: vec![ContentBlock::ToolUse {
                    id: "toolu_w1".to_string(),
                    name: "write_file".to_string(),
                    input: serde_json::json!({
                        "path": "notes.md",
                        "content": "from the sending device\n",
                    }),
                }],
                stop_reason: Some("tool_use".to_string()),
                stopped: false,
                model: "write-then-speak".to_string(),
            });
        }
        Ok(TurnOutcome {
            text: "done".to_string(),
            blocks: vec![ContentBlock::text("done")],
            stop_reason: Some("end_turn".to_string()),
            stopped: false,
            model: "write-then-speak".to_string(),
        })
    }
}

async fn send_consenting(
    client: &reqwest::Client,
    addr: SocketAddr,
    id: &str,
    agent_tools: serde_json::Value,
) {
    client
        .post(format!("http://{addr}/instances/{id}/send"))
        .json(&serde_json::json!({
            "text": "commit that for me",
            "clientKey": "c1",
            "agentTools": agent_tools,
        }))
        .send()
        .await
        .unwrap();
}

#[tokio::test]
async fn the_gate_looks_up_the_sending_device_not_the_one_that_created_the_instance() {
    // device-a holds `commit` and opens the window; device-b holds nothing and
    // sends `agentTools: ["commit"]` into it. If the gate read the CREATOR's
    // grants, that send would be handed `commit_repo` — a device granted
    // nothing spending a device granted everything, by picking the right
    // window to shout into.
    //
    // The second send is the control, and it is what makes the first
    // assertion mean anything: the SAME instance, the SAME manifest, the SAME
    // consent, from the device that actually holds the grant — and there the
    // verb does land on the turn. Without it, "offered nothing" would be
    // satisfied by a fixture that could never offer anything.
    let data_dir = tempfile::tempdir().unwrap().keep();
    let probe = OfferProbe::new();
    let (addr, sessions) = spawn_gated(
        &data_dir,
        probe.clone(),
        registry_of_two(vec![Grant::Commit], Vec::new()),
        ALL_THREE_GRANTED,
    )
    .await;

    let device_a = client_presenting(TOKEN_A);
    let device_b = client_presenting(TOKEN_B);
    let id = create_instance(&device_a, addr).await;

    send_consenting(&device_b, addr, &id, serde_json::json!(["commit"])).await;
    wait_for_assistant_message(&sessions, &id, 1).await;

    send_consenting(&device_a, addr, &id, serde_json::json!(["commit"])).await;
    wait_for_assistant_message(&sessions, &id, 2).await;

    let offers = probe.offered.lock().unwrap();
    assert_eq!(
        offers[0],
        BASE_TOOLS.map(str::to_string),
        "a zero-grant sender is offered the base set and nothing else: {:?}",
        offers[0]
    );
    assert!(
        offers[1].contains(&"commit_repo".to_string()),
        "the granted device's own send must still be offered it: {:?}",
        offers[1]
    );
}

#[tokio::test]
async fn an_effect_names_the_device_that_sent_the_turn_not_the_one_that_created_the_instance() {
    // Attribution follows the same name the gate does, and it has to: a record
    // naming the creator would credit the authority to a device that spent
    // none of it. `write_file` needs no consent at all, so this isolates the
    // attribution half — nothing here is gated, and the name still has to be
    // the sender's.
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(WriteThenSpeak {
        calls: Mutex::new(0),
    });
    let (addr, sessions) = spawn_gated(
        &data_dir,
        provider,
        registry_of_two(vec![Grant::Commit], vec![Grant::Commit]),
        ALL_THREE_GRANTED,
    )
    .await;

    let device_a = client_presenting(TOKEN_A);
    let device_b = client_presenting(TOKEN_B);
    let id = create_instance(&device_a, addr).await;

    send_consenting(&device_b, addr, &id, serde_json::Value::Null).await;
    wait_for_assistant_message(&sessions, &id, 2).await;

    let events = sessions.store().read_from(&id, 0).unwrap();
    let changed = events
        .iter()
        .find(|e| e.event_type == "file_changed")
        .expect("the write must have been recorded");
    assert_eq!(
        changed.actor.principal.as_ref().map(|p| p.name.as_str()),
        Some("device-b"),
        "the effect names whoever sent the turn that produced it"
    );
}

#[tokio::test]
async fn evict_with_an_event_id_not_in_the_stream_is_404_unknown_event() {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let provider = Arc::new(ScriptedProvider::new(vec!["ok"]));
    let (addr, _sessions) = spawn(&data_dir, provider).await;
    let client = authed_client();
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
