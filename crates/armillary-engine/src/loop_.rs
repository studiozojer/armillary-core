//! The loop: one turn, start to finish — the 5% of the harness the other
//! 95% (log, projection, provider trait, subscribe, instance routes) was
//! built for.
//!
//! `run_turn` is spawned by `routes::session_ops::send` AFTER that handler
//! has already appended the durable `user_message` and returned its 201
//! receipt (P-2: the injected message is recorded first; the phone's bubble
//! reconciles even if everything below this point fails). Nothing in this
//! module has an HTTP caller left to answer — every observable effect is
//! either a durable `Sessions::append` or a transient
//! `Sessions::broadcast_transient`, and a failure that can't be handed back
//! to a request is `eprintln!`'d (I-5: never silently swallowed) rather than
//! invented into a status code nobody is listening for.
//!
//! Module named `loop_` (not `r#loop`) per the task brief: `r#loop` reads as
//! noise at every call site, and `loop_` with this doc comment is the
//! quieter spelling of the same module.

use crate::log::envelope::{Actor, EventEnvelope, Role};
use crate::projection::{project_context, ProjectionError};
use crate::provider::{ProviderError, TurnOutcome};
use crate::sessions::{NewEvent, SessionError, Sessions};
use crate::state::SharedState;
use std::io;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::{mpsc, watch};

/// Runs blocking log I/O off the async worker thread — same rationale as
/// `blocking::run` (see that module's doc: a blocking `std::fs` call parks a
/// tokio worker thread, starving whatever else shares it, e.g. a live SSE
/// subscriber). Shaped for `io::Result` rather than an HTTP response, since
/// nothing in this module has an HTTP caller left to answer.
async fn spawn_blocking_io<T, F>(f: F) -> io::Result<T>
where
    F: FnOnce() -> io::Result<T> + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(result) => result,
        Err(_) => Err(io::Error::other("blocking log I/O task panicked")),
    }
}

async fn read_all(sessions: &Arc<Sessions>, stream: &str) -> io::Result<Vec<EventEnvelope>> {
    let sessions = sessions.clone();
    let stream = stream.to_string();
    spawn_blocking_io(move || sessions.store().read_from(&stream, 0)).await
}

async fn append_blocking(
    sessions: &Arc<Sessions>,
    stream: &str,
    ev: NewEvent,
) -> Result<EventEnvelope, SessionError> {
    let sessions = sessions.clone();
    let stream = stream.to_string();
    match tokio::task::spawn_blocking(move || sessions.append(&stream, ev)).await {
        Ok(result) => result,
        Err(_) => Err(SessionError::Log(io::Error::other("append blocking task panicked"))),
    }
}

/// Clears the turn slot on drop — success, interruption, an early return on
/// failure, or a panic unwinding through this function, alike. This is the
/// "finally-shaped block" the brief asks for, built from `std::ops::Drop`
/// rather than a new dependency: nothing but a normal return or an
/// unwinding panic can skip a `Drop` impl, which a hand-rolled "clear it at
/// the bottom" could quietly miss on any of this function's several early
/// returns.
struct EndTurnGuard {
    sessions: Arc<Sessions>,
    stream: String,
}

impl Drop for EndTurnGuard {
    fn drop(&mut self) {
        self.sessions.end_turn(&self.stream);
    }
}

/// The operator label a transient/durable assistant event's `actor.instance`
/// carries: the operator this instance was created with, or `"dispatcher"`
/// when it has none (a bare/operator-less instance is still the dispatcher's
/// own voice, not an anonymous one).
fn operator_label(events: &[EventEnvelope]) -> String {
    events
        .iter()
        .find(|e| e.event_type == "instance_created")
        .and_then(|e| e.data.get("operator"))
        .and_then(|v| v.as_str())
        .unwrap_or("dispatcher")
        .to_string()
}

fn assistant_actor(operator: &str) -> Actor {
    Actor {
        role: Role::Operator,
        instance: Some(operator.to_string()),
    }
}

fn now_rfc3339() -> String {
    humantime::format_rfc3339_millis(SystemTime::now()).to_string()
}

/// Builds one transient `assistant_delta` envelope — I-4: `seq` MUST be 0,
/// and the payload is a snapshot (`textSoFar`, not an increment), matching
/// exactly what `ModelProvider::run_turn`'s sink already hands this module.
fn transient_delta_envelope(stream: &str, operator: &str, generation: &str, text_so_far: &str) -> EventEnvelope {
    EventEnvelope {
        stream: stream.to_string(),
        id: uuid::Uuid::new_v4().to_string(),
        seq: 0,
        ts: now_rfc3339(),
        actor: assistant_actor(operator),
        event_type: "assistant_delta".to_string(),
        thread: None,
        parent: None,
        version: 1,
        cost: None,
        data: serde_json::json!({ "textSoFar": text_so_far, "generation": generation }),
    }
}

/// Machine codes named by the task brief for a provider failure. Not a
/// closed Rust enum on the wire (the assistant_message's `data.error` is a
/// plain string, matching I-2's untyped `data`), but centralized here so the
/// three shapes stay in one place.
fn machine_code_for_provider_error(e: &ProviderError) -> String {
    match e {
        ProviderError::NoApiKey => "no_api_key".to_string(),
        ProviderError::Http(_) => "provider_http".to_string(),
        ProviderError::Api { status, .. } => format!("provider_api_{status}"),
    }
}

/// Appends the durable, failure-shaped `assistant_message` the brief's step 6
/// decision names: `{ text: "", generation, interrupted: true, error: code }`
/// — the turn is over, the record says it ended abnormally, and the
/// projection's existing interrupted-marker covers rendering. No new event
/// type minted mid-sprint (ratified in the brief; revisit with
/// machine-verdicts).
///
/// `model` is the CONFIGURED model string (`state.model.model`), not one a
/// provider ever confirmed running — a failure path by definition has no
/// `TurnOutcome::model` to report, since it either never reached the
/// provider or the provider itself is what failed. Included anyway so the
/// success and failure shapes agree on which fields an `assistant_message`
/// always carries, rather than the failure shape silently being the one
/// case missing it.
async fn fail_turn(
    sessions: &Arc<Sessions>,
    stream: &str,
    operator: &str,
    generation: &str,
    code: &str,
    model: &str,
) {
    let ev = NewEvent {
        actor: assistant_actor(operator),
        event_type: "assistant_message".to_string(),
        data: serde_json::json!({
            "text": "",
            "generation": generation,
            "interrupted": true,
            "error": code,
            "model": model,
        }),
    };
    if let Err(e) = append_blocking(sessions, stream, ev).await {
        // I-5: never silently swallowed. There is no HTTP caller left to
        // hand this to — the send request already returned its 201 — so the
        // visible surface left is stderr.
        eprintln!(
            "log_write_failed appending a failure-shaped assistant_message (turn error {code:?}) \
             for stream {stream:?}: {e:?}"
        );
    }
}

/// Re-records a fresh `boot` event on `BootDrift`: re-reads the SAME `path`
/// the drifted event pointed to (reusing `projection::resolve_boot_path` so
/// the root-containment check is not re-derived at a second call site),
/// hashes the current bytes, and appends `{ path, sha256 }` with `actor:
/// system` — the disk content is re-affirmed as current truth. Returns the
/// brief's `boot_unreadable` machine code on any failure along the way (an
/// unreadable file, or the append itself failing — both leave the turn with
/// no system prompt it can trust, so both fold into the same named code).
async fn rerecord_boot(state: &SharedState, stream: &str, path: &str) -> Result<(), &'static str> {
    let root = state.root.clone();
    let path_owned = path.to_string();
    let resolved = tokio::task::spawn_blocking(move || {
        crate::projection::resolve_boot_path(&root, &path_owned)
    })
    .await
    .map_err(|_| "boot_unreadable")?
    .map_err(|_| "boot_unreadable")?;

    let bytes = tokio::fs::read(&resolved).await.map_err(|_| "boot_unreadable")?;
    let sha256 = crate::hash::sha256_hex(&bytes);

    let ev = NewEvent {
        actor: Actor {
            role: Role::System,
            instance: None,
        },
        event_type: "boot".to_string(),
        data: serde_json::json!({ "path": path, "sha256": sha256 }),
    };
    append_blocking(&state.sessions, stream, ev)
        .await
        .map_err(|_| "boot_unreadable")?;
    Ok(())
}

/// Runs one turn to completion (or interruption, or failure), then clears
/// the `TurnHandle` — always, via `EndTurnGuard`. Spawned by
/// `routes::session_ops::send`; `cancel_rx` is the receiver half of the
/// `watch` channel that route installed into `Sessions` before spawning
/// this, so `interrupt` (racing this task from another request) and this
/// task's own read of `cancel` are the same channel throughout.
pub async fn run_turn(state: SharedState, stream: String, generation: String, cancel_rx: watch::Receiver<bool>) {
    let _end_turn_guard = EndTurnGuard {
        sessions: state.sessions.clone(),
        stream: stream.clone(),
    };

    // project_context over the WHOLE log — `read_from(0)` and `read_from(1)`
    // are equivalent here (every persisted `seq` starts at 1; nothing durable
    // is ever seq 0 — I-4), so either spelling reads the full history.
    let events = match read_all(&state.sessions, &stream).await {
        Ok(events) => events,
        Err(e) => {
            eprintln!("failed to read stream {stream:?} for a turn: {e}");
            fail_turn(&state.sessions, &stream, "dispatcher", &generation, "boot_unreadable", &state.model.model).await;
            return;
        }
    };
    let operator = operator_label(&events);

    let turn = match project_context(&events, &state.root) {
        Ok(turn) => turn,
        Err(ProjectionError::BootDrift { path }) => {
            if let Err(code) = rerecord_boot(&state, &stream, &path).await {
                fail_turn(&state.sessions, &stream, &operator, &generation, code, &state.model.model).await;
                return;
            }
            let events = match read_all(&state.sessions, &stream).await {
                Ok(events) => events,
                Err(e) => {
                    eprintln!("failed to re-read stream {stream:?} after re-recording boot: {e}");
                    fail_turn(&state.sessions, &stream, &operator, &generation, "boot_unreadable", &state.model.model).await;
                    return;
                }
            };
            match project_context(&events, &state.root) {
                Ok(turn) => turn,
                Err(_) => {
                    fail_turn(&state.sessions, &stream, &operator, &generation, "boot_unreadable", &state.model.model).await;
                    return;
                }
            }
        }
        Err(ProjectionError::BootUnreadable { .. }) | Err(ProjectionError::Transient) => {
            fail_turn(&state.sessions, &stream, &operator, &generation, "boot_unreadable", &state.model.model).await;
            return;
        }
    };

    // Each sink snapshot -> a transient assistant_delta (I-4: seq 0, never
    // persisted). A separate task drains the channel concurrently with
    // `run_turn` below so the provider's sink sends never block on a slow
    // broadcast.
    let (tx, mut rx) = mpsc::channel::<String>(32);
    let relay_sessions = state.sessions.clone();
    let relay_stream = stream.clone();
    let relay_operator = operator.clone();
    let relay_generation = generation.clone();
    let relay = tokio::spawn(async move {
        while let Some(text_so_far) = rx.recv().await {
            let ev = transient_delta_envelope(&relay_stream, &relay_operator, &relay_generation, &text_so_far);
            relay_sessions.broadcast_transient(&relay_stream, ev);
        }
    });

    let outcome = state.provider.run_turn(turn, tx, cancel_rx).await;
    // `tx` was moved into `run_turn` and dropped when it returned, so the
    // relay's `rx.recv()` has already (or is about to) observe `None` —
    // awaiting it here just makes sure every transient it already queued is
    // broadcast before this function's caller (nothing — `tokio::spawn`) sees
    // this task end.
    let _ = relay.await;

    match outcome {
        Ok(TurnOutcome { text, stopped, model }) => {
            if stopped {
                // Step 5: the `interrupt` event is recorded BEFORE the
                // partial assistant_message — actor user, since a stop is
                // always user-initiated in this loop (there is no other
                // source of `cancel: true` yet).
                let interrupt_ev = NewEvent {
                    actor: Actor {
                        role: Role::User,
                        instance: None,
                    },
                    event_type: "interrupt".to_string(),
                    data: serde_json::json!({}),
                };
                if let Err(e) = append_blocking(&state.sessions, &stream, interrupt_ev).await {
                    eprintln!("log_write_failed appending interrupt for stream {stream:?}: {e:?}");
                }
            }
            // `interrupted` is always present (rather than omitted when
            // false) — one consistent shape for both branches, matching the
            // brief's "pick one and be consistent" allowance.
            let data = serde_json::json!({
                "text": text,
                "generation": generation,
                "interrupted": stopped,
                "model": model,
            });
            let assistant_ev = NewEvent {
                actor: assistant_actor(&operator),
                event_type: "assistant_message".to_string(),
                data,
            };
            if let Err(e) = append_blocking(&state.sessions, &stream, assistant_ev).await {
                eprintln!("log_write_failed appending assistant_message for stream {stream:?}: {e:?}");
            }
        }
        Err(e) => {
            let code = machine_code_for_provider_error(&e);
            fail_turn(&state.sessions, &stream, &operator, &generation, &code, &state.model.model).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::store::LogStore;
    use crate::provider::KeylessProvider;
    use crate::sessions::Sessions;
    use crate::state::{AppState, ModelConfig};

    fn model_config() -> ModelConfig {
        ModelConfig {
            model: "claude-sonnet-5".to_string(),
            api_key: None,
        }
    }

    async fn create_instance(sessions: &Sessions, operator: Option<&str>) -> String {
        let id = uuid::Uuid::new_v4().to_string();
        sessions
            .append(
                &id,
                NewEvent {
                    actor: Actor {
                        role: Role::System,
                        instance: None,
                    },
                    event_type: "instance_created".to_string(),
                    data: serde_json::json!({ "operator": operator, "startedAt": now_rfc3339() }),
                },
            )
            .unwrap();
        sessions
            .append(
                &id,
                NewEvent {
                    actor: Actor {
                        role: Role::User,
                        instance: None,
                    },
                    event_type: "user_message".to_string(),
                    data: serde_json::json!({ "text": "hi", "clientKey": "c1" }),
                },
            )
            .unwrap();
        id
    }

    #[tokio::test]
    async fn keyless_provider_turn_produces_the_error_shaped_assistant_message() {
        let data_dir = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        let store = LogStore::open(data_dir.path()).unwrap();
        let sessions = Arc::new(Sessions::new(store));
        let id = create_instance(&sessions, Some("tycho")).await;

        let state: SharedState = Arc::new(AppState {
            root: root.path().canonicalize().unwrap(),
            sessions: sessions.clone(),
            model: model_config(),
            provider: Arc::new(KeylessProvider),
        });

        let generation = uuid::Uuid::new_v4().to_string();
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        run_turn(state, id.clone(), generation.clone(), cancel_rx).await;

        let events = sessions.store().read_from(&id, 0).unwrap();
        let last = events.last().unwrap();
        assert_eq!(last.event_type, "assistant_message");
        assert_eq!(last.data["text"], "");
        assert_eq!(last.data["interrupted"], true);
        assert_eq!(last.data["error"], "no_api_key");
        assert_eq!(last.data["generation"], generation);
        // The failure shape must carry `model` just like the success shape
        // does — the configured model string, since a failure path never
        // gets a `TurnOutcome::model` to report.
        assert_eq!(last.data["model"], "claude-sonnet-5");
        assert_eq!(last.actor.instance.as_deref(), Some("tycho"));
    }

    #[tokio::test]
    async fn operator_label_falls_back_to_dispatcher_when_instance_has_no_operator() {
        let data_dir = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        let store = LogStore::open(data_dir.path()).unwrap();
        let sessions = Arc::new(Sessions::new(store));
        let id = create_instance(&sessions, None).await;

        let state: SharedState = Arc::new(AppState {
            root: root.path().canonicalize().unwrap(),
            sessions: sessions.clone(),
            model: model_config(),
            provider: Arc::new(KeylessProvider),
        });

        let generation = uuid::Uuid::new_v4().to_string();
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        run_turn(state, id.clone(), generation, cancel_rx).await;

        let events = sessions.store().read_from(&id, 0).unwrap();
        let last = events.last().unwrap();
        assert_eq!(last.actor.instance.as_deref(), Some("dispatcher"));
    }
}
