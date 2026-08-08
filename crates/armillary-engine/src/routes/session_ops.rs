//! `POST /instances/{id}/send|interrupt|evict` — the loop's HTTP face.
//!
//! Each handler's own job is small and synchronous: validate the instance
//! exists, do the one durable append (or none, for `interrupt`) this
//! endpoint owns, and — for `send` only — hand off to `loop_::run_turn` in a
//! spawned task rather than awaiting it. The turn itself (provider call,
//! transient deltas, the completion/interrupt/failure append) is
//! `loop_.rs`'s job, not this module's; nothing here reaches into
//! `project_context` or `ModelProvider` directly.

use crate::blocking;
use crate::loop_;
use crate::log::envelope::{Actor, Role};
use crate::log::store::LogStore;
use crate::sessions::{NewEvent, SessionError, Sessions, TurnHandle};
use crate::state::SharedState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SendRequest {
    pub text: String,
    pub client_key: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SendReceipt {
    pub id: String,
    pub seq: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvictRequest {
    pub event_id: String,
}

/// An instance exists iff the store has a durable log for it (v0: nothing
/// subscribes/sends before something durable has been appended, so
/// `store().streams()` is the whole check) — matching
/// `routes::subscribe::subscribe`'s existing existence check, not a second
/// notion of "known instance."
fn stream_exists(store: &LogStore, stream: &str) -> Result<bool, SessionError> {
    Ok(store.streams()?.iter().any(|s| s == stream))
}

async fn require_known_instance(state: &SharedState, id: &str) -> Result<(), (StatusCode, String)> {
    let sessions = state.sessions.clone();
    let id = id.to_string();
    let known = blocking::run(move || {
        stream_exists(sessions.store(), &id).map_err(SessionError::into_response)
    })
    .await?;
    if !known {
        return Err((StatusCode::NOT_FOUND, "unknown_instance".to_string()));
    }
    Ok(())
}

/// `POST /instances/{id}/send` — the turn, step 1-2 (see `constitution/
/// instances.md` P-2, A-4):
///
/// 1. Validate the instance exists and no turn is already in flight for its
///    stream (A-4: one turn per stream — `SessionError::TurnInProgress` ->
///    409 `turn_in_progress`, checked/claimed atomically by
///    `Sessions::begin_turn` so two concurrent `send`s can't both win).
/// 2. Append the durable `user_message` (I-5: a failed write is 500
///    `log_write_failed`, and releases the turn slot just claimed so a
///    retry isn't blocked by a turn that never started). The response
///    carries THIS event's `id`/`seq` — P-2: the receipt is the recorded
///    fact's identity, returned before the model ever runs.
/// 3. Spawn `loop_::run_turn` and return 201 immediately — the turn runs
///    independently of this request, which has already gotten its answer.
pub async fn send(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(body): Json<SendRequest>,
) -> Result<(StatusCode, Json<SendReceipt>), (StatusCode, String)> {
    require_known_instance(&state, &id).await?;

    let generation = uuid::Uuid::new_v4().to_string();
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let handle = TurnHandle {
        cancel: cancel_tx,
        generation: generation.clone(),
    };
    state
        .sessions
        .begin_turn(&id, handle)
        .map_err(SessionError::into_response)?;

    let sessions = state.sessions.clone();
    let stream = id.clone();
    let text = body.text;
    let client_key = body.client_key;
    let append_result = blocking::run(move || {
        sessions
            .append(
                &stream,
                NewEvent {
                    actor: Actor {
                        role: Role::User,
                        instance: None,
                        principal: None,
                    },
                    event_type: "user_message".to_string(),
                    data: serde_json::json!({ "text": text, "clientKey": client_key }),
                },
            )
            .map_err(SessionError::into_response)
    })
    .await;

    let user_event = match append_result {
        Ok(ev) => ev,
        Err(e) => {
            // The turn never actually started — release the slot rather
            // than leaving this stream permanently 409ing on every retry.
            state.sessions.end_turn(&id);
            return Err(e);
        }
    };

    let receipt = SendReceipt {
        id: user_event.id,
        seq: user_event.seq,
    };

    tokio::spawn(loop_::run_turn(state.clone(), id, generation, cancel_rx));

    Ok((StatusCode::CREATED, Json(receipt)))
}

/// `POST /instances/{id}/interrupt` — always 204, whether or not a turn is
/// running (idempotent): `Sessions::interrupt` is a no-op when there is
/// nothing claimed.
///
/// S-3 (enforceable beats advisory): this is the client's side of the
/// enforcement mechanism, not a request the model gets to weigh in on — see
/// `Sessions::interrupt`'s doc for the halt itself.
///
/// One race is inherent, not a bug: if this lands after the provider has
/// already finished producing its outcome but before `loop_::run_turn` has
/// appended the durable `assistant_message` (the turn slot is still claimed
/// until then — `EndTurnGuard` clears it on return, which is after that
/// append), the signal reaches nobody listening. This still answers 204 —
/// the client asked for a stop and one is not owed a distinct answer for
/// "too late" — and the record it produces is simply honest: the turn had
/// already completed, so the durable `assistant_message` shows a completed
/// turn, not an interrupted one.
pub async fn interrupt(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    require_known_instance(&state, &id).await?;
    state.sessions.interrupt(&id);
    Ok(StatusCode::NO_CONTENT)
}

/// All filesystem work for one evict: unknown instance -> 404
/// `unknown_instance`; a target `eventId` this stream never recorded -> 404
/// `unknown_event`; otherwise append the durable `context_evict` (P-1:
/// evicted, not deleted — the target event still lives in the log, it is
/// simply absent from the next `project_context`).
fn append_evict(sessions: &Sessions, stream: &str, event_id: &str) -> Result<(), SessionError> {
    if !stream_exists(sessions.store(), stream)? {
        return Err(SessionError::UnknownInstance);
    }
    let events = sessions.store().read_from(stream, 0)?;
    if !events.iter().any(|e| e.id == event_id) {
        return Err(SessionError::UnknownEvent);
    }
    sessions.append(
        stream,
        NewEvent {
            actor: Actor {
                role: Role::User,
                instance: None,
                principal: None,
            },
            event_type: "context_evict".to_string(),
            data: serde_json::json!({ "target": event_id }),
        },
    )?;
    Ok(())
}

/// `POST /instances/{id}/evict {"eventId"}` -> 204, or 404
/// `unknown_instance`/`unknown_event`.
pub async fn evict(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(body): Json<EvictRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    let sessions = state.sessions.clone();
    blocking::run(move || {
        append_evict(&sessions, &id, &body.event_id).map_err(SessionError::into_response)
    })
    .await?;
    Ok(StatusCode::NO_CONTENT)
}
