//! Instance routes: create, list, attach.
//!
//! The instance registry is **log-derived, not a second source of truth**:
//! list and attach both read `LogStore::streams()` and each stream's own
//! events. There is no separate table an instance could be created in and
//! then missing from, or vice versa — the log is the only place an instance
//! is recorded, per this repo's standing rule that the log is the truth and
//! everything else is a projection over it.

use crate::log::envelope::{Actor, EventEnvelope, Role};
use crate::log::store::LogStore;
use crate::sessions::{NewEvent, SessionError};
use crate::state::SharedState;
use crate::blocking;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateRequest {
    #[serde(default)]
    pub operator: Option<String>,
}

#[derive(Serialize, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Instance {
    pub id: String,
    pub operator: Option<String>,
    pub stream: String,
    pub started_at: String,
    pub last_seq: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachInfo {
    pub instance: Instance,
    pub earliest_seq: u64,
    pub head_seq: u64,
}

/// Reconstructs an `Instance` from a stream's first event and its current
/// head. `None` when the first event is not `instance_created` — defensive:
/// nothing in this codebase writes a stream's first event as anything else,
/// but the registry being log-derived means this function is the one place
/// that discipline could quietly break, so it is checked rather than assumed.
fn instance_from_first_event(stream: &str, first: &EventEnvelope, head_seq: u64) -> Option<Instance> {
    if first.event_type != "instance_created" {
        return None;
    }
    let started_at = first.data.get("startedAt")?.as_str()?.to_string();
    let operator = first
        .data
        .get("operator")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Some(Instance {
        id: stream.to_string(),
        operator,
        stream: stream.to_string(),
        started_at,
        last_seq: head_seq,
    })
}

/// All filesystem work for a listing: one instance per stream whose first
/// event is `instance_created`.
fn list_instances(store: &LogStore) -> Result<Vec<Instance>, SessionError> {
    let mut out = Vec::new();
    for stream in store.streams()? {
        let events = store.read_from(&stream, 0)?;
        let Some(first) = events.first() else {
            continue;
        };
        let head_seq = events.last().map(|e| e.seq).unwrap_or(0);
        match instance_from_first_event(&stream, first, head_seq) {
            Some(instance) => out.push(instance),
            None => eprintln!(
                "warning: stream {stream:?}'s first event is {:?}, not instance_created — \
                 skipping it from the instance listing",
                first.event_type
            ),
        }
    }
    out.sort_by(|a, b| a.started_at.cmp(&b.started_at));
    Ok(out)
}

/// All filesystem work for one attach: `LogStore::read_from` is the replay
/// half of A-1's `subscribe(stream, from_seq)`; this just derives the
/// summary an attaching client needs before it starts consuming events.
fn attach_info(store: &LogStore, id: &str) -> Result<AttachInfo, SessionError> {
    let events = store.read_from(id, 0)?;
    let first = events.first().ok_or(SessionError::UnknownInstance)?;
    let head_seq = events.last().map(|e| e.seq).unwrap_or(0);
    let earliest_seq = first.seq;
    let instance = instance_from_first_event(id, first, head_seq)
        .ok_or(SessionError::UnknownInstance)?;

    Ok(AttachInfo {
        instance,
        earliest_seq,
        head_seq,
    })
}

/// Appends the router's `boot` event, or logs and skips.
///
/// **Skip, never fail.** An instance that works without identity beats an
/// instance that cannot be created — the same posture as `KeylessProvider`
/// (the engine serves regardless; you discover the gap at first use). The
/// accepted cost, recorded in the design: a misconfigured boot path is
/// invisible from the phone, indistinguishable from having declared none.
/// Phase 2 makes it a durable event.
///
/// Reuses `projection::resolve_boot_path` rather than re-deriving root
/// containment — that function is `pub(crate)` for exactly this reason, so an
/// absolute or `..`-laden path is rejected by the same code the projection
/// trusts. Mirrors `loop_::rerecord_boot`'s shape (resolve off-thread, then
/// `tokio::fs::read`) since both call sites read the same kind of file for
/// the same reason.
async fn append_boot_event(state: &SharedState, stream: &str, rel: &str) {
    let root = state.root.clone();
    let rel_owned = rel.to_string();
    let resolved = match tokio::task::spawn_blocking(move || {
        crate::projection::resolve_boot_path(&root, &rel_owned)
    })
    .await
    {
        Ok(Ok(path)) => path,
        _ => {
            eprintln!(
                "warning: [router] boot = {rel:?} does not resolve under the workspace root — \
                 creating instance {stream} with no boot event (it will have no system prompt)"
            );
            return;
        }
    };

    let bytes = match tokio::fs::read(&resolved).await {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!(
                "warning: [router] boot = {rel:?} is unreadable ({e}) — creating instance \
                 {stream} with no boot event (it will have no system prompt)"
            );
            return;
        }
    };
    let sha256 = crate::hash::sha256_hex(&bytes);

    let sessions = state.sessions.clone();
    let stream_owned = stream.to_string();
    let path_for_event = rel.to_string();
    let appended = tokio::task::spawn_blocking(move || {
        sessions.append(
            &stream_owned,
            NewEvent {
                actor: Actor {
                    role: Role::System,
                    instance: None,
                },
                event_type: "boot".to_string(),
                data: serde_json::json!({ "path": path_for_event, "sha256": sha256 }),
            },
        )
    })
    .await;

    match appended {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => eprintln!("warning: failed to append the boot event for {stream}: {e:?}"),
        Err(_) => eprintln!("warning: the boot append task panicked for {stream}"),
    }
}

pub async fn create(
    State(state): State<SharedState>,
    Json(body): Json<CreateRequest>,
) -> Result<(StatusCode, Json<Instance>), (StatusCode, String)> {
    let sessions = state.sessions.clone();
    let operator = body.operator;
    let id = uuid::Uuid::new_v4().to_string();
    let started_at = humantime::format_rfc3339_millis(SystemTime::now()).to_string();

    let stream = id.clone();
    let ev = blocking::run(move || {
        sessions
            .append(
                &stream,
                NewEvent {
                    actor: Actor {
                        role: Role::System,
                        instance: None,
                    },
                    event_type: "instance_created".to_string(),
                    data: serde_json::json!({ "operator": operator, "startedAt": started_at }),
                },
            )
            .map_err(SessionError::into_response)
    })
    .await?;

    // AFTER instance_created, never before: instance_from_first_event requires
    // a stream's first event to be instance_created, and list_instances skips
    // any stream where it is not. Boot-first would make every instance
    // invisible to the list screen.
    if let Some(rel) = state.boot.clone() {
        append_boot_event(&state, &id, &rel).await;
    }

    // last_seq stays ev.seq (1, from instance_created) even when a boot event
    // just landed at seq 2: this response describes the instance as CREATED,
    // not its current head. A client that wants the head calls attach.
    let instance = instance_from_first_event(&id, &ev, ev.seq).ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        "log_write_failed".to_string(),
    ))?;

    Ok((StatusCode::CREATED, Json(instance)))
}

pub async fn list(
    State(state): State<SharedState>,
) -> Result<Json<Vec<Instance>>, (StatusCode, String)> {
    let sessions = state.sessions.clone();
    let instances =
        blocking::run(move || list_instances(sessions.store()).map_err(SessionError::into_response))
            .await?;
    Ok(Json(instances))
}

pub async fn attach(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<AttachInfo>, (StatusCode, String)> {
    let sessions = state.sessions.clone();
    let info = blocking::run(move || {
        attach_info(sessions.store(), &id).map_err(SessionError::into_response)
    })
    .await?;
    Ok(Json(info))
}
