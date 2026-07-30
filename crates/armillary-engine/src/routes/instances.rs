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
/// containment — that function is `pub(crate)` for exactly this reason, so what
/// this call site accepts is exactly what the projection will later accept.
///
/// `resolve_boot_path` is NOT a full path guard, and this function must not be
/// read as inheriting one. All it does is `root.join(rel)`, `canonicalize`, and
/// `starts_with(root)`. Because `Path::join` with an absolute argument *replaces*
/// the base, an absolute path that happens to sit inside root canonicalizes to
/// something under root and passes. That is precisely why the caller-side
/// absolute-path refusal below exists — it is the only thing rejecting an
/// absolute `[router] boot`, and any future caller of `resolve_boot_path` needs
/// its own. (`guard::resolve` is the resolver that does reject absolute paths and
/// `..` components before canonicalizing; unifying the three resolvers on it is
/// parked for Phase 2.)
///
/// Mirrors `loop_::rerecord_boot`'s shape (resolve off-thread, then
/// `tokio::fs::read`) since both call sites read the same kind of file for
/// the same reason.
async fn append_boot_event(state: &SharedState, stream: &str, rel: &str) {
    // `constitution/composition.md` (Task 1) documents `[router] boot` as a
    // path RELATIVE to root. An absolute path that happens to sit inside
    // root would still pass `resolve_boot_path`'s containment check below —
    // and then get written verbatim into a durable, portable log, where it
    // would break on any machine whose root sits somewhere else. Refused
    // rather than silently relativized (e.g. via `strip_prefix`): rewriting
    // it would hide a misconfiguration instead of surfacing it.
    if std::path::Path::new(rel).is_absolute() {
        eprintln!(
            "warning: [router] boot = {rel:?} is an absolute path, but this key is documented \
             as relative to the workspace root — creating instance {stream} with no boot event \
             (it will have no system prompt)"
        );
        return;
    }

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
                "warning: [router] boot = {rel:?} cannot be resolved (missing, or outside the \
                 workspace root) — creating instance {stream} with no boot event (it will have \
                 no system prompt)"
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

    // A non-UTF-8 boot file must never be appended: `projection.rs`'s "boot"
    // arm decodes the bytes with `String::from_utf8`, and a decode failure
    // there surfaces as `ProjectionError::BootUnreadable`, which
    // `loop_::run_turn` routes to `fail_turn(..., "boot_unreadable", ...)` —
    // NOT to the drift-recovery branch (only a SHA mismatch on an
    // already-recorded event re-records). Appending an event that can never
    // be read back would turn a benign misconfiguration into every future
    // turn on this stream failing forever — the exact inversion of
    // skip-never-fail this function exists to avoid.
    if std::str::from_utf8(&bytes).is_err() {
        eprintln!(
            "warning: [router] boot = {rel:?} is not valid UTF-8 — creating instance {stream} \
             with no boot event (appending it would make every turn on this stream fail)"
        );
        return;
    }

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

/// **DD-1 — record the composition as its own durable event.**
///
/// The one writer, used both at instance creation and by the loop's
/// drift repair, so "what a composition event contains" cannot come to depend
/// on which path wrote it.
///
/// C-3 as running code: the manifests are parsed by a TOML parser and the
/// answer is recorded, so a model never re-derives it from raw bytes — a local
/// model once read commented-out examples as a live composition. P-2: recorded
/// before it is ever projected. I-1: the drift repair appends a new event
/// rather than editing the old one.
///
/// `Role::System`, like `boot` — the engine determined this, and the log must
/// not claim the operator did.
pub(crate) async fn append_composition_event(
    state: &SharedState,
    stream: &str,
) -> Result<(), &'static str> {
    let root = state.root.clone();
    let data = match tokio::task::spawn_blocking(move || crate::tools::composition_event_data(&root))
        .await
    {
        Ok(Ok(data)) => data,
        // Presence-gated (C-4): a workspace whose manifests will not parse is a
        // misconfiguration, not a reason to refuse a session. A bare clone
        // parses to "nothing composed", which is a true and useful answer.
        _ => return Err("composition_unreadable"),
    };

    let sessions = state.sessions.clone();
    let stream_owned = stream.to_string();
    let appended = tokio::task::spawn_blocking(move || {
        sessions.append(
            &stream_owned,
            NewEvent {
                actor: Actor {
                    role: Role::System,
                    instance: None,
                },
                event_type: "composition".to_string(),
                data,
            },
        )
    })
    .await;

    match appended {
        Ok(Ok(_)) => Ok(()),
        _ => Err("log_write_failed"),
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
    if let Some(rel) = state.boot.as_deref() {
        append_boot_event(&state, &id, rel).await;
    }

    // DD-1, and after boot for the same reason boot comes after
    // instance_created: the earlier events' seq numbers are documented and
    // referenced, and there is no reason to renumber them. Skip-never-fail —
    // a workspace whose manifests will not parse still gets a session, it just
    // gets one that has to read them with a tool.
    if let Err(code) = append_composition_event(&state, &id).await {
        eprintln!("warning: no composition event for {id} ({code})");
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
