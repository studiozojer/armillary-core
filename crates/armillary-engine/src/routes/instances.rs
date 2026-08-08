//! Instance routes: create, list, attach.
//!
//! The instance registry is **log-derived, not a second source of truth**:
//! list and attach both read `LogStore::streams()` and each stream's own
//! events. There is no separate table an instance could be created in and
//! then missing from, or vice versa — the log is the only place an instance
//! is recorded, per this repo's standing rule that the log is the truth and
//! everything else is a projection over it.

use crate::auth::Caller;
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
    /// **WD-9.** Whether this instance may write `modules.toml` /
    /// `modules.local.toml`. Defaults to **false**: D-4 makes the manifest
    /// operator-writable in principle, and this is the per-session grant that
    /// decides whether *this* session is the one doing it.
    ///
    /// One bit today, and knowingly a degenerate case of a glob-scoped grant —
    /// when `delete_file` and `move_file` arrive, a second boolean is the wrong
    /// move and the shape to grow into is `patterns: ["modules*.toml"]`.
    #[serde(default)]
    pub may_write_composition: bool,

    /// Which model pilots this instance, pinned at creation and immutable
    /// for its life (design decision 1). **Never validated here** — not the
    /// slug, not the provider's key presence (decision 3): an instance
    /// pinned to a model this host cannot pilot is created successfully and
    /// fails its first turn with `no_api_key`, on an event that names the
    /// model. That keeps the refusal durable in the instance's own log
    /// rather than evaporating in an HTTP response, and it preserves
    /// core#19 decision 1's pass-through freedom — a slug this engine has
    /// never heard of is catalog news, not a request to reject.
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Serialize, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Instance {
    pub id: String,
    pub operator: Option<String>,
    pub stream: String,
    pub started_at: String,
    pub last_seq: u64,
    pub may_write_composition: bool,
    pub model: Option<String>,
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
    // The one reader for a per-instance string field, shared with `loop_`'s
    // resolvers — see `created_str_field` for why both the absence and the
    // empty-string filters are there.
    let read = crate::loop_::created_str_field;

    // Defaulted, not required: an instance created before this field existed
    // must list as `false` rather than fail to parse. The same defaulting
    // `loop_::may_write_composition` uses — and the registry being log-derived
    // means this is where every listing and attach gets it.
    let may_write_composition = first
        .data
        .get("mayWriteComposition")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Defaulted like the grant above, and for the same reason: an instance
    // created before this field existed must list and attach, not fail to
    // parse.
    let model = read("model")(first);

    Some(Instance {
        id: stream.to_string(),
        operator,
        stream: stream.to_string(),
        started_at,
        last_seq: head_seq,
        may_write_composition,
        model,
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

/// Hash one declared boot file, report it absent, or refuse the declaration.
///
/// **Skip-never-fail, one file at a time.** A workspace where one identity file
/// was renamed still gets a session, and the absence is *recorded* so the
/// projection can see it and a human can grep it. Silently loading two of three
/// files is the failure this shape exists to avoid.
///
/// Three outcomes, and the split between the last two is load-bearing:
///
/// - **Loadable** → `{path, sha256, present: true}`.
/// - **Declared fine but not loadable right now** (missing, unreadable, not
///   UTF-8) → `{path, present: false}`. Recorded, and re-checked every
///   projection, because a file that becomes loadable is real drift.
/// - **A malformed declaration** (absolute, or escaping the root) → `None`,
///   recorded nowhere. Nothing on disk can make it valid, so there is nothing
///   to re-check; recording it absent would make the projection drift on it
///   forever. This is a manifest bug, and the warning is the whole remedy.
async fn weigh_boot_file(root: &std::path::Path, rel: &str) -> Option<serde_json::Value> {
    let malformed = |why: &str| -> Option<serde_json::Value> {
        eprintln!("warning: boot file {rel:?} {why} — not recorded (fix the manifest)");
        None
    };
    let absent = |why: &str| {
        eprintln!("warning: boot file {rel:?} {why} — recorded absent, not loaded");
        Some(serde_json::json!({ "path": rel, "present": false }))
    };

    // Documented as RELATIVE to root. An absolute path that happens to sit
    // inside root would pass the containment check below and then be written
    // verbatim into a durable, portable log, breaking on any machine whose
    // root sits elsewhere. Refused rather than silently relativized —
    // rewriting it would hide the misconfiguration instead of surfacing it.
    if std::path::Path::new(rel).is_absolute() {
        return malformed("is absolute, and this key is documented as relative to the root");
    }

    let (root_owned, rel_owned) = (root.to_path_buf(), rel.to_string());
    let resolved = match tokio::task::spawn_blocking(move || {
        crate::projection::resolve_boot_path(&root_owned, &rel_owned)
    })
    .await
    {
        Ok(Ok(path)) => path,
        // Conflated on purpose, and the conflation costs nothing: a path that
        // escaped the root is malformed, a path that is merely missing is
        // absent, and `resolve_boot_path` cannot tell them apart. Recording it
        // absent is the safe reading of the ambiguity — an escaping path
        // cannot become loadable through the resolver, so it never drifts.
        _ => return absent("cannot be resolved (missing, or outside the workspace root)"),
    };

    let Ok(bytes) = tokio::fs::read(&resolved).await else {
        return absent("is unreadable");
    };

    // A non-UTF-8 boot file must never be recorded present: the projection
    // decodes with `String::from_utf8`, and a failure there surfaces as
    // `BootUnreadable`, which `run_turn` routes to `fail_turn` — NOT to
    // drift recovery, which only fires on a SHA mismatch. Recording bytes
    // that can never be read back would turn a benign misconfiguration into
    // every future turn on this stream failing forever.
    if std::str::from_utf8(&bytes).is_err() {
        return absent("is not valid UTF-8");
    }

    Some(serde_json::json!({
        "path": rel,
        "sha256": crate::hash::sha256_hex(&bytes),
        "present": true,
    }))
}

/// **B-2 — boot the summoned operator's own declared files.**
///
/// The ordered list is the router's own `boot` (if declared) followed by the
/// summoned operator's, and order is load-bearing: it is what the manifest
/// meant, and it is prefix-cache order, where an edit late in the list
/// invalidates less than one early.
///
/// **The declaration is read once, at instance creation, and then fixed.**
/// Editing an operator's `boot` list mid-session does not re-boot a running
/// instance — it takes effect on the next one. A session keeps the identity it
/// booted with. (Editing a boot *file* is different and IS picked up: that is
/// `BootDrift`, and the repair re-hashes the same declared paths.)
///
/// Reuses `projection::resolve_boot_path` rather than re-deriving root
/// containment, so what this accepts is exactly what the projection will later
/// accept. That function is NOT a full path guard — see its own docs and
/// `guard.rs`'s TRIPWIRE — which is why `weigh_boot_file` refuses absolute
/// paths itself. These paths come from a manifest, never from a client.
async fn append_boot_event(
    state: &SharedState,
    stream: &str,
    operator: Option<&str>,
    snap: Option<&crate::snapshot::WorkspaceSnapshot>,
) {
    let mut declared: Vec<String> = Vec::new();
    if let Some(rel) = state.boot.as_deref() {
        declared.push(rel.to_string());
    }

    // The same snapshot the composition event records — one read at create,
    // so the two records cannot disagree about which manifests they saw.
    // `None` (an unloadable manifest) skips the operator's boot list, which
    // is what the old parse-failure path did too.
    if let (Some(name), Some(snap)) = (operator, snap) {
        if let Some(op) = snap.composition.operators.iter().find(|o| o.name == name) {
            declared.extend(op.boot.clone().unwrap_or_default());
        }
    }

    // C-4: nothing declared means no boot event and no system prompt. Not an
    // error — a bare clone is a working host.
    if declared.is_empty() {
        return;
    }

    let mut files = Vec::with_capacity(declared.len());
    for rel in &declared {
        if let Some(entry) = weigh_boot_file(&state.root, rel).await {
            files.push(entry);
        }
    }

    // Every declared path was malformed, so there is nothing to record and
    // nothing a later projection could re-check.
    if files.is_empty() {
        return;
    }

    let sessions = state.sessions.clone();
    let stream_owned = stream.to_string();
    let appended = tokio::task::spawn_blocking(move || {
        sessions.append(
            &stream_owned,
            NewEvent {
                actor: Actor {
                    role: Role::System,
                    instance: None,
                    principal: None,
                },
                event_type: "boot".to_string(),
                data: serde_json::json!({ "files": files }),
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
    // A FRESH snapshot, deliberately: this signature's remaining caller is
    // the loop's drift repair, whose whole job is recording current state.
    let root = state.root.clone();
    let snap = match tokio::task::spawn_blocking(move || {
        crate::snapshot::WorkspaceSnapshot::load(&root)
    })
    .await
    {
        Ok(Ok(snap)) => snap,
        _ => return Err("composition_unreadable"),
    };
    append_composition_event_from(state, stream, &snap).await
}

/// `append_composition_event` over an already-loaded snapshot — the create
/// path shares one snapshot between this and the boot event.
pub(crate) async fn append_composition_event_from(
    state: &SharedState,
    stream: &str,
    snap: &crate::snapshot::WorkspaceSnapshot,
) -> Result<(), &'static str> {
    let data = match crate::tools::composition_event_data_from(snap, &state.root) {
        Ok(data) => data,
        // Presence-gated (C-4): a workspace whose manifests will not parse is a
        // misconfiguration, not a reason to refuse a session. A bare clone
        // parses to "nothing composed", which is a true and useful answer.
        Err(_) => return Err("composition_unreadable"),
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
                    principal: None,
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

/// Authenticated but not grant-checked: `sync`/`push` are authorities over
/// git, and there is no "may run a turn" grant because nothing has needed to
/// deny one. The `Caller` is taken here because instance creation is the
/// route through which a file write inherits its principal (design § 3.4) —
/// so this parameter is load-bearing for Stage 3 even before it is read.
pub async fn create(
    State(state): State<SharedState>,
    caller: Caller,
    Json(body): Json<CreateRequest>,
) -> Result<(StatusCode, Json<Instance>), (StatusCode, String)> {
    let sessions = state.sessions.clone();
    // Normalized on the WRITE side, not only on the read. `model` needed a
    // read-side filter because its write side once admitted `""`; `principal`
    // has never been written, so the wound is closed at the source and never
    // opens. The read filter in `created_str_field` stays anyway — it does a
    // different job, absorbing instances created before this field existed.
    let principal = Some(caller.0.name.clone()).filter(|s| !s.trim().is_empty());
    let operator = body.operator;
    // `operator` is moved into the `instance_created` payload below; B-2 needs
    // the name again afterwards to resolve that operator's declared boot.
    let operator_name = operator.clone();
    let may_write_composition = body.may_write_composition;
    // Empty or whitespace-only becomes `None` HERE, on the write path — not
    // only on the two read paths (`instance_from_first_event`'s and
    // `models::declared_default`'s `.filter(|s| !s.is_empty())`). Without
    // this, `{"model": ""}` logs `"model": ""` verbatim in `instance_created`
    // while every read of that same event reports `null` — the log and the
    // API would disagree about what this instance was created with.
    let model = body.model.filter(|s| !s.trim().is_empty());
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
                        principal: None,
                    },
                    event_type: "instance_created".to_string(),
                    data: serde_json::json!({
                        "operator": operator,
                        "startedAt": started_at,
                        "mayWriteComposition": may_write_composition,
                        "model": model,
                        // Who asked for this instance to exist. Every write
                        // the instance later performs inherits it (§ 3.4):
                        // there is no HTTP write route, so a file write is a
                        // model tool inside a turn, and the turn belongs to
                        // this instance.
                        "principal": principal,
                    }),
                },
            )
            .map_err(SessionError::into_response)
    })
    .await?;

    // AFTER instance_created, never before: instance_from_first_event requires
    // a stream's first event to be instance_created, and list_instances skips
    // any stream where it is not. Boot-first would make every instance
    // invisible to the list screen.
    // B-2: which files boot depends on WHO was summoned, so this is no longer
    // gated on `state.boot` alone — an operator can declare a boot where the
    // router declares none. The function itself is presence-gated and returns
    // without writing when nothing is declared either way.
    // ONE snapshot for both records below: the boot event's operator lookup
    // and the composition event's digests derive from the same read, so the
    // race the old second parse accepted is gone rather than tolerated.
    let snap_root = state.root.clone();
    let snap = tokio::task::spawn_blocking(move || {
        crate::snapshot::WorkspaceSnapshot::load(&snap_root)
    })
    .await
    .ok()
    .and_then(|r| r.ok());

    append_boot_event(&state, &id, operator_name.as_deref(), snap.as_ref()).await;

    // DD-1, and after boot for the same reason boot comes after
    // instance_created: the earlier events' seq numbers are documented and
    // referenced, and there is no reason to renumber them. Skip-never-fail —
    // a workspace whose manifests will not parse still gets a session, it just
    // gets one that has to read them with a tool.
    match &snap {
        Some(snap) => {
            if let Err(code) = append_composition_event_from(&state, &id, snap).await {
                eprintln!("warning: no composition event for {id} ({code})");
            }
        }
        None => eprintln!("warning: no composition event for {id} (composition_unreadable)"),
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
