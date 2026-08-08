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
use crate::provider::ProviderError;
#[cfg(test)]
use crate::provider::TurnOutcome;
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

/// **WD-9.** Whether this instance may write the workspace manifests.
///
/// Read from `instance_created.data`, exactly as `operator_label` reads the
/// operator: per-instance facts already live there, and deriving it from the
/// log rather than from a side table keeps I-1's "the log is the truth" honest.
///
/// `pub` rather than private so `tests/routes.rs` can prove the WRITER
/// (`routes::instances::create`) and this READER agree on the key. A private
/// reader tested against a hand-built event proves only that the reader is
/// self-consistent — which is the seam-defect shape this repo has shipped once
/// already.
pub fn may_write_composition(events: &[EventEnvelope]) -> bool {
    events
        .iter()
        .find(|e| e.event_type == "instance_created")
        .and_then(|e| e.data.get("mayWriteComposition"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Which model pilots this instance, as recorded at its creation. `None`
/// when the field is absent — an instance created before per-instance
/// models existed must keep piloting, so the caller falls back to the
/// engine's default rather than this returning a placeholder. Same shape
/// and same defaulting discipline as `may_write_composition` above, for
/// the same reason: the registry is log-derived, so the first event is the
/// only place either answer lives.
pub fn model_for(events: &[EventEnvelope]) -> Option<String> {
    events
        .iter()
        .find(|e| e.event_type == "instance_created")
        .and_then(|e| e.data.get("model"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

fn assistant_actor(operator: &str) -> Actor {
    Actor {
        role: Role::Operator,
        instance: Some(operator.to_string()),
        principal: None,
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
/// `model` is the INSTANCE's resolved model (`loop_::model_for`, falling
/// back to `state.model.model` when unrecorded) — not one a provider ever
/// confirmed running, since a failure path by definition has no
/// `TurnOutcome::model` to report: it either never reached the provider or
/// the provider itself is what failed. Included anyway so the success and
/// failure shapes agree on which fields an `assistant_message` always
/// carries, rather than the failure shape silently being the one case
/// missing it. One caller — the `log_unreadable` path in `run_turn` — passes
/// `state.model.model` directly rather than a resolved instance model: the
/// read that would have named the instance's model is the one that failed.
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

/// Re-records a fresh `boot` event on `BootDrift`: re-reads the **same declared
/// paths** the drifted event listed, re-hashes their current bytes, and appends
/// a new `{ files: [...] }` event with `actor: system` — the disk content is
/// re-affirmed as current truth. I-1: the correction is a new event, never an
/// edit. Returns `boot_unreadable` on any failure (an unreadable file, or the
/// append itself failing — both leave the turn with no system prompt it can
/// trust, so both fold into the same named code).
///
/// **The declared paths come from the drifted event, not from the manifest.**
/// A boot *file* changing is drift and is repaired here; a boot *declaration*
/// changing is a composition change, and a running instance keeps the identity
/// it booted with. Reading the manifest here would silently re-scope a live
/// session's identity mid-turn.
///
/// A path recorded absent is re-checked and may now be present — an identity
/// that gained a file changed as surely as one that edited a file, and the
/// projection treats both as drift, so both must be repairable here.
///
/// **Deliberately lacks two guards that `routes::instances::append_boot_event`
/// has** — the absolute-path refusal and the UTF-8 check — and the asymmetry is
/// reasoned, not forgotten. This function never introduces a path: it can only
/// re-affirm one some earlier event already recorded, so an absolute path here
/// was already durable and refusing it now would strand the stream rather than
/// clean it. And a non-UTF-8 boot file fails this turn either way — the
/// re-projection immediately below returns `BootUnreadable` on the decode.
async fn rerecord_boot(
    state: &SharedState,
    stream: &str,
    paths: &[String],
) -> Result<(), &'static str> {
    let mut files = Vec::with_capacity(paths.len());
    for path in paths {
        let root = state.root.clone();
        let path_owned = path.clone();
        let resolved = tokio::task::spawn_blocking(move || {
            crate::projection::resolve_boot_path(&root, &path_owned)
        })
        .await
        .map_err(|_| "boot_unreadable")?;

        files.push(match resolved {
            Ok(resolved) => match tokio::fs::read(&resolved).await {
                Ok(bytes) => serde_json::json!({
                    "path": path,
                    "sha256": crate::hash::sha256_hex(&bytes),
                    "present": true,
                }),
                // Recorded absent rather than failing the turn: a renamed
                // identity file must not brick a running session, and the
                // absence is now visible in the log.
                Err(_) => serde_json::json!({ "path": path, "present": false }),
            },
            Err(_) => serde_json::json!({ "path": path, "present": false }),
        });
    }

    let ev = NewEvent {
        actor: Actor {
            role: Role::System,
            instance: None,
            principal: None,
        },
        event_type: "boot".to_string(),
        data: serde_json::json!({ "files": files }),
    };
    append_blocking(&state.sessions, stream, ev)
        .await
        .map_err(|_| "boot_unreadable")?;
    Ok(())
}

/// The paths the last standing `boot` event declared, in order.
///
/// Accepts both shapes for the same reason the projection does: a stream
/// written before B-2 carries a single `{path, sha256}`, and the repair must
/// work on history it did not write.
fn declared_boot_paths(events: &[crate::log::envelope::EventEnvelope]) -> Vec<String> {
    let Some(ev) = events.iter().rfind(|e| e.event_type == "boot") else {
        return Vec::new();
    };
    if let Some(files) = ev.data.get("files").and_then(|f| f.as_array()) {
        return files
            .iter()
            .filter_map(|f| f.get("path").and_then(|p| p.as_str()))
            .map(str::to_string)
            .collect();
    }
    ev.data
        .get("path")
        .and_then(|p| p.as_str())
        .map(|p| vec![p.to_string()])
        .unwrap_or_default()
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

    let events = match read_all(&state.sessions, &stream).await {
        Ok(events) => events,
        Err(e) => {
            eprintln!("failed to read stream {stream:?} for a turn: {e}");
            // The engine's default, necessarily: the read that would have
            // told us the instance's model is the one that just failed.
            fail_turn(&state.sessions, &stream, "dispatcher", &generation, "log_unreadable", &state.model.model).await;
            return;
        }
    };
    let operator = operator_label(&events);
    // Read once per turn, from the same event slice `operator_label` and
    // `may_write_composition` were given — no extra I/O. Absent, the
    // engine's default pilots, so an instance created before per-instance
    // models keeps working unchanged.
    let write_grant = may_write_composition(&events);
    let model = model_for(&events).unwrap_or_else(|| state.model.model.clone());

    // Text produced by rounds already finished. The provider restarts its own
    // accumulator on every call, so without this the phone's bubble would jump
    // backwards at each round boundary. I-4 still holds: every transient
    // carries a snapshot of the whole turn so far, never an increment.
    let mut turn_text = String::new();
    let mut round = 0usize;
    let mut stalled = 0usize;
    // Per-tool failure memory. letta's legacy loop dropped a just-failed tool
    // from the offered set and halted when the set emptied; that mechanism was
    // lost in their rewrite with no successor. It is the difference between a
    // bound that ESCALATES and one that lets a model retry the same denied path
    // until the round cap eats the budget.
    let mut failed_tools: std::collections::HashSet<String> = std::collections::HashSet::new();

    loop {
        round += 1;

        // A stop that arrives between rounds is observed here. Inside a round
        // the provider owns the signal; between them nothing else was watching.
        if *cancel_rx.borrow() {
            record_interrupt(&state, &stream, &operator, &generation, &turn_text, &model).await;
            return;
        }

        let turn = match project_healing(&state, &stream, &operator, &generation, &model).await {
            Some(turn) => turn,
            None => return, // fail_turn already recorded the reason
        };

        // At a bound the tools come off and `tool_choice: none` forces prose,
        // so the person always gets an answer instead of a turn that stops
        // mid-investigation. Measured: accepted on this engine's model with
        // adaptive thinking on.
        let at_bound = round >= MAX_ROUNDS || stalled >= MAX_STALLED_ROUNDS;
        let offered: Vec<crate::tools::ToolDef> = if at_bound {
            Vec::new()
        } else {
            crate::tools::registry()
                .iter()
                .filter(|t| !failed_tools.contains(t.def.name))
                .map(|t| t.def.clone())
                .collect()
        };
        let force_text = at_bound || offered.is_empty();
        let req = crate::provider::TurnRequest {
            turn,
            tools: offered,
            tool_choice: force_text.then_some(crate::provider::ToolChoice::ForceText),
        };

        // A fresh channel per round, and the relay prefixes what earlier rounds
        // produced. Hoisting one channel across the whole turn would keep a
        // sender alive that nothing drops, so the relay's `recv` never returns
        // `None`, `run_turn` never returns, `EndTurnGuard` never fires, and the
        // stream 409s on every later send for the life of the process.
        let (tx, mut rx) = mpsc::channel::<String>(32);
        let relay_sessions = state.sessions.clone();
        let relay_stream = stream.clone();
        let relay_operator = operator.clone();
        let relay_generation = generation.clone();
        let prefix = turn_text.clone();
        let relay = tokio::spawn(async move {
            while let Some(chunk) = rx.recv().await {
                let snapshot = format!("{prefix}{chunk}");
                let ev = transient_delta_envelope(&relay_stream, &relay_operator, &relay_generation, &snapshot);
                relay_sessions.broadcast_transient(&relay_stream, ev);
            }
        });

        let outcome = state
            .providers
            .provider_for(&model)
            .run_turn(req, tx, cancel_rx.clone())
            .await;
        let _ = relay.await;

        let outcome = match outcome {
            Ok(o) => o,
            Err(e) => {
                let code = machine_code_for_provider_error(&e);
                fail_turn(&state.sessions, &stream, &operator, &generation, &code, &model).await;
                return;
            }
        };

        let calls: Vec<(String, String, serde_json::Value)> = outcome
            .blocks
            .iter()
            .filter_map(|b| match b {
                crate::projection::ContentBlock::ToolUse { id, name, input } => {
                    Some((id.clone(), name.clone(), input.clone()))
                }
                _ => None,
            })
            .collect();

        if !outcome.text.is_empty() {
            turn_text.push_str(&outcome.text);
        }

        // Wire-shaped, via the same encoder build_request_body uses — one
        // encoding, two readers. Persisted only when the round also produced
        // text or calls: a thinking-only cut round would replay as an
        // assistant turn of nothing but thinking, an unmeasured wire shape
        // with no continuation value.
        let thinking_blocks: Vec<serde_json::Value> = outcome
            .blocks
            .iter()
            .filter(|b| {
                matches!(
                    b,
                    crate::projection::ContentBlock::Thinking { .. }
                        | crate::projection::ContentBlock::RedactedThinking { .. }
                )
            })
            .map(crate::provider::block_json)
            .collect();
        let persist_thinking =
            !thinking_blocks.is_empty() && (!outcome.text.is_empty() || !calls.is_empty());

        // Interrupted mid-generation. The `interrupt` event is recorded BEFORE
        // the partial assistant message — a pinned ordering, and the honest
        // one: the record reads "the user stopped it, and this is what had been
        // produced", not the reverse.
        if outcome.stopped {
            let ev = NewEvent {
                actor: Actor { role: Role::User, instance: None, principal: None },
                event_type: "interrupt".to_string(),
                data: serde_json::json!({}),
            };
            if let Err(e) = append_blocking(&state.sessions, &stream, ev).await {
                eprintln!("log_write_failed appending interrupt for stream {stream:?}: {e:?}");
            }
            let mut data = serde_json::json!({
                "text": outcome.text,
                "generation": generation,
                "interrupted": true,
                "model": outcome.model,
            });
            if persist_thinking {
                data["thinking"] = serde_json::json!(thinking_blocks.clone());
            }
            let partial = NewEvent {
                actor: assistant_actor(&operator),
                event_type: "assistant_message".to_string(),
                data,
            };
            match append_blocking(&state.sessions, &stream, partial).await {
                // Any call the model had begun is answered before the turn
                // closes. An unanswered `tool_use` is a 400 that kills every
                // later turn on this stream, so an interrupt must not be able
                // to leave one behind.
                Ok(ev) => answer_all(&state, &stream, &ev.id, &calls, "interrupted").await,
                Err(e) => eprintln!(
                    "log_write_failed appending interrupted assistant_message for stream {stream:?}: {e:?}"
                ),
            }
            return;
        }

        // The assistant event is always recorded — it is what happened, and its
        // id is the batch parent for this round's calls. An empty `text` is
        // honest and contributes no block to the projection; synthesizing prose
        // to fill it would put words in the operator's mouth in a durable record.
        let mut data = serde_json::json!({
            "text": outcome.text,
            "generation": generation,
            "interrupted": outcome.stopped,
            "model": outcome.model,
        });
        if persist_thinking {
            data["thinking"] = serde_json::json!(thinking_blocks);
        }
        let assistant_id = match append_blocking(
            &state.sessions,
            &stream,
            NewEvent {
                actor: assistant_actor(&operator),
                event_type: "assistant_message".to_string(),
                data,
            },
        )
        .await
        {
            Ok(ev) => ev.id,
            Err(e) => {
                eprintln!("log_write_failed appending assistant_message for stream {stream:?}: {e:?}");
                return;
            }
        };

        if calls.is_empty() {
            return; // the model spoke; the turn is over
        }

        // The bound already withheld the tools and set `tool_choice: none` for
        // the round just finished — that was the last chance to produce prose.
        // If a call came back anyway, answer it so the log stays valid and then
        // stop. A loop whose only bound is the provider honouring its own
        // request is not bounded; a hostile or buggy provider would run forever.
        if force_text {
            answer_all(&state, &stream, &assistant_id, &calls, "bound_reached").await;
            eprintln!(
                "stream {stream:?}: provider returned a tool call after tools were withheld \
                 at round {round}; ending the turn"
            );
            return;
        }

        // Record every call, then execute. Both carry `parent` so the batch is
        // a filter rather than a positional walk.
        let mut produced_content = false;
        for (id, name, input) in &calls {
            let use_ev = NewEvent {
                actor: assistant_actor(&operator),
                event_type: "tool_use".to_string(),
                data: serde_json::json!({ "id": id, "name": name, "input": input }),
            };
            if let Err(e) = append_child(&state, &stream, &assistant_id, use_ev).await {
                eprintln!("log_write_failed appending tool_use for stream {stream:?}: {e:?}");
                return;
            }

            let ctx = crate::tools::ToolCtx {
                root: state.root.clone(),
                may_write_composition: write_grant,
            };
            let (n, i) = (name.clone(), input.clone());
            let executed = tokio::task::spawn_blocking(move || crate::tools::dispatch(&n, &i, &ctx))
                .await
                .unwrap_or_else(|_| Err(crate::tools::ToolError { status: "tool_panicked", detail: String::new() }));

            let (status, content, is_error) = match executed {
                Ok(out) => {
                    for effect in &out.effects {
                        // A `match`, not a `let`-destructure: when the git
                        // verbs add a second variant, a missing case must be a
                        // compile error rather than a silently dropped event.
                        let ev = match effect {
                            crate::tools::Effect::FileChanged { path, op, before, after } => NewEvent {
                                // `Role::Tool`, matching `tool_result` — the
                                // model asked, the tool answered.
                                actor: Actor { role: Role::Tool, instance: None, principal: None },
                                event_type: "file_changed".to_string(),
                                data: serde_json::json!({
                                    "path": path, "op": op, "before": before, "after": after,
                                }),
                            },
                        };
                        // Appended BEFORE the tool_result below: the effect
                        // preceded the report of it, and a replay should read
                        // that way.
                        if let Err(e) = append_child(&state, &stream, &assistant_id, ev).await {
                            // I-5: a failed log write surfaces to its writer.
                            // Deliberately NOT a `return` like the tool_use
                            // failure above — the file is already on disk, so
                            // this is a record we failed to keep, not a
                            // mutation we failed to make, and abandoning the
                            // turn would leave the model with no tool_result
                            // for a write that actually happened.
                            eprintln!(
                                "log_write_failed appending file_changed for stream {stream:?}: {e:?}"
                            );
                        }
                    }
                    produced_content |= !out.text.is_empty();
                    ("ok".to_string(), out.text, false)
                }
                Err(err) => {
                    // Escalate rather than repeat: a tool that just failed is
                    // not offered again this turn.
                    failed_tools.insert(name.clone());
                    (err.status.to_string(), err.detail, true)
                }
            };
            if let Err(e) = append_tool_result(&state, &stream, &assistant_id, id, &status, &content, is_error).await {
                eprintln!("log_write_failed appending tool_result for stream {stream:?}: {e:?}");
                return;
            }
        }

        stalled = if produced_content { 0 } else { stalled + 1 };
    }
}

/// The runaway guard.
///
/// Deliberately high. Every surveyed harness sets this between 50 and infinity,
/// and ycc's source argues the case directly: a small cap guillotines ordinary
/// multi-step work mid-task. This is a cost backstop against a degenerate loop,
/// not a budget — the no-progress detector below is what catches a model going
/// in circles, and it fires far sooner.
pub const MAX_ROUNDS: usize = 64;

/// Consecutive rounds yielding no new tool output before the loop calls it stuck.
///
/// A bare round counter never notices a model reading the same empty directory
/// four times; every specimen pairs its cap with a detector that RESETS on
/// progress, and this is that.
const MAX_STALLED_ROUNDS: usize = 3;

/// Append an event that belongs to a tool batch, linked to the assistant event
/// that owns it.
async fn append_child(
    state: &SharedState,
    stream: &str,
    parent: &str,
    ev: NewEvent,
) -> Result<EventEnvelope, SessionError> {
    let sessions = state.sessions.clone();
    let stream = stream.to_string();
    let parent = parent.to_string();
    match tokio::task::spawn_blocking(move || sessions.append_child(&stream, &parent, ev)).await {
        Ok(result) => result,
        Err(_) => Err(SessionError::Log(io::Error::other("append blocking task panicked"))),
    }
}

async fn append_tool_result(
    state: &SharedState,
    stream: &str,
    parent: &str,
    tool_use_id: &str,
    status: &str,
    content: &str,
    is_error: bool,
) -> Result<EventEnvelope, SessionError> {
    append_child(
        state,
        stream,
        parent,
        NewEvent {
            // D21: `Role::Tool` has been in the schema since v0.1 and
            // constructed nowhere. The alternative — reusing the operator
            // actor — would make the log assert the model produced its own
            // tool results, which is false and undercuts the whole point of a
            // status the model cannot overwrite.
            actor: Actor { role: Role::Tool, instance: None, principal: None },
            event_type: "tool_result".to_string(),
            data: serde_json::json!({
                "toolUseId": tool_use_id,
                "status": status,
                "content": content,
                "isError": is_error,
            }),
        },
    )
    .await
}

/// Answer every call in `calls` with the same status — the interrupt and
/// crash paths, where no tool ran.
async fn answer_all(
    state: &SharedState,
    stream: &str,
    parent: &str,
    calls: &[(String, String, serde_json::Value)],
    status: &str,
) {
    for (id, _, _) in calls {
        if let Err(e) = append_tool_result(state, stream, parent, id, status, "the call did not run", true).await {
            eprintln!("log_write_failed answering {id} on stream {stream:?}: {e:?}");
        }
    }
}

async fn record_interrupt(
    state: &SharedState,
    stream: &str,
    operator: &str,
    generation: &str,
    text_so_far: &str,
    model: &str,
) {
    let ev = NewEvent {
        actor: Actor { role: Role::User, instance: None, principal: None },
        event_type: "interrupt".to_string(),
        data: serde_json::json!({}),
    };
    if let Err(e) = append_blocking(&state.sessions, stream, ev).await {
        eprintln!("log_write_failed appending interrupt for stream {stream:?}: {e:?}");
    }
    let ev = NewEvent {
        actor: assistant_actor(operator),
        event_type: "assistant_message".to_string(),
        data: serde_json::json!({
            "text": text_so_far,
            "generation": generation,
            "interrupted": true,
            "model": model,
        }),
    };
    if let Err(e) = append_blocking(&state.sessions, stream, ev).await {
        eprintln!("log_write_failed appending interrupted assistant_message for stream {stream:?}: {e:?}");
    }
}

/// Read, project, and heal until the projection is something the provider will
/// accept — or give up loudly.
///
/// Two repairs, both heal-FORWARD: they append a real event and re-project,
/// rather than manufacturing a block the log has no record of. P-2 wants
/// injected content recorded first, I-1 says correction is a new event, and a
/// repair invisible in the log is one nobody can audit.
async fn project_healing(
    state: &SharedState,
    stream: &str,
    operator: &str,
    generation: &str,
    model: &str,
) -> Option<crate::projection::ModelTurn> {
    // Bounded: each pass must consume at least one distinct fault, so a
    // repair that fails to make progress cannot spin.
    for _ in 0..4 {
        let events = match read_all(&state.sessions, stream).await {
            Ok(events) => events,
            Err(e) => {
                eprintln!("failed to read stream {stream:?}: {e}");
                fail_turn(&state.sessions, stream, operator, generation, "log_unreadable", model).await;
                return None;
            }
        };

        match project_context(&events, &state.root) {
            Ok(turn) => return Some(turn),

            // `path` names the file that moved, which is the diagnostic. The
            // repair re-derives the WHOLE declared list — a boot event is one
            // event covering N files, so re-recording half of it would leave
            // the stream projecting a system prompt assembled from two
            // different moments.
            Err(ProjectionError::BootDrift { path }) => {
                let paths = declared_boot_paths(&events);
                if paths.is_empty() {
                    eprintln!("drift on {path:?} but no boot event declares it on {stream:?}");
                    fail_turn(&state.sessions, stream, operator, generation, "boot_unreadable", model).await;
                    return None;
                }
                if let Err(code) = rerecord_boot(state, stream, &paths).await {
                    fail_turn(&state.sessions, stream, operator, generation, code, model).await;
                    return None;
                }
            }

            // DD-1: a manifest moved under the session. Re-derive and append a
            // fresh event — I-1, correction is a new event — through the same
            // writer instance creation uses, so the repair cannot produce a
            // differently-shaped composition than the original.
            Err(ProjectionError::CompositionDrift) => {
                if let Err(code) =
                    crate::routes::instances::append_composition_event(state, stream).await
                {
                    fail_turn(&state.sessions, stream, operator, generation, code, model).await;
                    return None;
                }
            }

            // The crash case: the engine died between appending a call and
            // appending its answer, so the log holds a shape the provider
            // refuses. Every stranded id is answered, not just the first.
            Err(ProjectionError::UnansweredToolUses { ids }) => {
                for id in ids {
                    let parent = events
                        .iter()
                        .find(|e| {
                            e.event_type == "tool_use"
                                && e.data.get("id").and_then(|v| v.as_str()) == Some(id.as_str())
                        })
                        .and_then(|e| e.parent.clone())
                        .unwrap_or_default();
                    if let Err(e) = append_tool_result(
                        state, stream, &parent, &id,
                        "no_result_recorded",
                        "the engine stopped before this call completed",
                        true,
                    )
                    .await
                    {
                        eprintln!("log_write_failed healing {id} on stream {stream:?}: {e:?}");
                        fail_turn(&state.sessions, stream, operator, generation, "heal_failed", model).await;
                        return None;
                    }
                }
            }

            Err(ProjectionError::BootUnreadable { .. }) | Err(ProjectionError::Transient) => {
                fail_turn(&state.sessions, stream, operator, generation, "boot_unreadable", model).await;
                return None;
            }
        }
    }

    eprintln!("projection did not converge after repairs on stream {stream:?}");
    fail_turn(&state.sessions, stream, operator, generation, "projection_unstable", model).await;
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::store::LogStore;
    use crate::provider::{self, KeylessProvider};
    use crate::sessions::Sessions;
    use crate::state::{AppState, ModelConfig};

    fn bare_event(seq: u64, id: &str, event_type: &str, data: serde_json::Value) -> EventEnvelope {
        EventEnvelope {
            stream: "s1".to_string(),
            id: id.to_string(),
            seq,
            ts: "2026-08-02T00:00:00Z".to_string(),
            actor: Actor {
                role: Role::System,
                instance: None,
                principal: None,
            },
            event_type: event_type.to_string(),
            thread: None,
            parent: None,
            version: 1,
            cost: None,
            data,
        }
    }

    #[test]
    fn the_composition_grant_is_read_from_instance_created_and_defaults_off() {
        // WD-9. Per-instance facts live in `instance_created.data` — this reads
        // the grant exactly as `operator_label` reads the operator, so the flag
        // is log-derived rather than held in a side table that could disagree
        // with the log. The default is asserted rather than assumed: a grant
        // that defaults on is the whole protection gone.
        let granted = vec![bare_event(
            1,
            "i1",
            "instance_created",
            serde_json::json!({ "operator": "tycho", "startedAt": "t", "mayWriteComposition": true }),
        )];
        let plain = vec![bare_event(
            1,
            "i1",
            "instance_created",
            serde_json::json!({ "operator": "tycho", "startedAt": "t" }),
        )];

        assert!(may_write_composition(&granted));
        assert!(!may_write_composition(&plain));
    }

    #[test]
    fn the_model_is_read_from_instance_created_and_absent_when_unrecorded() {
        let with = vec![bare_event(
            1,
            "i1",
            "instance_created",
            serde_json::json!({ "operator": "tycho", "model": "zen/deepseek-v4-flash" }),
        )];
        assert_eq!(model_for(&with).as_deref(), Some("zen/deepseek-v4-flash"));

        // An instance created before this field existed. It must resolve to
        // None so the caller falls back to the engine default — never fail,
        // never a bare empty string.
        let without = vec![bare_event(
            1,
            "i1",
            "instance_created",
            serde_json::json!({ "operator": "tycho" }),
        )];
        assert_eq!(model_for(&without), None);
    }

    fn model_config() -> ModelConfig {
        ModelConfig {
            model: "claude-sonnet-5".to_string(),
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
                        principal: None,
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
                        principal: None,
                    },
                    event_type: "user_message".to_string(),
                    data: serde_json::json!({ "text": "hi", "clientKey": "c1" }),
                },
            )
            .unwrap();
        id
    }

    /// One scripted outcome per provider call, so a test can drive a
    /// multi-round turn. `ScriptedProvider` replays the same script on every
    /// call, which would make round 2 identical to round 1 — and trip the
    /// no-progress detector on a healthy turn.
    #[derive(Debug)]
    struct RoundScript {
        rounds: std::sync::Mutex<std::collections::VecDeque<TurnOutcome>>,
    }

    impl RoundScript {
        fn new(rounds: Vec<TurnOutcome>) -> Self {
            RoundScript {
                rounds: std::sync::Mutex::new(rounds.into()),
            }
        }
    }

    #[async_trait::async_trait]
    impl crate::provider::ModelProvider for RoundScript {
        async fn run_turn(
            &self,
            _req: crate::provider::TurnRequest,
            _sink: mpsc::Sender<String>,
            _cancel: watch::Receiver<bool>,
        ) -> Result<TurnOutcome, crate::provider::ProviderError> {
            // A real provider cannot call a tool that was not offered. The
            // double must not either, or the loop's bound looks effective when
            // it is only being obeyed.
            if _req.tools.is_empty() {
                return Ok(TurnOutcome {
                    text: "forced to speak".to_string(),
                    blocks: vec![crate::projection::ContentBlock::text("forced to speak")],
                    stop_reason: Some("end_turn".to_string()),
                    stopped: false,
                    model: "round-script".to_string(),
                });
            }
            let next = self.rounds.lock().unwrap().pop_front();
            Ok(next.unwrap_or(TurnOutcome {
                text: "script exhausted".to_string(),
                blocks: vec![crate::projection::ContentBlock::text("script exhausted")],
                stop_reason: Some("end_turn".to_string()),
                stopped: false,
                model: "round-script".to_string(),
            }))
        }
    }

    fn says(text: &str) -> TurnOutcome {
        TurnOutcome {
            text: text.to_string(),
            blocks: vec![crate::projection::ContentBlock::text(text)],
            stop_reason: Some("end_turn".to_string()),
            stopped: false,
            model: "round-script".to_string(),
        }
    }

    fn calls(id: &str, name: &str) -> TurnOutcome {
        calls_with(id, name, serde_json::json!({}))
    }

    fn calls_with(id: &str, name: &str, input: serde_json::Value) -> TurnOutcome {
        TurnOutcome {
            text: String::new(),
            blocks: vec![crate::projection::ContentBlock::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
                input,
            }],
            stop_reason: Some("tool_use".to_string()),
            stopped: false,
            model: "round-script".to_string(),
        }
    }

    async fn state_with(
        provider: std::sync::Arc<dyn crate::provider::ModelProvider>,
        sessions: Arc<Sessions>,
        root: &std::path::Path,
    ) -> SharedState {
        Arc::new(AppState {
            root: root.canonicalize().unwrap(),
            sessions,
            model: model_config(),
            providers: provider::fixed(provider),
            models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
            registry_dir: tempfile::tempdir().unwrap().keep(),
            anthropic_key_present: false,
            zen_key_present: false,
            boot: None,
        })
    }

    #[tokio::test]
    async fn a_write_through_a_real_turn_appends_file_changed_before_its_tool_result() {
        // MUTATION-CHECKED. The unit tests stop at `dispatch` and prove the
        // verb produces an Effect; nothing else proves the LOOP turns that
        // Effect into a durable event. That is the whole of WD-10's seam, and
        // a body that describes an effect no one records is worth nothing.
        //
        // Order is asserted, not incidental: `file_changed` precedes
        // `tool_result` because the effect preceded the report of it, and a
        // replay should read that way.
        let data_dir = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("modules.toml"), "[[repos]]\nname='r'\npath='p'\n")
            .unwrap();
        let store = LogStore::open(data_dir.path()).unwrap();
        let sessions = Arc::new(Sessions::new(store));
        let id = create_instance(&sessions, Some("tycho")).await;

        let provider = std::sync::Arc::new(RoundScript::new(vec![
            calls_with(
                "toolu_w1",
                "write_file",
                serde_json::json!({ "path": "notes.md", "content": "written by a turn\n" }),
            ),
            says("done"),
        ]));
        let state = state_with(provider, sessions.clone(), root.path()).await;

        let (_c, cancel_rx) = watch::channel(false);
        run_turn(state, id.clone(), "gen-1".to_string(), cancel_rx).await;

        // The write is real, on the real disk.
        assert_eq!(
            std::fs::read_to_string(root.path().join("notes.md")).unwrap(),
            "written by a turn\n"
        );

        let events = sessions.store().read_from(&id, 0).unwrap();
        let types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
        assert_eq!(
            types,
            vec![
                "instance_created",
                "user_message",
                "assistant_message",
                "tool_use",
                "file_changed",
                "tool_result",
                "assistant_message"
            ],
            "{types:?}"
        );

        let changed = events
            .iter()
            .find(|e| e.event_type == "file_changed")
            .unwrap();
        assert_eq!(changed.data["path"], "notes.md");
        assert_eq!(changed.data["op"], "created");
        assert!(changed.data["before"].is_null(), "a create has no prior content");
        assert_eq!(
            changed.data["after"],
            crate::hash::sha256_hex(b"written by a turn\n")
        );
        // `Role::Tool`, matching `tool_result` — the model asked, the tool
        // answered.
        assert_eq!(changed.actor.role, Role::Tool);
        // Durable, so it must carry a real seq (I-4: 0 is transient).
        assert!(changed.seq > 0);
    }

    #[tokio::test]
    async fn captured_thinking_lands_on_the_assistant_message_in_wire_shape() {
        // The round that thinks and calls carries its thinking on the
        // assistant_message, wire-shaped; a round that captured none must not
        // carry the key at all, so pre-thinking events stay byte-identical.
        // The tool round's text is EMPTY on purpose — calls count as content
        // for the persistence guard, and this is the shape the live probe
        // elicited (thinking straight into tool_use, no prose).
        let data_dir = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("modules.toml"), "[[repos]]\nname='r'\npath='p'\n")
            .unwrap();
        let store = LogStore::open(data_dir.path()).unwrap();
        let sessions = Arc::new(Sessions::new(store));
        let id = create_instance(&sessions, Some("tycho")).await;

        let provider = std::sync::Arc::new(RoundScript::new(vec![
            TurnOutcome {
                text: String::new(),
                blocks: vec![
                    crate::projection::ContentBlock::Thinking {
                        thinking: "let me look".to_string(),
                        signature: "sig-1".to_string(),
                    },
                    crate::projection::ContentBlock::ToolUse {
                        id: "toolu_th1".to_string(),
                        name: "get_composition".to_string(),
                        input: serde_json::json!({}),
                    },
                ],
                stop_reason: Some("tool_use".to_string()),
                stopped: false,
                model: "round-script".to_string(),
            },
            says("done"),
        ]));
        let state = state_with(provider, sessions.clone(), root.path()).await;

        let (_c, cancel_rx) = watch::channel(false);
        run_turn(state, id.clone(), "gen-1".to_string(), cancel_rx).await;

        let events = sessions.store().read_from(&id, 0).unwrap();
        let with_thinking: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == "assistant_message" && e.data.get("thinking").is_some())
            .collect();
        assert_eq!(with_thinking.len(), 1, "exactly the tool round carries thinking");
        assert_eq!(
            with_thinking[0].data["thinking"],
            serde_json::json!([
                {"type": "thinking", "thinking": "let me look", "signature": "sig-1"}
            ])
        );

        let last = events
            .iter()
            .rfind(|e| e.event_type == "assistant_message")
            .unwrap();
        assert!(last.data.get("thinking").is_none());
    }

    #[tokio::test]
    async fn a_refused_write_emits_no_file_changed_at_all() {
        // D-1's central discipline: this records EFFECT, not intent. The moment
        // it logs a refusal it is a second copy of `tool_use` and worth
        // nothing. A guard denial is a `tool_result` error and nothing else.
        let data_dir = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("modules.toml"), "[[repos]]\nname='r'\npath='p'\n")
            .unwrap();
        let store = LogStore::open(data_dir.path()).unwrap();
        let sessions = Arc::new(Sessions::new(store));
        let id = create_instance(&sessions, Some("tycho")).await;

        let provider = std::sync::Arc::new(RoundScript::new(vec![
            calls_with(
                "toolu_w1",
                "write_file",
                serde_json::json!({ "path": "secrets.json", "content": "{}" }),
            ),
            says("refused"),
        ]));
        let state = state_with(provider, sessions.clone(), root.path()).await;

        let (_c, cancel_rx) = watch::channel(false);
        run_turn(state, id.clone(), "gen-1".to_string(), cancel_rx).await;

        assert!(!root.path().join("secrets.json").exists());

        let events = sessions.store().read_from(&id, 0).unwrap();
        assert!(
            !events.iter().any(|e| e.event_type == "file_changed"),
            "a refusal logged an effect it never had"
        );
        let result = events
            .iter()
            .find(|e| e.event_type == "tool_result")
            .unwrap();
        assert_eq!(result.data["status"], "denied_credential");
        assert_eq!(result.data["isError"], true);
    }

    #[tokio::test]
    async fn a_tool_call_is_executed_and_answered_and_the_turn_continues() {
        let data_dir = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("modules.toml"), "[[repos]]\nname='r'\npath='p'\n")
            .unwrap();
        let store = LogStore::open(data_dir.path()).unwrap();
        let sessions = Arc::new(Sessions::new(store));
        let id = create_instance(&sessions, Some("tycho")).await;

        let provider = std::sync::Arc::new(RoundScript::new(vec![
            calls("toolu_01A", "get_composition"),
            says("there is one repo, r"),
        ]));
        let state = state_with(provider, sessions.clone(), root.path()).await;

        let (_c, cancel_rx) = watch::channel(false);
        run_turn(state, id.clone(), "gen-1".to_string(), cancel_rx).await;

        let events = sessions.store().read_from(&id, 0).unwrap();
        let types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
        // The assistant event precedes its calls and is their batch parent. Its
        // `text` is empty here — the model called a tool without speaking — and
        // the projection contributes no block for it, because an empty text
        // block is a 400 and an assistant message of pure tool_use is legal.
        assert_eq!(
            types,
            vec![
                "instance_created",
                "user_message",
                "assistant_message",
                "tool_use",
                "tool_result",
                "assistant_message"
            ],
            "{types:?}"
        );

        let tool_use = events.iter().find(|e| e.event_type == "tool_use").unwrap();
        let tool_result = events.iter().find(|e| e.event_type == "tool_result").unwrap();

        assert_eq!(tool_use.data["id"], "toolu_01A");
        assert_eq!(tool_result.data["toolUseId"], "toolu_01A");
        assert_eq!(tool_result.data["status"], "ok");
        assert_eq!(tool_result.data["isError"], false);
        assert!(
            tool_result.data["content"].as_str().unwrap().contains("\"r\""),
            "the real composition should have been read: {}",
            tool_result.data["content"]
        );

        // D21: the log must not claim the operator produced the tool result.
        assert_eq!(tool_use.actor.role, Role::Operator, "the model asked");
        assert_eq!(tool_result.actor.role, Role::Tool, "the tool answered");

        // The batch link that makes eviction and orphan repair work.
        assert!(tool_use.parent.is_some(), "tool_use needs a batch parent");
        assert_eq!(tool_use.parent, tool_result.parent, "same batch");

        let last = events.last().unwrap();
        assert_eq!(last.data["text"], "there is one repo, r");
    }

    #[tokio::test]
    async fn a_session_can_find_a_file_and_read_a_string_no_earlier_event_carried() {
        // **Success criterion 1**, and the reason it replaced v1's: the
        // composition carries every repo's `note`, so "the model named a repo"
        // was satisfiable by reciting. A string that exists only inside a file
        // the model chose to open is not. This is the whole of 2b in one test —
        // list to discover, read to know.
        let data_dir = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("modules.toml"), "[[repos]]\nname='r'\npath='p'\n")
            .unwrap();
        std::fs::create_dir(root.path().join("notes")).unwrap();
        std::fs::write(
            root.path().join("notes/board.md"),
            "# board\n\nthe cormorant dries its wings\n",
        )
        .unwrap();

        let store = LogStore::open(data_dir.path()).unwrap();
        let sessions = Arc::new(Sessions::new(store));
        let id = create_instance(&sessions, Some("tycho")).await;

        let provider = std::sync::Arc::new(RoundScript::new(vec![
            calls_with("t1", "list_directory", serde_json::json!({ "path": "notes" })),
            calls_with(
                "t2",
                "read_file",
                serde_json::json!({ "path": "notes/board.md" }),
            ),
            says("the board says the cormorant dries its wings"),
        ]));
        let state = state_with(provider, sessions.clone(), root.path()).await;

        let (_c, cancel_rx) = watch::channel(false);
        run_turn(state, id.clone(), "gen-1".to_string(), cancel_rx).await;

        let events = sessions.store().read_from(&id, 0).unwrap();
        let results: Vec<&EventEnvelope> = events
            .iter()
            .filter(|e| e.event_type == "tool_result")
            .collect();
        assert_eq!(results.len(), 2, "both calls must be answered");

        // The listing is what makes the file findable without being told.
        let listing = results[0].data["content"].as_str().unwrap();
        assert_eq!(results[0].data["status"], "ok", "{listing}");
        assert!(listing.contains("board.md"), "{listing}");

        let secret = "cormorant dries its wings";
        let read = results[1].data["content"].as_str().unwrap();
        assert!(read.contains(secret), "the file was not actually read: {read}");
        assert!(read.contains("[end of file]"), "{read}");

        // The ungameable half: nothing before the read carried the string, so
        // it cannot have been recited.
        let read_at = events
            .iter()
            .position(|e| e.id == results[1].id)
            .expect("the result is in the log");
        for earlier in &events[..read_at] {
            assert!(
                !earlier.data.to_string().contains(secret),
                "{} carried the string before the read: {}",
                earlier.event_type,
                earlier.data
            );
        }
    }

    #[tokio::test]
    async fn a_failing_tool_is_answered_with_its_machine_code_and_the_turn_continues() {
        let data_dir = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        let store = LogStore::open(data_dir.path()).unwrap();
        let sessions = Arc::new(Sessions::new(store));
        let id = create_instance(&sessions, None).await;

        let provider = std::sync::Arc::new(RoundScript::new(vec![
            calls("toolu_01B", "no_such_tool"),
            says("that tool does not exist"),
        ]));
        let state = state_with(provider, sessions.clone(), root.path()).await;

        let (_c, cancel_rx) = watch::channel(false);
        run_turn(state, id.clone(), "gen-1".to_string(), cancel_rx).await;

        let events = sessions.store().read_from(&id, 0).unwrap();
        let result = events.iter().find(|e| e.event_type == "tool_result").unwrap();

        assert_eq!(result.data["status"], "unknown_tool");
        assert_eq!(result.data["isError"], true);
        assert!(
            !result.data["content"].as_str().unwrap().is_empty(),
            "is_error with empty content is a 400 — every failure must render something"
        );
        // Success criterion 2: a refusal does not end the turn.
        assert_eq!(events.last().unwrap().event_type, "assistant_message");
        assert_eq!(events.last().unwrap().data["text"], "that tool does not exist");
    }

    #[tokio::test]
    async fn the_runaway_bound_forces_text_rather_than_looping_forever() {
        let data_dir = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        let store = LogStore::open(data_dir.path()).unwrap();
        let sessions = Arc::new(Sessions::new(store));
        let id = create_instance(&sessions, None).await;

        // A model that only ever calls tools. Without a bound this never ends.
        let endless: Vec<TurnOutcome> = (0..200)
            .map(|i| calls(&format!("toolu_{i}"), "get_composition"))
            .collect();
        let provider = std::sync::Arc::new(RoundScript::new(endless));
        let state = state_with(provider, sessions.clone(), root.path()).await;

        let (_c, cancel_rx) = watch::channel(false);
        run_turn(state, id.clone(), "gen-1".to_string(), cancel_rx).await;

        let events = sessions.store().read_from(&id, 0).unwrap();
        let rounds = events.iter().filter(|e| e.event_type == "tool_use").count();

        assert!(rounds > 1, "it should have looped at all: {rounds}");
        assert!(
            rounds <= MAX_ROUNDS,
            "the runaway guard did not hold: {rounds} rounds"
        );
        assert_eq!(
            events.last().unwrap().event_type,
            "assistant_message",
            "the turn must end with something the person can read"
        );
    }

    #[tokio::test]
    async fn a_real_tool_round_survives_the_validating_provider() {
        // The test that makes the others mean something. `RoundScript` answers
        // anything, so "the turn completed" says nothing about whether the
        // message list we built would survive the API. Wrapping it in
        // `ValidatingProvider` enforces the five measured 400s on every round —
        // orphaned calls, orphaned results, results that do not lead their
        // turn, empty text blocks, and is_error with empty content.
        let data_dir = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("modules.toml"), "[[repos]]\nname='r'\npath='p'\n")
            .unwrap();
        let store = LogStore::open(data_dir.path()).unwrap();
        let sessions = Arc::new(Sessions::new(store));
        let id = create_instance(&sessions, Some("tycho")).await;

        let provider = std::sync::Arc::new(crate::provider::ValidatingProvider::new(
            RoundScript::new(vec![
                calls("toolu_01A", "get_composition"),
                calls("toolu_01B", "no_such_tool"),
                says("one repo, and that other tool does not exist"),
            ]),
        ));
        let state = state_with(provider, sessions.clone(), root.path()).await;

        let (_c, cancel_rx) = watch::channel(false);
        run_turn(state, id.clone(), "gen-1".to_string(), cancel_rx).await;

        let events = sessions.store().read_from(&id, 0).unwrap();
        let last = events.last().unwrap();

        // A refusal from the validator surfaces as provider_api_400. Reaching
        // the scripted final answer means every intermediate projection was
        // one the real API would have accepted.
        assert_eq!(
            last.data.get("error"),
            None,
            "a round built an invalid message list: {:?}",
            last.data
        );
        assert_eq!(
            last.data["text"],
            "one repo, and that other tool does not exist"
        );
        // Both rounds happened, including the one whose tool failed.
        assert_eq!(events.iter().filter(|e| e.event_type == "tool_result").count(), 2);
    }

    #[tokio::test]
    async fn a_call_stranded_by_a_crash_is_healed_and_the_next_turn_runs() {
        // Success criterion 3. The engine dies between appending a `tool_use`
        // and appending its answer; the log then holds a shape the provider
        // refuses. Heal-forward: a real `tool_result` is APPENDED and the turn
        // proceeds — the repair is in the log where it can be audited, not
        // manufactured inside the projection where nothing records it.
        let data_dir = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        let store = LogStore::open(data_dir.path()).unwrap();
        let sessions = Arc::new(Sessions::new(store));
        let id = create_instance(&sessions, None).await;

        // Hand-append the wreckage a killed engine leaves behind.
        let assistant = sessions
            .append(
                &id,
                NewEvent {
                    actor: assistant_actor("dispatcher"),
                    event_type: "assistant_message".to_string(),
                    data: serde_json::json!({"text": "", "generation": "g0", "interrupted": false}),
                },
            )
            .unwrap();
        sessions
            .append_child(
                &id,
                &assistant.id,
                NewEvent {
                    actor: assistant_actor("dispatcher"),
                    event_type: "tool_use".to_string(),
                    data: serde_json::json!({"id": "toolu_orphan", "name": "get_composition", "input": {}}),
                },
            )
            .unwrap();

        let provider = std::sync::Arc::new(crate::provider::ValidatingProvider::new(
            RoundScript::new(vec![says("picking up where that left off")]),
        ));
        let state = state_with(provider, sessions.clone(), root.path()).await;

        let (_c, cancel_rx) = watch::channel(false);
        run_turn(state, id.clone(), "gen-1".to_string(), cancel_rx).await;

        let events = sessions.store().read_from(&id, 0).unwrap();
        let heal = events
            .iter()
            .find(|e| e.event_type == "tool_result")
            .expect("the orphan must be answered by a real appended event");

        assert_eq!(heal.data["toolUseId"], "toolu_orphan");
        assert_eq!(heal.data["status"], "no_result_recorded");
        assert_eq!(heal.data["isError"], true);
        assert_eq!(
            heal.parent.as_deref(),
            Some(assistant.id.as_str()),
            "the heal joins the batch it repairs"
        );

        let last = events.last().unwrap();
        assert_eq!(last.data.get("error"), None, "the turn should have run: {:?}", last.data);
        assert_eq!(last.data["text"], "picking up where that left off");
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
            providers: provider::fixed(Arc::new(KeylessProvider)),
            models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
            registry_dir: tempfile::tempdir().unwrap().keep(),
            anthropic_key_present: false,
            zen_key_present: false,
            boot: None,
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
        // does — the RESOLVED model (`model_for`, falling back to
        // `state.model.model`), not a raw config string, since a failure
        // path never gets a `TurnOutcome::model` to report.
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
            providers: provider::fixed(Arc::new(KeylessProvider)),
            models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
            registry_dir: tempfile::tempdir().unwrap().keep(),
            anthropic_key_present: false,
            zen_key_present: false,
            boot: None,
        });

        let generation = uuid::Uuid::new_v4().to_string();
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        run_turn(state, id.clone(), generation, cancel_rx).await;

        let events = sessions.store().read_from(&id, 0).unwrap();
        let last = events.last().unwrap();
        assert_eq!(last.actor.instance.as_deref(), Some("dispatcher"));
    }
}
