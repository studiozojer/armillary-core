//! Projection and flattening — P-1/P-3/P-4.
//!
//! `project_context` is the **total reducer** (P-3): a two-pass fold over a
//! stream's durable events into the shape a model actually sees. Pass one
//! collects the evicted-id set from `context_evict` events; pass two matches
//! every durable type explicitly and either contributes nothing, extends
//! `system`, or appends a `ProviderMessage` — an evicted event's arm never
//! runs (P-1: evicted is not deleted from the *log*, but it is absent from
//! *this* projection, which is exactly the distinction P-1 draws).
//!
//! `event_type` is a `String` (I-2), not a closed Rust enum, so the match
//! below cannot be exhaustive in the compiler's sense — there is always a
//! catch-all binding. What stands in for compile-time totality here is two
//! things together: (1) the catch-all never contributes silently — it
//! appends a visible `[unhandled event type: ...]` narrative line rather
//! than dropping the event, so a gap in coverage shows up in the transcript
//! itself; and (2) `HANDLED_TYPES` below, checked against `DURABLE_TYPES` by
//! a test, so a ninth durable type fails CI the moment it's declared, not
//! whenever someone notices a blank spot in a session.
//!
//! Flattening (typed event -> `ProviderMessage`, P-4) happens in the same
//! pass as projection here rather than as a textually separate function —
//! but it is still a **separate stage** in the P-4 sense that matters: nothing
//! upstream of the per-event match loses its type before this point, and the
//! provider-message shape this function returns is never itself persisted or
//! matched on again. The boot-file read is the one exception to "pure": it is
//! kept at this single edge (P-2's projection boundary) rather than smeared
//! across callers.

use crate::hash::sha256_hex;
use crate::log::envelope::EventEnvelope;
#[cfg(test)]
use crate::log::envelope::DURABLE_TYPES;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// What a model actually sees for one turn: an optional system prompt (from
/// the last-standing `boot` event) plus the flattened, alternation-merged
/// message list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelTurn {
    pub system: Option<String>,
    pub messages: Vec<ProviderMessage>,
}

/// One flattened message. `content` has already had same-role neighbors
/// merged (see `merge_consecutive`) — this is the P-4 provider shape, not
/// the log's typed shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderMessage {
    pub role: ProviderRole,
    pub content: String,
}

/// Anthropic's two-role wire vocabulary — deliberately narrower than
/// `log::envelope::Role` (which has five). `Actor::role` names *who did it*
/// in the log; `ProviderRole` names *which of the two buckets the provider
/// channel accepts*, which is why `dispatch`/`return` narrative lines below
/// are rendered as `User`, not invented as a third role the wire has no slot
/// for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderRole {
    User,
    Assistant,
}

/// Failure modes specific to the boot-read edge — the only I/O this
/// otherwise-pure stage performs. Nothing else in `project_context` can
/// fail: an event with a missing/malformed field degrades to an empty
/// string rather than erroring, because the reducer's job is to be total
/// over *types* (P-3), not to validate payload shape a schema doesn't pin
/// yet (`schema/event.schema.json`: "per-type payload schemas are a later
/// version").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionError {
    /// The file at `data.path` no longer hashes to `data.sha256` — content
    /// moved under the boot event's feet. The caller (Task 11) re-records a
    /// fresh `boot` event; this function only detects the drift.
    BootDrift { path: String },
    /// `data.path` could not be read at all — missing, permission-denied,
    /// not valid UTF-8, or (deliberately folded into this same variant
    /// rather than a separate `Escaped` case) it canonicalizes outside
    /// `root`. A boot path has no legitimate reason to leave the workspace
    /// root, so "escaped" and "unreadable" are one caller-visible outcome.
    BootUnreadable { path: String },
    /// A transient (`seq == 0`, I-4) event reached the reducer. Durable-only
    /// is a caller invariant (transients are never persisted, so a `seq`
    /// slice pulled from the log never contains one) — this is the
    /// release-mode half of the debug_assert at the top of pass two; see
    /// that assert's comment for why both exist.
    Transient,
}

/// The event types `project_context`'s match gives an explicit arm to.
/// Mirrors the match arms below one-for-one; kept as a separate `const`
/// (rather than, say, deriving it from the match) purely so a test can
/// compare it against `DURABLE_TYPES` without needing a `String`-keyed enum
/// this codebase doesn't have (I-2 events are typed as strings on the wire).
/// P-3: if `DURABLE_TYPES` grows a ninth entry, `handled_types_cover_every_durable_type`
/// below fails until this list — and the match — grow to match it. `cfg(test)`
/// because nothing in `project_context` itself is driven by this list (the
/// match arms are hand-mirrored, not generated from it) — its only job is
/// standing as the other half of that test's comparison.
#[cfg(test)]
const HANDLED_TYPES: &[&str] = &[
    "instance_created",
    "boot",
    "user_message",
    "assistant_message",
    "interrupt",
    "context_evict",
    "dispatch",
    "return",
];

/// Join `rel` under `root` and require the canonical result to stay under
/// `root` — an absolute or `..`-laden `data.path` is not a different kind of
/// error, it is the same `BootUnreadable` a missing file produces, since
/// neither is ever a legitimate boot source.
///
/// `pub(crate)` (not private): the loop (`loop_.rs`) re-records a fresh
/// `boot` event on `BootDrift` by re-reading the SAME path this function
/// resolved the first time — it reuses this resolution rather than
/// re-deriving root-containment logic at a second call site.
pub(crate) fn resolve_boot_path(root: &Path, rel: &str) -> Result<PathBuf, ProjectionError> {
    let unreadable = || ProjectionError::BootUnreadable {
        path: rel.to_string(),
    };
    let root_canonical = root.canonicalize().map_err(|_| unreadable())?;
    let candidate_canonical = root_canonical.join(rel).canonicalize().map_err(|_| unreadable())?;
    if !candidate_canonical.starts_with(&root_canonical) {
        return Err(unreadable());
    }
    Ok(candidate_canonical)
}

/// Merge consecutive same-role messages with a blank line. Anthropic
/// requires strict user/assistant alternation; a raw flattening can produce
/// same-role neighbors (e.g. two `user_message` events back to back, or a
/// `dispatch` narrative line — itself `User`-role — right after a real user
/// message). Whether the *first* message is `User` is left alone: real logs
/// always open on a user message, and enforcing that here would be P-4
/// reaching past its own stage into a concern (what the provider layer
/// requires of message 0) that Task 10 owns.
fn merge_consecutive(raw: Vec<ProviderMessage>) -> Vec<ProviderMessage> {
    let mut merged: Vec<ProviderMessage> = Vec::with_capacity(raw.len());
    for msg in raw {
        match merged.last_mut() {
            Some(last) if last.role == msg.role => {
                last.content.push_str("\n\n");
                last.content.push_str(&msg.content);
            }
            _ => merged.push(msg),
        }
    }
    merged
}

fn text_field(data: &serde_json::Value, key: &str) -> String {
    data.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}

/// P-1: the context window is a projection of the log. P-3: total over
/// `DURABLE_TYPES` — see the module doc for what "total" means when the
/// type tag is a `String`, not a closed enum.
pub fn project_context(
    events: &[EventEnvelope],
    root: &Path,
) -> Result<ModelTurn, ProjectionError> {
    // Pass 1 (P-1): collect the evicted-id set. `context_evict` itself
    // contributes nothing to the transcript in pass 2 — it is consumed
    // entirely here.
    let mut evicted: HashSet<&str> = HashSet::new();
    for ev in events {
        if ev.event_type == "context_evict" {
            if let Some(target) = ev.data.get("target").and_then(|v| v.as_str()) {
                evicted.insert(target);
            }
        }
    }

    // "Last one wins" (see the `"boot"` arm below) extends to VALIDITY, not
    // just content: only the last non-evicted `boot` event's hash is ever
    // checked against disk. Without this, a `boot` event that drifted and
    // was later superseded by a fresh, correct one (Task 11's re-record-on-
    // `BootDrift` recovery) would keep failing every future projection
    // forever — the stale predecessor is water under the bridge once a
    // later boot has replaced it as "the" system prompt.
    let last_boot_id: Option<&str> = events
        .iter()
        .rfind(|ev| ev.event_type == "boot" && !evicted.contains(ev.id.as_str()))
        .map(|ev| ev.id.as_str());

    let mut system: Option<String> = None;
    let mut raw: Vec<ProviderMessage> = Vec::new();

    for ev in events {
        // I-4: seq 0 is transient and MUST NOT reach the durable reducer
        // path. debug_assert catches a wiring bug loudly in dev/test;
        // the explicit check below is the release-mode fallback so a
        // caller mistake fails closed (Transient) instead of silently
        // projecting a hint as if it were a fact.
        debug_assert_ne!(
            ev.seq, 0,
            "project_context: transient event (seq == 0) reached the durable reducer (I-4/P-3)"
        );
        if ev.seq == 0 {
            return Err(ProjectionError::Transient);
        }

        if evicted.contains(ev.id.as_str()) {
            // P-1: evicted, not deleted — the event still lives in the log,
            // it is simply absent from this particular projection.
            continue;
        }

        match ev.event_type.as_str() {
            "instance_created" => {}

            // Multiple `boot` events: LAST one wins. A later boot is a
            // re-record (Task 11 re-records on drift; an operator's boot
            // file can also legitimately change across a long-lived
            // stream), and P-1 treats the log as append-only truth — the
            // *projection* of "current system prompt" is properly a fold
            // that keeps overwriting, not a first-write-wins cache.
            "boot" if Some(ev.id.as_str()) != last_boot_id => {
                // Superseded by a later boot event — see `last_boot_id`'s
                // doc above. Contributes nothing, same as `instance_created`.
            }

            "boot" => {
                let path = text_field(&ev.data, "path");
                let expected_sha = text_field(&ev.data, "sha256");
                let resolved = resolve_boot_path(root, &path)?;
                let bytes = std::fs::read(&resolved).map_err(|_| ProjectionError::BootUnreadable {
                    path: path.clone(),
                })?;
                let actual_sha = sha256_hex(&bytes);
                if actual_sha != expected_sha {
                    return Err(ProjectionError::BootDrift { path });
                }
                let content = String::from_utf8(bytes).map_err(|_| ProjectionError::BootUnreadable {
                    path: path.clone(),
                })?;
                system = Some(content);
            }

            "user_message" => {
                raw.push(ProviderMessage {
                    role: ProviderRole::User,
                    content: text_field(&ev.data, "text"),
                });
            }

            "assistant_message" => {
                let mut content = text_field(&ev.data, "text");
                if ev.data.get("interrupted").and_then(|v| v.as_bool()).unwrap_or(false) {
                    // The `interrupt` event itself carries no separate
                    // arm (below) — the flag on *this* message is the
                    // carrier, so the marker belongs here, not there.
                    content.push_str("\n[generation stopped by user]");
                }
                raw.push(ProviderMessage {
                    role: ProviderRole::Assistant,
                    content,
                });
            }

            // The interrupted flag on the assistant_message carries this;
            // the standalone `interrupt` event is durable (for audit / other
            // projections) but contributes nothing to *this* transcript.
            "interrupt" => {}

            // Consumed entirely in pass 1.
            "context_evict" => {}

            "dispatch" => {
                // Data schema for `dispatch`/`return` isn't pinned by
                // `schema/event.schema.json` yet ("per-type payload schemas
                // are a later version") — field names here are this task's
                // decision, not a cited rule. `operator` is preferred (this
                // workspace's vocabulary: an operator, not an agent); `child`
                // is accepted as a fallback identifier so a caller that only
                // has a bare stream-relationship still renders something
                // besides the empty string.
                //
                // `operator` is required-but-nullable in the wider schema
                // (`string | null`), so a conformant dispatch to an anonymous
                // child carries the key with a JSON `null` value, not an
                // absent key. `.get("operator").or_else(|| .get("child"))`
                // falls through only on a MISSING key — with `operator: null`
                // present, `and_then(as_str)` on it already yields `None` and
                // `or_else` never runs, so every dispatch would render `?`
                // even when `data.child` names the target. Chaining
                // `and_then(as_str)` per field first, THEN falling back,
                // is what actually treats "present but null" the same as
                // "absent".
                let target = ev
                    .data
                    .get("operator")
                    .and_then(|v| v.as_str())
                    .or_else(|| ev.data.get("child").and_then(|v| v.as_str()))
                    .unwrap_or("?");
                let child_stream = text_field(&ev.data, "childStream");
                raw.push(ProviderMessage {
                    role: ProviderRole::User,
                    content: format!("[dispatched {target} — child stream {child_stream}]"),
                });
            }

            "return" => {
                let child = text_field(&ev.data, "child");
                let content = match ev.data.get("summary").and_then(|v| v.as_str()) {
                    Some(summary) => format!("[child {child} returned: {summary}]"),
                    None => format!("[child {child} returned]"),
                };
                raw.push(ProviderMessage {
                    role: ProviderRole::User,
                    content,
                });
            }

            // Never silent (P-3): visible in the transcript rather than
            // dropped, so a gap in coverage shows up where a human or the
            // exhaustiveness test can see it, instead of vanishing.
            other => {
                raw.push(ProviderMessage {
                    role: ProviderRole::User,
                    content: format!("[unhandled event type: {other}]"),
                });
            }
        }
    }

    Ok(ModelTurn {
        system,
        messages: merge_consecutive(raw),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::envelope::{Actor, Role};
    use serde_json::json;

    fn ev(seq: u64, id: &str, event_type: &str, data: serde_json::Value) -> EventEnvelope {
        EventEnvelope {
            stream: "s1".to_string(),
            id: id.to_string(),
            seq,
            ts: "2026-07-27T00:00:00Z".to_string(),
            actor: Actor {
                role: Role::User,
                instance: None,
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
    fn boot_user_assistant_user_flattens_in_order_with_system_set() {
        let dir = tempfile::tempdir().unwrap();
        let bytes: &[u8] = b"# system prompt";
        std::fs::write(dir.path().join("boot.md"), bytes).unwrap();
        let sha = sha256_hex(bytes);

        let events = vec![
            ev(1, "b1", "boot", json!({"path": "boot.md", "sha256": sha})),
            ev(2, "u1", "user_message", json!({"text": "hi"})),
            ev(3, "a1", "assistant_message", json!({"text": "hello"})),
            ev(4, "u2", "user_message", json!({"text": "bye"})),
        ];

        let turn = project_context(&events, dir.path()).unwrap();

        assert_eq!(turn.system.as_deref(), Some("# system prompt"));
        assert_eq!(turn.messages.len(), 3);
        assert_eq!(turn.messages[0].content, "hi");
        assert_eq!(turn.messages[0].role, ProviderRole::User);
        assert_eq!(turn.messages[1].content, "hello");
        assert_eq!(turn.messages[1].role, ProviderRole::Assistant);
        assert_eq!(turn.messages[2].content, "bye");
        assert_eq!(turn.messages[2].role, ProviderRole::User);
    }

    #[test]
    fn a_later_boot_event_overwrites_system_not_the_first() {
        // P-1: the log is append-only, but the *projection* of "current
        // system prompt" is a fold that keeps overwriting — a re-recorded
        // boot supersedes the one before it, so the last boot standing
        // wins, not the first.
        let dir = tempfile::tempdir().unwrap();
        let first_bytes: &[u8] = b"# first boot";
        let second_bytes: &[u8] = b"# second boot, supersedes the first";
        std::fs::write(dir.path().join("first.md"), first_bytes).unwrap();
        std::fs::write(dir.path().join("second.md"), second_bytes).unwrap();

        let events = vec![
            ev(
                1,
                "b1",
                "boot",
                json!({"path": "first.md", "sha256": sha256_hex(first_bytes)}),
            ),
            ev(2, "u1", "user_message", json!({"text": "hi"})),
            ev(
                3,
                "b2",
                "boot",
                json!({"path": "second.md", "sha256": sha256_hex(second_bytes)}),
            ),
        ];

        let turn = project_context(&events, dir.path()).unwrap();

        assert_eq!(turn.system.as_deref(), Some("# second boot, supersedes the first"));
    }

    #[test]
    fn a_stale_drifted_boot_event_superseded_by_a_later_valid_one_does_not_error() {
        // "Last one wins" (the test above) extends to VALIDITY, not just
        // content: Task 11's recovery on `BootDrift` re-records a FRESH
        // `boot` event rather than editing history, so the drifted `b1`
        // stays in the log forever. If every historical boot event were
        // still validated, `b1`'s permanent mismatch would fail EVERY
        // future projection despite `b2` having already fixed things — this
        // pins that only the last standing boot event's hash is checked.
        let dir = tempfile::tempdir().unwrap();
        let correct_bytes: &[u8] = b"# current boot content";
        std::fs::write(dir.path().join("boot.md"), correct_bytes).unwrap();

        let events = vec![
            ev(
                1,
                "b1",
                "boot",
                // Same path, deliberately wrong hash — simulates a boot
                // event nobody ever re-records the FILE for, just a fresh
                // event superseding it.
                json!({"path": "boot.md", "sha256": "0".repeat(64)}),
            ),
            ev(2, "u1", "user_message", json!({"text": "hi"})),
            ev(
                3,
                "b2",
                "boot",
                json!({"path": "boot.md", "sha256": sha256_hex(correct_bytes)}),
            ),
        ];

        let turn = project_context(&events, dir.path()).expect("the stale b1 must not fail this projection");

        assert_eq!(turn.system.as_deref(), Some("# current boot content"));
    }

    #[test]
    fn dispatch_falls_back_to_child_when_operator_is_present_but_null() {
        // The Important finding this test pins: `operator` is
        // required-but-nullable (`string | null`) in the wider schema, so a
        // conformant dispatch to an anonymous child carries the key with a
        // JSON `null` value, not an absent key. The naive
        // `.get("operator").or_else(|| .get("child"))` falls through only on
        // a MISSING key, so with `operator: null` present it never reaches
        // `child` at all and every such dispatch would wrongly render `?`.
        let events = vec![ev(
            1,
            "d1",
            "dispatch",
            json!({"operator": null, "child": "worker-7", "childStream": "child-stream-1"}),
        )];

        let turn = project_context(&events, Path::new(".")).unwrap();

        assert_eq!(
            turn.messages[0].content,
            "[dispatched worker-7 — child stream child-stream-1]"
        );
    }

    #[test]
    fn evicted_user_message_is_absent_and_context_evict_contributes_nothing() {
        let events = vec![
            ev(1, "u1", "user_message", json!({"text": "will be evicted"})),
            ev(2, "u2", "user_message", json!({"text": "stays"})),
            ev(3, "e1", "context_evict", json!({"target": "u1"})),
        ];

        let turn = project_context(&events, Path::new(".")).unwrap();

        assert_eq!(turn.messages.len(), 1);
        assert_eq!(turn.messages[0].content, "stays");
    }

    #[test]
    fn interrupted_assistant_message_gets_trailing_marker() {
        let events = vec![ev(
            1,
            "a1",
            "assistant_message",
            json!({"text": "partial", "interrupted": true}),
        )];

        let turn = project_context(&events, Path::new(".")).unwrap();

        assert_eq!(turn.messages[0].content, "partial\n[generation stopped by user]");
    }

    #[test]
    fn consecutive_same_role_messages_merge_with_blank_line() {
        let events = vec![
            ev(1, "u1", "user_message", json!({"text": "first"})),
            ev(2, "u2", "user_message", json!({"text": "second"})),
            ev(3, "a1", "assistant_message", json!({"text": "reply"})),
        ];

        let turn = project_context(&events, Path::new(".")).unwrap();

        assert_eq!(turn.messages.len(), 2);
        assert_eq!(turn.messages[0].role, ProviderRole::User);
        assert_eq!(turn.messages[0].content, "first\n\nsecond");
        assert_eq!(turn.messages[1].role, ProviderRole::Assistant);
        assert_eq!(turn.messages[1].content, "reply");
    }

    #[test]
    fn handled_types_cover_every_durable_type() {
        // P-3: the compile-time-adjacent guard. `event_type` is a `String`
        // (I-2), so the match in `project_context` cannot be exhaustive in
        // the compiler's sense — this test stands in for that. Add a ninth
        // durable type to `DURABLE_TYPES` and this fails until `HANDLED_TYPES`
        // (and the match arms it mirrors) grow to cover it.
        assert_eq!(HANDLED_TYPES.len(), DURABLE_TYPES.len());
        for t in DURABLE_TYPES {
            assert!(
                HANDLED_TYPES.contains(t),
                "{t} is durable but has no explicit arm in project_context / HANDLED_TYPES"
            );
        }
    }

    #[test]
    fn totality_every_durable_type_projects_without_an_unhandled_marker() {
        let dir = tempfile::tempdir().unwrap();
        let boot_bytes: &[u8] = b"# boot content";
        std::fs::write(dir.path().join("boot.md"), boot_bytes).unwrap();
        let sha = sha256_hex(boot_bytes);

        let mut events = Vec::new();
        for (i, t) in DURABLE_TYPES.iter().enumerate() {
            let seq = (i + 1) as u64;
            let id = format!("id-{seq}");
            let data = match *t {
                "instance_created" => json!({}),
                "boot" => json!({"path": "boot.md", "sha256": sha}),
                "user_message" => json!({"text": "hello"}),
                "assistant_message" => json!({"text": "hi", "interrupted": false}),
                "interrupt" => json!({}),
                "context_evict" => json!({"target": "nonexistent-id"}),
                "dispatch" => json!({"operator": "tycho", "childStream": "child-1"}),
                "return" => json!({"child": "child-1", "summary": "done"}),
                other => panic!("test fixture missing a data payload for durable type {other}"),
            };
            events.push(ev(seq, &id, t, data));
        }

        let turn = project_context(&events, dir.path())
            .expect("every durable type must project without error");

        assert!(turn.system.as_deref().unwrap_or("").contains("boot content"));
        for m in &turn.messages {
            assert!(
                !m.content.contains("[unhandled"),
                "unexpected unhandled marker for a known durable type: {}",
                m.content
            );
        }
    }

    #[test]
    fn boot_sha_mismatch_is_boot_drift() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("boot.md"), b"actual content").unwrap();

        let events = vec![ev(
            1,
            "b1",
            "boot",
            json!({"path": "boot.md", "sha256": "0000000000000000000000000000000000000000000000000000000000000000"}),
        )];

        let err = project_context(&events, dir.path()).unwrap_err();

        assert!(matches!(err, ProjectionError::BootDrift { ref path } if path == "boot.md"));
    }

    #[test]
    fn boot_path_escaping_root_is_boot_unreadable() {
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.md"), b"nope").unwrap();
        let root = tempfile::tempdir().unwrap();

        let outside_name = outside.path().file_name().unwrap().to_string_lossy().to_string();
        let escape_path = format!("../{outside_name}/secret.md");

        let events = vec![ev(
            1,
            "b1",
            "boot",
            json!({"path": escape_path, "sha256": "irrelevant"}),
        )];

        let err = project_context(&events, root.path()).unwrap_err();

        assert!(matches!(err, ProjectionError::BootUnreadable { .. }));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "seq == 0")]
    fn transient_event_panics_via_debug_assert_in_debug_builds() {
        // I-4: a transient (seq 0) event is never persisted, so it should
        // never reach the reducer at all — this is a caller-contract
        // violation, loudly asserted in debug (see `ProjectionError::Transient`
        // for the release-mode half of this same guard).
        let events = vec![ev(0, "id0", "user_message", json!({"text": "hi"}))];
        let _ = project_context(&events, Path::new("."));
    }
}
