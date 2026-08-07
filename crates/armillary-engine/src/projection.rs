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
///
/// No `Eq`: `ContentBlock::ToolUse` carries a `serde_json::Value`, which is
/// `PartialEq` but not `Eq` (floats). `PartialEq` is all any caller or test
/// needs.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelTurn {
    pub system: Option<String>,
    pub messages: Vec<ProviderMessage>,
}

/// One flattened message. `content` has already had same-role neighbors
/// merged (see `merge_consecutive`) — this is the P-4 provider shape, not
/// the log's typed shape.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderMessage {
    pub role: ProviderRole,
    pub content: Vec<ContentBlock>,
}

/// A single content block, mirroring the Anthropic shapes this engine
/// can produce. Materialized to JSON by `provider::build_request_body` — this
/// type is deliberately wire-*shaped* but not wire-*encoded*, so the encoding
/// lives at one edge (P-4's separate flattening stage).
///
/// **`ToolResult` carries no `status` field, and must not grow one.** Measured
/// against the live API: a `status` key inside a `tool_result` block returns
/// 400 `invalid_request_error` — "Extra inputs are not permitted". The typed
/// status is sovereign in the *log*; what crosses to the model is `is_error`
/// plus whatever the projection rendered into `content`.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentBlock {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
    /// Opaque: captured from the stream, persisted, echoed verbatim. The
    /// engine never reads `thinking` or `signature` — the documented replay
    /// contract is "unchanged", and absence is the only alternative the API
    /// cannot distinguish from a turn that never thought.
    Thinking {
        thinking: String,
        signature: String,
    },
    /// Encrypted thinking the API returns pre-opaque. Same rule, no fields
    /// the engine can even pretend to read.
    RedactedThinking {
        data: String,
    },
}

impl ContentBlock {
    /// Convenience for the common single-text case.
    pub fn text(s: impl Into<String>) -> Self {
        ContentBlock::Text(s.into())
    }
}

/// One message whose content is exactly one text block — the shape every
/// pre-tool projection arm produces.
fn text_message(role: ProviderRole, text: impl Into<String>) -> ProviderMessage {
    ProviderMessage {
        role,
        content: vec![ContentBlock::Text(text.into())],
    }
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
    /// A declared boot file could not be read at all.
    ///
    /// **Defensive only since B-2, and a candidate for removal.** `read_boot`
    /// now judges each file by comparing its *current* verdict against the
    /// recorded one, and every way a file can fail to load — missing,
    /// permission-denied, not valid UTF-8, canonicalizing outside `root` —
    /// registers as a changed verdict, which is `BootDrift`. Drift heals
    /// forward; this kills the turn. So the cases that used to arrive here now
    /// arrive there, and a stream that would have been permanently dead
    /// repairs itself instead.
    ///
    /// What remains is one branch that should be unreachable: a UTF-8 decode
    /// failing after the same bytes already passed a UTF-8 check. Kept rather
    /// than `unwrap`ped. Removing the variant means giving
    /// `ProjectionError::Transient` its own machine code, since the two share
    /// an arm in `loop_::project_healing` — a small change, but it renames a
    /// user-visible string, so it is recorded here rather than slipped in.
    BootUnreadable { path: String },
    /// **DD-1.** The workspace manifests no longer hash to what the last
    /// standing `composition` event recorded — one was edited, created, or
    /// removed since it was written, so the session is describing a workspace
    /// that no longer exists.
    ///
    /// Carries no path, unlike `BootDrift`: the composition is derived from
    /// *all* the manifests together, so the recovery re-derives the whole
    /// thing rather than re-reading one file. Which manifest moved is a
    /// diagnostic, not something the repair needs.
    ///
    /// **Only the manifests are watched.** A protocol *source* (a board, an
    /// athanor) changes constantly without the composition changing at all,
    /// so its digest is neither recorded nor checked — see
    /// `tools::composition_event_data`. The cost is that a protocol source
    /// appearing or disappearing mid-session does not re-derive `present`
    /// until something else does.
    CompositionDrift,
    /// One or more `tool_use` events in this projection have no answering
    /// `tool_result`. The provider rejects that outright, so the caller must
    /// heal before it can take a turn.
    ///
    /// **Carries every unanswered id, not one.** The `BootDrift` precedent
    /// recovers a single failure per cycle, which is fine when there is only
    /// ever one boot event — but a batch can strand several calls at once (an
    /// engine killed after appending two `tool_use` events and no results), and
    /// a one-at-a-time recovery would heal one per round while the projection
    /// keeps failing.
    ///
    /// Heal-forward: the caller appends a real `tool_result` event per id and
    /// re-projects. It does not synthesize a block here — P-2 wants the
    /// injected content recorded first, I-1 says correction is a new event, and
    /// a repair invisible in the log is a repair nobody can audit.
    UnansweredToolUses { ids: Vec<String> },
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
    "composition",
    "user_message",
    "assistant_message",
    "interrupt",
    "context_evict",
    "dispatch",
    "return",
    "tool_use",
    "tool_result",
    "file_changed",
];

/// Resolve a boot event's `data.path` under `root`, through the same guard
/// every other path in this engine passes.
///
/// A `data.path` that escapes root is not a different kind of error, it is the
/// same `BootUnreadable` a missing file produces, since neither is ever a
/// legitimate boot source.
///
/// **This was containment WITHOUT judgement** — a bare canonicalize plus
/// `starts_with`, which `guard.rs`'s TRIPWIRE named as safe only while nothing
/// let a client choose or write a boot path. D-4 makes `modules.local.toml`
/// operator-writable, so a `boot` array could name a credential and the next
/// instance created would boot with it in its system prompt. `guard::resolve`
/// performs the same containment check and then judges every component of the
/// canonical path, so that declaration is now refused.
///
/// `is_openable` is deliberately NOT added on top: `weigh_boot_file` already
/// refuses non-UTF-8, which covers the binary case without a second rule to
/// keep in sync. And `guard::resolve` additionally rejects absolute paths and
/// `..` components, which this function used to accept — `weigh_boot_file`
/// refuses absolutes itself before ever appending one, and `rerecord_boot`
/// records an unresolvable path absent, so neither caller changes behaviour
/// beyond refusing more.
///
/// `pub(crate)` (not private): the loop (`loop_.rs`) re-records a fresh
/// `boot` event on `BootDrift` by re-reading the SAME path this function
/// resolved the first time — it reuses this resolution rather than
/// re-deriving root-containment logic at a second call site.
pub(crate) fn resolve_boot_path(root: &Path, rel: &str) -> Result<PathBuf, ProjectionError> {
    crate::guard::resolve(root, rel).map_err(|_| ProjectionError::BootUnreadable {
        path: rel.to_string(),
    })
}

/// Merge consecutive same-role messages with a blank line. Anthropic
/// requires strict user/assistant alternation; a raw flattening can produce
/// same-role neighbors (e.g. two `user_message` events back to back, or a
/// `dispatch` narrative line — itself `User`-role — right after a real user
/// message). Whether the *first* message is `User` is left alone HERE: real
/// logs always open on a user message, so this merge pass has no opinion on
/// it — that is `ensure_opens_on_user`'s job, run once over this function's
/// output (see its doc for why message 0 is not always `User` once eviction
/// is in play).
fn merge_consecutive(raw: Vec<ProviderMessage>) -> Vec<ProviderMessage> {
    let mut merged: Vec<ProviderMessage> = Vec::with_capacity(raw.len());
    for msg in raw {
        match merged.last_mut() {
            Some(last) if last.role == msg.role => {
                for block in msg.content {
                    append_block(&mut last.content, block);
                }
            }
            _ => merged.push(msg),
        }
    }
    merged
}

/// Append a block, folding text into text.
///
/// Two adjacent `Text` blocks are legal on the wire but wasteful: folding them
/// preserves the single-block shortcut that keeps text-only turns byte-identical
/// to the pre-tool build, which is the only reason the request goldens still
/// hold after this migration. The `"\n\n"` is the separator the string version
/// of this function used, kept so the merged text is byte-identical too.
///
/// **Ordering caveat for whoever adds tool events.** Anthropic requires
/// `tool_result` blocks to form an uninterrupted *leading* sequence in the user
/// turn that answers a `tool_use` (measured: text before a result, or a result
/// after intervening text, is a 400). This function appends in log order and
/// therefore does NOT enforce that. It is correct today only because no arm
/// below produces a `ToolResult`; the ordering rule belongs to the branch that
/// adds them, not here.
fn append_block(blocks: &mut Vec<ContentBlock>, block: ContentBlock) {
    match (blocks.last_mut(), &block) {
        (Some(ContentBlock::Text(prev)), ContentBlock::Text(next)) => {
            prev.push_str("\n\n");
            prev.push_str(next);
        }
        _ => blocks.push(block),
    }
}

/// The evict seam: Anthropic requires the message list to open on `User` and
/// to alternate strictly. A real log always opens on a user message, so
/// ordinarily nothing here needs to act (`merge_consecutive`'s doc leaves
/// "is message 0 `User`?" alone for exactly that reason). But `context_evict`
/// (P-1) can legitimately remove the log's first `user_message` from THIS
/// projection — evicting the earliest turn is not a special case, it's the
/// same mechanism as evicting any other event — and when it does, whatever
/// was originally message 1 (an `Assistant` message, or an assistant-shaped
/// narrative line) becomes message 0. Handed to the provider as-is, that is
/// an `Assistant`-opening turn, which the API rejects outright — every
/// subsequent turn on that stream would then fail `provider_api_400` with no
/// way for the client to recover, since nothing about *this* turn caused it.
///
/// The fix is to prepend an honest placeholder `User` message rather than to
/// special-case eviction in pass two above: it keeps alternation valid, and
/// it makes the eviction visible to the model instead of silently editing
/// history out from under it.
fn ensure_opens_on_user(mut messages: Vec<ProviderMessage>) -> Vec<ProviderMessage> {
    let opens_on_assistant = matches!(
        messages.first(),
        Some(ProviderMessage { role: ProviderRole::Assistant, .. })
    );
    if opens_on_assistant {
        messages.insert(
            0,
            text_message(
                ProviderRole::User,
                "[earlier user message removed from context]",
            ),
        );
    }
    messages
}

/// What a model should try instead, per guard/route machine code.
///
/// D6′'s "render the recovery action, not just the code" is not a pedagogic
/// preference — it is load-bearing twice. A bare `denied_credential` tells the
/// model nothing about what to do next, and (measured) `is_error: true` with
/// **empty** content is a 400, so the error path must render *something* or it
/// bricks the stream.
fn recovery_hint(status: &str) -> &'static str {
    match status {
        "denied_credential" => "this path holds credential material and is never served; read something else",
        "denied_noise" => "this path is build output or the engine's own data; read something else",
        "outside_workspace" => "paths are relative and must stay inside the workspace root",
        "not_found" => "nothing is at that path; list the parent directory first",
        "not_openable" => "that file type is not served as text; try .md, .爻, .toml, .json or a source file",
        "too_large" => "the file exceeds the byte ceiling; re-read a page of it",
        "not_text" => "the file is not valid UTF-8 and cannot be served as text",
        "is_a_directory" => "that is a directory; list it instead of reading it",
        "not_a_directory" => "that is a file; read it instead of listing it",
        "malformed_path" => "the path is not usable as written",
        "unknown_tool" => "no tool by that name exists; use one of the tools declared in this request",
        "invalid_input" => "the arguments did not match the tool's schema; check the names and types and call it again",
        "read_failed" => "the file could not be read from disk; try again or read something else",
        "composition_unreadable" => "the workspace manifests could not be parsed; read them as files instead",
        // The write verbs. A status with no arm here renders "the call did not
        // succeed", which tells the model nothing to do next.
        "composition_locked" => "this file defines the workspace's composition and this session was not granted permission to write it; change something else, or ask for a session that may compose",
        "no_match" => "old_string does not appear in that file; re-read the file and copy the exact text, without read_file's line-number prefixes",
        "ambiguous_match" => "old_string appears more than once; include more surrounding lines until exactly one match remains, or call edit_file once per occurrence",
        "write_failed" => "the file could not be written to disk; check the path and try again, or write somewhere else",
        // The three ways a call ends without its tool ever running. Each says
        // whether repeating it is worth anything — `interrupted` and
        // `no_result_recorded` are retryable, `bound_reached` is not.
        "interrupted" => "the turn was stopped before this call ran; ask again if you still need it",
        "no_result_recorded" => "this call was never answered — the engine restarted mid-turn; call it again if you still need it",
        "bound_reached" => "this turn hit its tool-call limit; answer with what you have rather than calling again",
        "tool_panicked" => "the tool crashed on this input; try different arguments or a different tool",
        // An unknown code still renders non-empty, which is what the wire needs.
        _ => "the call did not succeed",
    }
}

/// Render a tool result's model-visible content.
///
/// On success the tool's own output crosses unchanged (an empty successful
/// result is legal). On failure the machine code is preserved **verbatim** —
/// so a human reading the transcript sees the same string the log holds — and
/// is followed by the recovery hint, guaranteeing non-empty content.
fn render_tool_result(status: &str, body: &str, is_error: bool) -> String {
    if !is_error {
        return body.to_string();
    }
    let hint = recovery_hint(status);
    if body.is_empty() {
        format!("{status}: {hint}")
    } else {
        format!("{status}: {hint}\n{body}")
    }
}

/// One entry of a boot event's file list, as recorded.
struct BootFile {
    path: String,
    /// Absent when the file was not on disk at record time (`present: false`).
    sha256: Option<String>,
}

/// The recorded boot file list, accepting both shapes.
///
/// **Both, deliberately.** Every stream written before B-2 carries a single
/// `{path, sha256}`, and a log that cannot project its own history is not an
/// append-only log. The single-file form reads as a one-element list.
fn boot_files(data: &serde_json::Value) -> Vec<BootFile> {
    if let Some(files) = data.get("files").and_then(|f| f.as_array()) {
        return files
            .iter()
            .map(|f| BootFile {
                path: text_field(f, "path"),
                sha256: match f.get("present").and_then(|p| p.as_bool()) {
                    Some(false) => None,
                    _ => Some(text_field(f, "sha256")),
                },
            })
            .collect();
    }
    match data.get("path").and_then(|p| p.as_str()) {
        Some(path) => vec![BootFile {
            path: path.to_string(),
            sha256: Some(text_field(data, "sha256")),
        }],
        None => Vec::new(),
    }
}

/// **B-2.** Read a boot event's declared files and join them into one system
/// prompt, in the order the manifest declared.
///
/// Files are joined with a blank line and **nothing else** — no path banners,
/// no engine framing. This is identity prose, and interleaving machine markers
/// through someone's principles changes how they read. The composition already
/// tells a session where its operator lives if it wants to cite a source.
///
/// Returns `None` when nothing loaded, never `Some("")`: an empty system
/// prompt is a shape the request builder must not emit, and "nothing loaded"
/// is the same state as "nothing declared".
fn read_boot(root: &Path, data: &serde_json::Value) -> Result<Option<String>, ProjectionError> {
    let mut parts: Vec<String> = Vec::new();

    for file in boot_files(data) {
        let unreadable = || ProjectionError::BootUnreadable {
            path: file.path.clone(),
        };
        let drifted = || ProjectionError::BootDrift {
            path: file.path.clone(),
        };

        // What the file is RIGHT NOW, judged by exactly the rule the writer
        // used: resolvable under root, readable, and valid UTF-8. Anything
        // else is "not loadable", which is a state, not an error.
        let current: Option<Vec<u8>> = resolve_boot_path(root, &file.path)
            .ok()
            .and_then(|resolved| std::fs::read(&resolved).ok())
            .filter(|bytes| std::str::from_utf8(bytes).is_ok());

        // **Drift is a changed verdict, not a changed file.** Comparing the
        // recorded state against the current one handles all four cases with
        // one rule — and getting this wrong is what an earlier draft did: it
        // treated "recorded absent" as "must not exist", so a boot file that
        // existed but was refused for being non-UTF-8 drifted on every
        // projection, was repaired into a present entry, and then failed the
        // turn on the decode. A file that BECOMES loadable is real drift; a
        // file that was never loadable and still isn't has not changed.
        match (&file.sha256, current) {
            (Some(expected), Some(bytes)) if sha256_hex(&bytes) == *expected => {
                parts.push(String::from_utf8(bytes).map_err(|_| unreadable())?);
            }
            (Some(_), Some(_)) => return Err(drifted()),
            // Recorded present, no longer loadable — the file was deleted or
            // broken under the session. Drift, so the repair re-records it
            // absent and the stream keeps working.
            (Some(_), None) => return Err(drifted()),
            // Recorded absent and now loadable: an identity that GAINS a file
            // changed as surely as one that edits a file.
            (None, Some(_)) => return Err(drifted()),
            (None, None) => continue,
        }
    }

    Ok(if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    })
}

/// True when the manifests on disk no longer match what this `composition`
/// event recorded — edited, created, or removed.
///
/// Compares the recorded `(path, sha256)` pairs against the current ones as
/// sets, so all three changes are one comparison. A manifest the engine could
/// not read is absent from both sides, which is what "not composed" means.
fn manifests_drifted(root: &Path, data: &serde_json::Value) -> bool {
    let recorded: Vec<(String, String)> = data
        .get("manifests")
        .and_then(|m| m.as_array())
        .map(|entries| {
            entries
                .iter()
                .map(|e| (text_field(e, "path"), text_field(e, "sha256")))
                .collect()
        })
        .unwrap_or_default();

    let current: Vec<(String, String)> = crate::tools::MANIFEST_FILES
        .iter()
        .filter_map(|name| {
            std::fs::read(root.join(name))
                .ok()
                .map(|bytes| (name.to_string(), sha256_hex(&bytes)))
        })
        .collect();

    recorded != current
}

/// The model-visible composition.
///
/// **The framing line is load-bearing.** `boot` gets the system slot and there
/// is only one, so this rides in a User-role turn — and a User turn the user
/// did not write is exactly the confusion D3 created by forging a tool result.
/// Saying so in the content is the honest version: the model can tell who
/// authored it, which is as far toward the boundary as the channel carries
/// provenance (P-4).
fn render_composition(data: &serde_json::Value) -> String {
    let body = data.get("composition").unwrap_or(data);
    let rendered =
        serde_json::to_string_pretty(body).unwrap_or_else(|_| "unavailable".to_string());
    format!(
        "[engine] This workspace is composed of the following, derived by the engine from its \
         manifest files. This is a record the engine placed here, not a message from the user.\n\
         {rendered}"
    )
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

    // D14 — the eviction unit is the assistant turn plus its complete tool
    // batch, never a lone event.
    //
    // Measured both ways: a `tool_use` with no `tool_result` is a 400, and a
    // `tool_result` with no `tool_use` is a 400. So removing half a pair from a
    // projection kills every subsequent turn on the stream. The evict route
    // takes ONE event id and does not check its type, which makes that one
    // HTTP call away.
    //
    // Membership comes from `parent` — declared on the envelope since v0.1 and
    // constructed nowhere until now. A tool event's `parent` names the
    // assistant event that owns its batch, so membership is a filter. The
    // alternative, walking positionally from the assistant event, breaks the
    // moment anything else in the batch has already been evicted.
    fn batch_root(e: &EventEnvelope) -> Option<&str> {
        match e.event_type.as_str() {
            "tool_use" | "tool_result" => e.parent.as_deref(),
            _ => None,
        }
    }

    // Which batches lost a member? Evicting a tool event condemns its siblings;
    // evicting the owning assistant event condemns the whole batch under it.
    let mut condemned: HashSet<&str> = HashSet::new();
    for e in events {
        if !evicted.contains(e.id.as_str()) {
            continue;
        }
        match batch_root(e) {
            Some(root) => {
                condemned.insert(root);
            }
            None => {
                condemned.insert(e.id.as_str());
            }
        }
    }
    for e in events {
        if let Some(root) = batch_root(e) {
            if condemned.contains(root) {
                evicted.insert(e.id.as_str());
            }
        }
    }

    // Every surviving call must have a surviving answer. Eviction above is
    // batch-atomic so it cannot produce an orphan; what does is a crash between
    // the two appends, an interrupt, or a bound firing mid-batch.
    let answered: HashSet<&str> = events
        .iter()
        .filter(|e| e.event_type == "tool_result" && !evicted.contains(e.id.as_str()))
        .filter_map(|e| e.data.get("toolUseId").and_then(|v| v.as_str()))
        .collect();
    let unanswered: Vec<String> = events
        .iter()
        .filter(|e| e.event_type == "tool_use" && !evicted.contains(e.id.as_str()))
        .filter_map(|e| e.data.get("id").and_then(|v| v.as_str()))
        .filter(|id| !answered.contains(id))
        .map(str::to_string)
        .collect();
    if !unanswered.is_empty() {
        return Err(ProjectionError::UnansweredToolUses { ids: unanswered });
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

    // DD-1 inherits the same rule, for the same reason: the drift recovery
    // re-records rather than editing history, so validating every historical
    // composition would fail on the very event the repair just superseded.
    let last_composition_id: Option<&str> = events
        .iter()
        .rfind(|ev| ev.event_type == "composition" && !evicted.contains(ev.id.as_str()))
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
                system = read_boot(root, &ev.data)?;
            }

            // Superseded by a later composition — see `last_composition_id`.
            // Contributes nothing, and is NOT drift-checked.
            "composition" if Some(ev.id.as_str()) != last_composition_id => {}

            "composition" => {
                if manifests_drifted(root, &ev.data) {
                    return Err(ProjectionError::CompositionDrift);
                }
                raw.push(text_message(ProviderRole::User, render_composition(&ev.data)));
            }

            "user_message" => {
                raw.push(text_message(
                    ProviderRole::User,
                    text_field(&ev.data, "text"),
                ));
            }

            "assistant_message" => {
                let mut content = text_field(&ev.data, "text");
                if ev.data.get("interrupted").and_then(|v| v.as_bool()).unwrap_or(false) {
                    // The `interrupt` event itself carries no separate
                    // arm (below) — the flag on *this* message is the
                    // carrier, so the marker belongs here, not there.
                    content.push_str("\n[generation stopped by user]");
                }
                // An assistant turn that only called tools has no text, and an
                // empty text block is a 400 (measured). Legal alternative,
                // also measured: an assistant message carrying only `tool_use`
                // blocks. So contribute nothing here and let the `tool_use`
                // arm stand the message up — rather than synthesizing prose
                // the model never produced, which would put words in the
                // operator's mouth in a durable record.
                if !content.is_empty() {
                    raw.push(text_message(ProviderRole::Assistant, content));
                }
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
                raw.push(text_message(
                    ProviderRole::User,
                    format!("[dispatched {target} — child stream {child_stream}]"),
                ));
            }

            "return" => {
                let child = text_field(&ev.data, "child");
                let content = match ev.data.get("summary").and_then(|v| v.as_str()) {
                    Some(summary) => format!("[child {child} returned: {summary}]"),
                    None => format!("[child {child} returned]"),
                };
                raw.push(text_message(ProviderRole::User, content));
            }

            // Assistant-role, so `merge_consecutive` folds it into the
            // `assistant_message` that preceded it — one assistant turn
            // carrying its text and its calls, which is the shape the API
            // requires. No new assembly logic; the merge pass does it.
            "tool_use" => {
                raw.push(ProviderMessage {
                    role: ProviderRole::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: text_field(&ev.data, "id"),
                        name: text_field(&ev.data, "name"),
                        input: ev
                            .data
                            .get("input")
                            .cloned()
                            .unwrap_or_else(|| serde_json::Value::Object(Default::default())),
                    }],
                });
            }

            // User-role: the wire has no tool role, and `tool_result` blocks
            // ride in the user turn that answers the call.
            //
            // `status` is read here and NOT forwarded — the block type has no
            // field for it and the API rejects one. It survives to the model
            // only through `is_error` and the rendered content, which is P-4's
            // "as far toward the boundary as the channel allows". The typed
            // value stays in `data`, where eviction, the client and loop
            // control read it.
            "tool_result" => {
                let is_error = ev
                    .data
                    .get("isError")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                raw.push(ProviderMessage {
                    role: ProviderRole::User,
                    content: vec![ContentBlock::ToolResult {
                        tool_use_id: text_field(&ev.data, "toolUseId"),
                        content: render_tool_result(
                            &text_field(&ev.data, "status"),
                            &text_field(&ev.data, "content"),
                            is_error,
                        ),
                        is_error,
                    }],
                });
            }

            // WD-8: durable, deliberately NOT projected. The model already has
            // this from `tool_result`; rendering it again duplicates
            // information and spends tokens. Handled-by-skipping is still
            // handled — P-3 wants a case for every durable type, not a message
            // for every durable type. Without this arm the catch-all below
            // renders `[unhandled event type: file_changed]` into the model's
            // context, which is the opposite of the decision.
            "file_changed" => {}

            // Never silent (P-3): visible in the transcript rather than
            // dropped, so a gap in coverage shows up where a human or the
            // exhaustiveness test can see it, instead of vanishing.
            other => {
                raw.push(text_message(
                    ProviderRole::User,
                    format!("[unhandled event type: {other}]"),
                ));
            }
        }
    }

    Ok(ModelTurn {
        system,
        messages: ensure_opens_on_user(merge_consecutive(raw)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::envelope::{Actor, Role};
    use serde_json::json;

    /// Assert a message is exactly one text block, and return it.
    ///
    /// Deliberately panics on a multi-block message rather than concatenating.
    /// Every assertion below was written when `content` was a `String`; folding
    /// blocks together silently would let this migration change a message's
    /// shape without a single test noticing, which is the whole failure the
    /// request goldens exist to prevent one layer up.
    fn text_of(m: &ProviderMessage) -> &str {
        match m.content.as_slice() {
            [ContentBlock::Text(t)] => t,
            other => panic!("expected exactly one text block, got {other:?}"),
        }
    }

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
        assert_eq!(text_of(&turn.messages[0]), "hi");
        assert_eq!(turn.messages[0].role, ProviderRole::User);
        assert_eq!(text_of(&turn.messages[1]), "hello");
        assert_eq!(turn.messages[1].role, ProviderRole::Assistant);
        assert_eq!(text_of(&turn.messages[2]), "bye");
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

    // ---- B-2: an operator's boot is several declared files, in order ----

    /// Write `files` into `dir` and return the `files` array a boot event
    /// would carry for them, in the same order.
    fn boot_files(dir: &Path, files: &[(&str, &str)]) -> serde_json::Value {
        json!(files
            .iter()
            .map(|(name, body)| {
                std::fs::write(dir.join(name), body).unwrap();
                json!({ "path": name, "sha256": sha256_hex(body.as_bytes()), "present": true })
            })
            .collect::<Vec<_>>())
    }

    #[test]
    fn a_boot_event_listing_several_files_concatenates_them_in_declared_order() {
        // Order is the declaration, and it is load-bearing twice: it is what
        // the operator meant, and it is the prefix-cache order (stable first,
        // so an edit late in the list invalidates less).
        let dir = tempfile::tempdir().unwrap();
        let files = boot_files(
            dir.path(),
            &[("principles.md", "# principles"), ("voice.md", "# voice")],
        );

        let events = vec![
            ev(1, "b1", "boot", json!({ "files": files })),
            ev(2, "u1", "user_message", json!({"text": "hi"})),
        ];

        let turn = project_context(&events, dir.path()).unwrap();
        assert_eq!(turn.system.as_deref(), Some("# principles\n\n# voice"));
    }

    #[test]
    fn a_single_file_boot_event_still_projects_exactly_as_before() {
        // Every stream written before this change carries `{path, sha256}`.
        // Refusing to read them back would make the log's own history
        // unprojectable, which is the one thing an append-only log may not do.
        let dir = tempfile::tempdir().unwrap();
        let bytes: &[u8] = b"# the old shape";
        std::fs::write(dir.path().join("boot.md"), bytes).unwrap();

        let events = vec![ev(
            1,
            "b1",
            "boot",
            json!({"path": "boot.md", "sha256": sha256_hex(bytes)}),
        )];

        let turn = project_context(&events, dir.path()).unwrap();
        assert_eq!(turn.system.as_deref(), Some("# the old shape"));
    }

    #[test]
    fn one_moved_file_in_a_boot_list_is_drift_and_the_error_names_it() {
        let dir = tempfile::tempdir().unwrap();
        let files = boot_files(
            dir.path(),
            &[("principles.md", "# principles"), ("voice.md", "# voice")],
        );
        std::fs::write(dir.path().join("voice.md"), "# voice, rewritten").unwrap();

        let events = vec![ev(1, "b1", "boot", json!({ "files": files }))];

        match project_context(&events, dir.path()) {
            Err(ProjectionError::BootDrift { path }) => assert_eq!(path, "voice.md"),
            other => panic!("expected drift naming voice.md, got {other:?}"),
        }
    }

    #[test]
    fn a_declared_boot_file_that_was_absent_is_recorded_absent_and_stays_skipped() {
        // C-4: a declared path with nothing behind it is skipped, not an error
        // — the same posture `protocol_sources.present` takes in the
        // composition. Recording the absence is what keeps it visible; a boot
        // that silently loads two of three files is the failure this avoids.
        let dir = tempfile::tempdir().unwrap();
        let mut files = boot_files(dir.path(), &[("principles.md", "# principles")])
            .as_array()
            .unwrap()
            .clone();
        files.push(json!({ "path": "never-written.md", "present": false }));

        let events = vec![ev(1, "b1", "boot", json!({ "files": files }))];

        let turn = project_context(&events, dir.path()).unwrap();
        assert_eq!(turn.system.as_deref(), Some("# principles"));
    }

    #[test]
    fn a_file_that_appeared_where_one_was_recorded_absent_is_drift() {
        // The other direction, and it matters: an identity that GAINS a file
        // has changed as surely as one that edits a file, and a session going
        // on without it is describing an operator that no longer exists.
        let dir = tempfile::tempdir().unwrap();
        let events = vec![ev(
            1,
            "b1",
            "boot",
            json!({ "files": [{ "path": "voice.md", "present": false }] }),
        )];
        std::fs::write(dir.path().join("voice.md"), "# voice, written since").unwrap();

        assert!(matches!(
            project_context(&events, dir.path()),
            Err(ProjectionError::BootDrift { .. })
        ));
    }

    #[test]
    fn a_boot_event_with_no_readable_files_sets_no_system_prompt() {
        // Not an empty string: an empty system prompt is a shape the request
        // builder must not emit, and "nothing loaded" is the same state as
        // "nothing declared".
        let dir = tempfile::tempdir().unwrap();
        let events = vec![ev(1, "b1", "boot", json!({ "files": [] }))];

        let turn = project_context(&events, dir.path()).unwrap();
        assert_eq!(turn.system, None);
    }

    // ---- DD-1: the composition reaches a session as its own durable event ----

    /// A workspace whose manifests are on disk, plus the event that records
    /// them. `extra` lets a test change what the event claims.
    fn workspace_with_manifests() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("modules.toml"), "[router]\ncontains = [\"CLAUDE.md\"]\n")
            .unwrap();
        dir
    }

    fn composition_event(seq: u64, id: &str, root: &Path) -> EventEnvelope {
        ev(
            seq,
            id,
            "composition",
            crate::tools::composition_event_data(root).unwrap(),
        )
    }

    #[test]
    fn the_composition_projects_as_a_user_turn_the_engine_marks_as_its_own() {
        // P-2: recorded before projected. It rides in a User-role turn because
        // the system slot belongs to `boot` and there is only one — so the
        // framing has to say, in the content, that the user did not write this.
        let dir = workspace_with_manifests();
        let events = vec![
            composition_event(1, "c1", dir.path()),
            ev(2, "u1", "user_message", json!({"text": "what is here?"})),
        ];

        let turn = project_context(&events, dir.path()).unwrap();
        let first = &turn.messages[0];

        assert_eq!(first.role, ProviderRole::User);
        let text = match &first.content[0] {
            ContentBlock::Text(t) => t.clone(),
            other => panic!("expected text, got {other:?}"),
        };
        assert!(
            text.contains("[engine]"),
            "the model must be able to tell this from a user message: {text}"
        );
        assert!(text.contains("CLAUDE.md"), "the composition itself: {text}");
    }

    #[test]
    fn a_manifest_edited_under_a_recorded_composition_is_drift() {
        // The half D3 silently abandoned. A manifest is exactly the thing that
        // changes mid-session — you edit `modules.local.toml` and the session
        // is now describing a workspace that no longer exists.
        let dir = workspace_with_manifests();
        let events = vec![composition_event(1, "c1", dir.path())];

        std::fs::write(dir.path().join("modules.toml"), "[router]\ncontains = [\"README.md\"]\n")
            .unwrap();

        assert!(matches!(
            project_context(&events, dir.path()),
            Err(ProjectionError::CompositionDrift)
        ));
    }

    #[test]
    fn a_manifest_that_appeared_or_vanished_is_drift_too() {
        // Composition is what the manifests *collectively* declare, so a new
        // `modules.local.toml` changes it as surely as an edit does. Comparing
        // recorded digests to current ones catches all three the same way.
        let dir = workspace_with_manifests();
        let events = vec![composition_event(1, "c1", dir.path())];

        std::fs::write(dir.path().join("modules.local.toml"), "[[repos]]\nname='r'\npath='p'\n")
            .unwrap();
        assert!(
            matches!(project_context(&events, dir.path()), Err(ProjectionError::CompositionDrift)),
            "a manifest that appeared"
        );

        let with_both = vec![composition_event(1, "c1", dir.path())];
        std::fs::remove_file(dir.path().join("modules.local.toml")).unwrap();
        assert!(
            matches!(project_context(&with_both, dir.path()), Err(ProjectionError::CompositionDrift)),
            "a manifest that vanished"
        );
    }

    #[test]
    fn a_stale_drifted_composition_superseded_by_a_later_one_does_not_error() {
        // The same rule `boot` already lives by, and for the same reason:
        // recovery re-records rather than editing history, so the drifted
        // event stays in the log forever. Validating every historical one
        // would make the repair impossible — it would fail on the very event
        // it just superseded.
        let dir = workspace_with_manifests();
        let mut stale = composition_event(1, "c1", dir.path());
        stale.data["manifests"][0]["sha256"] = json!("0".repeat(64));

        let events = vec![
            stale,
            ev(2, "u1", "user_message", json!({"text": "hi"})),
            composition_event(3, "c2", dir.path()),
        ];

        let turn = project_context(&events, dir.path())
            .expect("the superseded c1 must not fail this projection");

        // And only ONE composition turn survives — the stale one contributes
        // nothing, exactly like a superseded boot.
        let engine_turns = turn
            .messages
            .iter()
            .filter(|m| {
                m.content.iter().any(|b| matches!(b, ContentBlock::Text(t) if t.contains("[engine]")))
            })
            .count();
        assert_eq!(engine_turns, 1, "{:?}", turn.messages);
    }

    #[test]
    fn an_evicted_composition_contributes_nothing_and_is_not_checked() {
        // P-1: evicted, not deleted. An evicted composition must not drift the
        // stream either — it is absent from this projection entirely.
        let dir = workspace_with_manifests();
        let mut stale = composition_event(1, "c1", dir.path());
        stale.data["manifests"][0]["sha256"] = json!("0".repeat(64));

        let events = vec![
            stale,
            ev(2, "e1", "context_evict", json!({"target": "c1"})),
            ev(3, "u1", "user_message", json!({"text": "hi"})),
        ];

        let turn = project_context(&events, dir.path()).expect("an evicted event is not checked");
        assert!(!turn.messages.iter().any(|m| m
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Text(t) if t.contains("[engine]")))));
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
            text_of(&turn.messages[0]),
            "[dispatched worker-7 — child stream child-stream-1]"
        );
    }

    #[test]
    fn evicting_the_first_user_message_prepends_an_honest_placeholder_and_stays_valid() {
        // The evict-seam finding: without the fix, evicting the log's only
        // user_message leaves the projection opening on Assistant, which
        // the Anthropic API rejects — every future turn on this stream
        // would then fail provider_api_400 with no way out.
        let dir = tempfile::tempdir().unwrap();
        let bytes: &[u8] = b"# system prompt";
        std::fs::write(dir.path().join("boot.md"), bytes).unwrap();
        let sha = sha256_hex(bytes);

        let events = vec![
            ev(1, "b1", "boot", json!({"path": "boot.md", "sha256": sha})),
            ev(2, "u1", "user_message", json!({"text": "hi"})),
            ev(3, "a1", "assistant_message", json!({"text": "hello"})),
            ev(4, "e1", "context_evict", json!({"target": "u1"})),
        ];

        let turn = project_context(&events, dir.path()).unwrap();

        assert_eq!(turn.messages.len(), 2);
        assert_eq!(turn.messages[0].role, ProviderRole::User);
        assert_eq!(text_of(&turn.messages[0]), "[earlier user message removed from context]");
        assert_eq!(turn.messages[1].role, ProviderRole::Assistant);
        assert_eq!(text_of(&turn.messages[1]), "hello");
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
        assert_eq!(text_of(&turn.messages[0]), "stays");
    }

    #[test]
    fn interrupted_assistant_message_gets_trailing_marker() {
        // A real log always opens on user_message (a bare assistant_message
        // with nothing before it only arises via eviction, covered by
        // `evicting_the_first_user_message_prepends_an_honest_placeholder_and_stays_valid`
        // above) — included here so this fixture isn't the artificial,
        // seam-triggering shape the interrupted-marker behavior itself has
        // nothing to do with.
        let events = vec![
            ev(1, "u1", "user_message", json!({"text": "go"})),
            ev(
                2,
                "a1",
                "assistant_message",
                json!({"text": "partial", "interrupted": true}),
            ),
        ];

        let turn = project_context(&events, Path::new(".")).unwrap();

        assert_eq!(text_of(&turn.messages[1]), "partial\n[generation stopped by user]");
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
        assert_eq!(text_of(&turn.messages[0]), "first\n\nsecond");
        assert_eq!(turn.messages[1].role, ProviderRole::Assistant);
        assert_eq!(text_of(&turn.messages[1]), "reply");
    }

    #[test]
    fn a_tool_round_projects_as_one_assistant_turn_and_one_answering_user_turn() {
        // The shape the API requires, assembled by the existing merge pass
        // rather than by new assembly logic: `tool_use` projects Assistant-role
        // so it folds into the assistant message that preceded it, and
        // `tool_result` projects User-role so it opens the answering turn.
        let events = vec![
            ev(1, "u1", "user_message", json!({"text": "what is composed?"})),
            ev(
                2,
                "a1",
                "assistant_message",
                json!({"text": "let me look", "interrupted": false}),
            ),
            ev(
                3,
                "tu1",
                "tool_use",
                json!({"id": "toolu_01AAA", "name": "get_composition", "input": {}}),
            ),
            ev(
                4,
                "tr1",
                "tool_result",
                json!({
                    "toolUseId": "toolu_01AAA",
                    "status": "ok",
                    "content": "4 operators, 17 repos",
                    "isError": false,
                }),
            ),
        ];

        let turn = project_context(&events, Path::new(".")).unwrap();

        assert_eq!(turn.messages.len(), 3, "{:#?}", turn.messages);
        assert_eq!(turn.messages[1].role, ProviderRole::Assistant);
        assert_eq!(
            turn.messages[1].content,
            vec![
                ContentBlock::Text("let me look".to_string()),
                ContentBlock::ToolUse {
                    id: "toolu_01AAA".to_string(),
                    name: "get_composition".to_string(),
                    input: json!({}),
                },
            ]
        );
        assert_eq!(turn.messages[2].role, ProviderRole::User);
        assert_eq!(
            turn.messages[2].content,
            vec![ContentBlock::ToolResult {
                tool_use_id: "toolu_01AAA".to_string(),
                content: "4 operators, 17 repos".to_string(),
                is_error: false,
            }]
        );
    }

    #[test]
    fn the_typed_status_stays_in_the_log_and_renders_into_the_result_content() {
        // S-1's split, made concrete. `status` is sovereign in `data` — eviction,
        // the client and loop control all read it — but the wire has no slot for
        // it (measured: a `status` key in a tool_result block is a 400). What
        // crosses is `is_error` plus a rendered prefix naming the refusal AND the
        // recovery, because `is_error` with empty content is itself a 400.
        let events = vec![
            ev(1, "u1", "user_message", json!({"text": "read the env file"})),
            ev(
                2,
                "tu1",
                "tool_use",
                json!({"id": "t1", "name": "read_file", "input": {"path": "repos/app/.env"}}),
            ),
            ev(
                3,
                "tr1",
                "tool_result",
                json!({
                    "toolUseId": "t1",
                    "status": "denied_credential",
                    "content": "",
                    "isError": true,
                }),
            ),
        ];

        let turn = project_context(&events, Path::new(".")).unwrap();
        let last = turn.messages.last().unwrap();

        match &last.content[0] {
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                assert!(*is_error);
                assert!(
                    content.contains("denied_credential"),
                    "the machine code must survive verbatim: {content}"
                );
                assert!(
                    !content.is_empty(),
                    "is_error with empty content is a 400 — the render must never produce one"
                );
            }
            other => panic!("expected a tool_result block, got {other:?}"),
        }
    }

    #[test]
    fn every_status_the_engine_can_emit_names_its_own_recovery() {
        // The sibling of `handled_types_cover_every_durable_type`, for the
        // status vocabulary. The fallback keeps an unlisted code non-empty —
        // which is what the wire needs — but a code that renders as "the call
        // did not succeed" is a dead end: the model learns that something
        // failed and nothing about what to do instead.
        //
        // Hand-maintained against the emitters, which are `tools.rs`
        // (`ToolError::new` plus everything `guard::GuardError::code` returns)
        // and `loop_.rs` (`append_tool_result` / `answer_all`). Grep for
        // `status:` and `answer_all(` when adding one.
        let emitted = [
            // guard
            "malformed_path", "outside_workspace", "not_found",
            "denied_credential", "denied_noise",
            // tools
            "unknown_tool", "invalid_input", "composition_unreadable",
            "is_a_directory", "not_a_directory", "not_openable", "not_text",
            "too_large", "read_failed",
            // loop
            "interrupted", "no_result_recorded", "bound_reached", "tool_panicked",
            // write
            "composition_locked", "no_match", "ambiguous_match", "write_failed",
        ];
        let fallback = recovery_hint("__a_status_no_one_emits__");

        for status in emitted {
            assert_ne!(
                recovery_hint(status),
                fallback,
                "`{status}` renders as the fallback, so it tells the model nothing to do"
            );
        }
    }

    /// A tool event carrying `parent` — the assistant event that owns the batch.
    fn child(seq: u64, id: &str, parent: &str, t: &str, data: serde_json::Value) -> EventEnvelope {
        let mut e = ev(seq, id, t, data);
        e.parent = Some(parent.to_string());
        e
    }

    fn a_batch_of_two() -> Vec<EventEnvelope> {
        vec![
            ev(1, "u1", "user_message", json!({"text": "look at two things"})),
            ev(2, "a1", "assistant_message", json!({"text": "on it"})),
            child(3, "tu1", "a1", "tool_use", json!({"id": "t1", "name": "get_composition", "input": {}})),
            child(4, "tu2", "a1", "tool_use", json!({"id": "t2", "name": "get_composition", "input": {}})),
            child(5, "tr1", "a1", "tool_result", json!({"toolUseId": "t1", "status": "ok", "content": "one", "isError": false})),
            child(6, "tr2", "a1", "tool_result", json!({"toolUseId": "t2", "status": "ok", "content": "two", "isError": false})),
        ]
    }

    #[test]
    fn evicting_one_tool_use_evicts_the_whole_batch_including_its_results() {
        // Measured: a tool_use with no result is a 400, and so is a result with
        // no use. So a batch is the eviction unit — the route takes one event
        // id, and every member has to go with it or the stream is dead.
        let mut events = a_batch_of_two();
        events.push(ev(7, "e1", "context_evict", json!({"target": "tu1"})));

        let turn = project_context(&events, Path::new(".")).unwrap();

        for m in &turn.messages {
            for block in &m.content {
                assert!(
                    !matches!(block, ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. }),
                    "a batch member survived eviction: {block:?}"
                );
            }
        }
    }

    #[test]
    fn evicting_one_tool_result_also_takes_the_call_it_answered() {
        // The other direction. D14 as first written only covered results whose
        // use was absent; evicting the RESULT strands the use, which fails just
        // as hard.
        let mut events = a_batch_of_two();
        events.push(ev(7, "e1", "context_evict", json!({"target": "tr2"})));

        let turn = project_context(&events, Path::new(".")).unwrap();

        for m in &turn.messages {
            for block in &m.content {
                assert!(
                    !matches!(block, ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. }),
                    "a batch member survived eviction: {block:?}"
                );
            }
        }
    }

    #[test]
    fn evicting_the_assistant_event_evicts_the_batch_it_owns() {
        let mut events = a_batch_of_two();
        events.push(ev(7, "e1", "context_evict", json!({"target": "a1"})));

        let turn = project_context(&events, Path::new(".")).unwrap();

        assert_eq!(
            turn.messages.len(),
            1,
            "only the user message should remain: {:#?}",
            turn.messages
        );
    }

    #[test]
    fn every_unanswered_call_in_a_batch_is_reported_not_just_the_first() {
        // The BootDrift precedent handles one failure per recovery cycle. A
        // batch can strand several at once — an engine killed after appending
        // two calls and no results — so the error has to carry all of them or
        // the loop heals one per round forever.
        let events = vec![
            ev(1, "u1", "user_message", json!({"text": "look at two things"})),
            ev(2, "a1", "assistant_message", json!({"text": "on it"})),
            child(3, "tu1", "a1", "tool_use", json!({"id": "t1", "name": "get_composition", "input": {}})),
            child(4, "tu2", "a1", "tool_use", json!({"id": "t2", "name": "get_composition", "input": {}})),
        ];

        let err = project_context(&events, Path::new(".")).unwrap_err();

        assert_eq!(
            err,
            ProjectionError::UnansweredToolUses {
                ids: vec!["t1".to_string(), "t2".to_string()]
            }
        );
    }

    #[test]
    fn an_answered_batch_does_not_report_orphans() {
        assert!(project_context(&a_batch_of_two(), Path::new(".")).is_ok());
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
                // No manifests on disk in this fixture, so an empty recorded
                // list is the non-drifted state.
                "composition" => json!({"manifests": [], "composition": {}}),
                "user_message" => json!({"text": "hello"}),
                "assistant_message" => json!({"text": "hi", "interrupted": false}),
                "interrupt" => json!({}),
                "context_evict" => json!({"target": "nonexistent-id"}),
                "dispatch" => json!({"operator": "tycho", "childStream": "child-1"}),
                "return" => json!({"child": "child-1", "summary": "done"}),
                "tool_use" => json!({"id": "toolu_fixture", "name": "get_composition", "input": {}}),
                "tool_result" => json!({
                    "toolUseId": "toolu_fixture",
                    "status": "ok",
                    "content": "fixture result",
                    "isError": false,
                }),
                // WD-8: this one is durable and deliberately projects NOTHING.
                // It still belongs in the fixture — the assertion below is that
                // no durable type reaches the catch-all, and "contributes no
                // message" and "contributes an [unhandled] marker" are exactly
                // what this test tells apart.
                "file_changed" => json!({
                    "path": "notes/example.md",
                    "op": "modified",
                    "before": "aa",
                    "after": "bb",
                }),
                other => panic!("test fixture missing a data payload for durable type {other}"),
            };
            events.push(ev(seq, &id, t, data));
        }

        let turn = project_context(&events, dir.path())
            .expect("every durable type must project without error");

        assert!(turn.system.as_deref().unwrap_or("").contains("boot content"));
        for m in &turn.messages {
            // Scans text blocks rather than requiring one — a durable type may
            // now legitimately project as a tool block, and the marker this
            // test hunts for is only ever emitted as text.
            for block in &m.content {
                if let ContentBlock::Text(t) = block {
                    assert!(
                        !t.contains("[unhandled"),
                        "unexpected unhandled marker for a known durable type: {t}"
                    );
                }
            }
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
    fn file_changed_is_durable_and_has_an_explicit_reducer_arm() {
        // MUTATION-CHECKED. P-3: the reducer obligation is enforced rather
        // than remembered — adding a durable type fails the suite until a case
        // exists for it.
        assert!(DURABLE_TYPES.contains(&"file_changed"));
        assert!(HANDLED_TYPES.contains(&"file_changed"));
    }

    #[test]
    fn a_file_changed_event_contributes_nothing_to_the_projection() {
        // MUTATION-CHECKED, and it asserts what the totality count CANNOT see.
        // `project_context`'s match ends in a catch-all that renders
        // `[unhandled event type: …]`, so "no arm" is not "not projected" — it
        // is projected as noise. WD-8: the model already learned what happened
        // from `tool_result`; rendering the event again duplicates information
        // and spends tokens. P-1 makes the window a projection, so a durable
        // event is not obliged to enter it.
        let events = vec![
            ev(1, "u1", "user_message", json!({"text": "write it"})),
            ev(
                2,
                "fc1",
                "file_changed",
                json!({
                    "path": "operators/tycho/todo.md",
                    "op": "modified",
                    "before": "aa", "after": "bb"
                }),
            ),
        ];

        let turn = project_context(&events, Path::new(".")).unwrap();
        let rendered = format!("{:?}", turn.messages);

        assert!(!rendered.contains("file_changed"), "{rendered}");
        assert!(!rendered.contains("unhandled event type"), "{rendered}");
        assert!(!rendered.contains("todo.md"), "{rendered}");
    }

    #[test]
    fn a_boot_path_naming_a_credential_is_refused() {
        // MUTATION-CHECKED. guard.rs's TRIPWIRE, closed. `resolve_boot_path`
        // read `data.path` with a bare root-containment check, bypassing
        // judge / is_credential / is_noise entirely. That was safe only while
        // nothing let a client choose a boot path — and D-4 makes the manifest
        // operator-writable, so an operator could declare
        // `boot = ["repos/x/Secrets.xcconfig"]` and the next instance created
        // would boot with a credential in its system prompt, past the one gate
        // built to prevent exactly that.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".env"), "TOKEN=hunter2\n").unwrap();
        std::fs::write(dir.path().join("Secrets.xcconfig"), "KEY=live\n").unwrap();
        std::fs::create_dir_all(dir.path().join("node_modules/pkg")).unwrap();
        std::fs::write(dir.path().join("node_modules/pkg/boot.md"), "# no\n").unwrap();
        std::fs::write(dir.path().join("real-boot.md"), "# yes\n").unwrap();

        for denied in [".env", "Secrets.xcconfig", "node_modules/pkg/boot.md"] {
            assert!(
                resolve_boot_path(dir.path(), denied).is_err(),
                "{denied} must never be readable as a boot file"
            );
        }

        // The other half, and it is what makes this test non-vacuous:
        // asserting only that things are refused would pass identically if the
        // function returned Err for everything.
        assert!(
            resolve_boot_path(dir.path(), "real-boot.md").is_ok(),
            "an ordinary boot file must still resolve"
        );
    }

    #[test]
    fn boot_path_escaping_root_never_loads_and_heals_instead_of_killing_the_stream() {
        // **Changed behaviour, recorded rather than slipped in.** This used to
        // be `BootUnreadable`, which fails the turn — so a stream carrying an
        // escaping boot path was permanently dead, every turn, forever. Under
        // B-2's verdict comparison it is drift instead: the repair re-records
        // the path absent, the projection skips it, and the session continues
        // with no system prompt.
        //
        // The file is not loaded either way, so nothing is loosened. What
        // changes is that a misconfiguration stops being fatal, which is the
        // posture this engine already takes everywhere else — an instance that
        // works without identity beats one that cannot run. The absence is
        // recorded in the log rather than only shouted at the turn, so it is
        // still visible to the client and greppable.
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

        assert!(
            matches!(err, ProjectionError::BootDrift { .. }),
            "escaping paths heal forward now: {err:?}"
        );
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
