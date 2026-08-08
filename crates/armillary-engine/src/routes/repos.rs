//! `/repos` — the four READ routes that make a composed repo addressable,
//! plus the three verbs that mutate one (`fetch`/`pull`/`push`) and the one
//! that mutates a whole group (`POST /repos/fetch`).
//!
//! **`repos::resolve` is the security boundary, and every handler here calls
//! it FIRST.** A manifest name is a KEY into `declared_modules`, never a path
//! fragment — see that function's own doc comment — so a miss is a 404
//! before any path is joined, canonicalized, or stat'd. There is no fallback
//! path construction anywhere below: a name that does not resolve simply has
//! nothing built from it.
//!
//! The four GET routes perform no fetch, no merge, and no mutation of any
//! kind. `GET /repos` deliberately calls `repos::read_one` with
//! `with_commit: false` (D5): the second fork it gates multiplies by every
//! composed repo, and NOT paying that multiplication on the list route is
//! this whole design's cost argument. `GET /repos/{name}` pays it once,
//! where one extra fork is unmeasurable.
//!
//! The four POST routes below are the only ones on this branch that change
//! anything. Each is now gated `registry ∧ manifest` (2026-08-07): the
//! `Caller` extractor authenticates first (401 with no mention of grants at
//! all), `crate::auth::require` checks the DEVICE's own grant from the
//! registry (403, naming `enroll`), THEN the existing shape resumes — resolve
//! the name (404 on a miss — an unknown repo is not this workspace's business
//! to refuse or permit), check the one MANIFEST gate that verb requires (403
//! naming the exact table and key, worded identically across every verb this
//! module gates), act, then `read_one(root, &module, true)` for the response
//! — **never** the state computed before acting. A git failure during the
//! act step does not 500; it lands in the response's `action_error` — a
//! typed `{kind, message}`, not a bare string, so a dirty-tree refusal and an
//! unreachable host are told apart without string-matching the message —
//! because twenty-three good answers and one error is the useful outcome
//! (repos.rs's `fetch_all` already makes this promise for the group verb;
//! the per-repo verbs make it too, reusing the same field rather than
//! inventing a second one `RepoState` would need to carry for no reader that
//! needs to tell them apart).

use crate::auth::Caller;
use crate::git::{self, GitError};
use crate::principals::Grant;
use crate::repos;
use crate::state::SharedState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};

/// Map a `GitError` to the response the brief for this feature describes for
/// the write verbs — `InvalidArg` is the caller's fault (400), the other two
/// are the read's own failure (502, since the upstream process — git — is
/// what did not answer, not this server). No route below can actually
/// produce `InvalidArg` today (every argument reaching `git::log` is a
/// manifest-derived path or a clamped integer, never request-shaped text),
/// but the mapping is total rather than partial so a future caller of
/// `git::log`/`git::status_v2` cannot introduce a silent 500 by omission.
fn git_error_response(e: GitError) -> (StatusCode, String) {
    match e {
        GitError::Timeout => (StatusCode::BAD_GATEWAY, "git timed out".to_string()),
        GitError::Failed(msg) => (StatusCode::BAD_GATEWAY, msg),
        GitError::InvalidArg(msg) => (StatusCode::BAD_REQUEST, msg),
    }
}

/// `git merge --ff-only`'s failure, folded to the typed error that lands in
/// `RepoState::action_error`.
///
/// Distinct from `git_error_response` above: that one is for a READ route,
/// where the failure of the read itself is the response. Here the git call
/// is the ACT step of a verb whose response is always 200 (or 403/404 from
/// the gate/resolve steps, which never reach this far) — the failure is data
/// riding inside an otherwise-successful `RepoState`, not a status code.
///
/// `pull_ff` touches no network — it is `git merge --ff-only @{u}` against
/// refs already on disk — so every non-timeout failure IS a fast-forward
/// refusal (a diverged branch, a missing upstream, anything else `--ff-only`
/// itself declines) rather than something reachability-shaped. That is a
/// structural fact about what `pull_ff` can fail at, not a guess: there is no
/// `"transport"` failure mode for a command that never dials out.
fn pull_action_error(e: GitError) -> repos::ActionError {
    match e {
        GitError::Timeout => repos::ActionError { kind: "timeout", message: "timed out".to_string() },
        GitError::Failed(msg) => repos::ActionError { kind: "not-fast-forwardable", message: msg },
        // Unreachable in practice: no argument reaching `git::pull_ff` is
        // request-derived. Carried like `Failed` rather than panicking.
        GitError::InvalidArg(msg) => repos::ActionError { kind: "not-fast-forwardable", message: msg },
    }
}

/// `git status --porcelain`'s own failure during `pull`'s pre-flight dirty
/// check — a different fact from a dirty-tree REFUSAL (that is
/// `"dirty"`, built at the call site) and from a fast-forward refusal (the
/// merge never even ran). Folded to `"transport"` as the closest fit in the
/// closed vocabulary: the status read itself did not answer, the same
/// judgment `git_error_response` makes of an identical `GitError` on a READ
/// route.
fn dirty_check_action_error(e: GitError) -> repos::ActionError {
    match e {
        GitError::Timeout => repos::ActionError { kind: "timeout", message: "timed out".to_string() },
        GitError::Failed(msg) => repos::ActionError { kind: "transport", message: msg },
        GitError::InvalidArg(msg) => repos::ActionError { kind: "transport", message: msg },
    }
}

/// `git push`'s failure, folded to the typed error that lands in
/// `RepoState::action_error`.
///
/// Unlike `pull_ff`, `push` genuinely talks to the network, so a non-timeout
/// failure has THREE real causes that do NOT share a shape:
///
/// - a non-fast-forward rejection (the remote has commits this push does
///   not) — git's own marker is `"! [rejected]"`, verified live 2026-08-04;
/// - a **policy refusal by the remote** (a protected branch, a pre-receive
///   hook that declines) — a DIFFERENT literal, `"! [remote rejected]"`,
///   verified live 2026-08-05 against a bare remote's `pre-receive` hook
///   exiting 1;
/// - a transport failure (the remote could not be reached at all) — neither
///   marker present.
///
/// The middle case is neither of the other two, and folding it into either
/// tells the user something that cannot work: it is not `"transport"` (the
/// remote was reached fine) and it is not `"not-fast-forwardable"` (pulling
/// first will not help — the remote declined on policy, not on history), so
/// it gets its own kind, `"refused-by-remote"`.
///
/// `"! [remote rejected]"` is checked FIRST, before `"! [rejected]"`. Checked
/// directly: `"! [remote rejected] main -> main (pre-receive hook
/// declined)"` does NOT contain `"[rejected]"` as a substring — `remote`
/// sits between the brackets and the word, so the two markers are already
/// mutually exclusive on git's current wording, and today's ordering does
/// not change which branch either message takes. The more specific marker is
/// still checked first, on the same reasoning `require_ok` states elsewhere
/// in this codebase for not inlining a shared block twice: matching the
/// broader pattern first is the version of this code that silently breaks
/// the day git's wording narrows the gap between the two, and nothing here
/// would announce that it broke.
///
/// Matching on these fixed, git-authored markers is not the caller-side
/// string-matching this design exists to end — that was about a CLIENT
/// having to parse prose to learn what kind of failure occurred; this is the
/// server doing the equivalent classification once, so the client never has
/// to.
fn push_action_error(e: GitError) -> repos::ActionError {
    match e {
        GitError::Timeout => repos::ActionError { kind: "timeout", message: "timed out".to_string() },
        GitError::Failed(msg) => {
            let kind = if msg.contains("[remote rejected]") {
                "refused-by-remote"
            } else if msg.contains("[rejected]") {
                "not-fast-forwardable"
            } else {
                "transport"
            };
            repos::ActionError { kind, message: msg }
        }
        // Unreachable in practice: no argument reaching `git::push` is
        // request-derived.
        GitError::InvalidArg(msg) => repos::ActionError { kind: "transport", message: msg },
    }
}

/// The MANIFEST half of `registry ∧ manifest`: the workspace's outer bound.
///
/// # Why a check that may never fire is still not decorative
///
/// This gate is expected to answer `true` on every machine for a long time —
/// David accepted a global `push` grant on 2026-08-07 ("in these early days
/// it makes it a lot easier on me"), so in practice the ceiling will not
/// deny. A check that never fires invites the daoUI precedent: the Figma
/// snapshot gate was DELETED on David's call because "a check whose name
/// claims more than it delivers is worse than an absent one, because it
/// reads as covered ground."
///
/// **That precedent does not apply here, and the difference is structural.**
/// The snapshot gate could not *possibly* see what its name claimed — it
/// compared `tokens.json` to a file generated from `tokens.json`, so no state
/// of the world would have made it fire. This gate is one edit away from
/// denying: remove `push = true` from `modules.local.toml`, commit, and every
/// host loses push on the next sync regardless of what any device registry
/// says. That single-edit, all-hosts revocation is the entire reason the
/// manifest keys were kept as a ceiling rather than retired into the registry
/// (design § 1.4). A fuse that has not blown is not a decorative check.
///
/// **The condition to revisit:** if a ceiling denial has never once been the
/// correct answer after a year of use, it is a fuse for a fire that does not
/// happen — and then deleting it is right.
///
/// One function shared by every verb this gate covers (fetch, fetch-one,
/// pull) — `Router.extra` is unvalidated by design (C-5), so a misspelled
/// `snyc` must read exactly like a deliberate off, and a second,
/// differently-worded refusal for the same gate would make that
/// indistinguishable from a genuine second gate.
fn sync_gate_denied() -> (StatusCode, String) {
    (
        StatusCode::FORBIDDEN,
        "this workspace has not granted the engine authority to run git. \
         Declare it by adding `sync = true` under `[router]` in \
         modules.local.toml (or modules.toml), then retry."
            .to_string(),
    )
}

/// 403 for a missing `[router] push = true` grant — independent from the
/// `sync` gate (D7 in `repos::push_enabled`'s doc comment): `sync` lets an
/// enrolled device make this host fetch and fast-forward; `push` additionally
/// lets it make this host publish, under the host user's own credential,
/// with no undo. Naming the table and key it looked for, same reasoning as
/// `sync_gate_denied`.
fn push_gate_denied() -> (StatusCode, String) {
    (
        StatusCode::FORBIDDEN,
        "this workspace has not granted the engine authority to push. \
         Declare it by adding `push = true` under `[router]` in \
         modules.local.toml (or modules.toml), then retry."
            .to_string(),
    )
}

#[derive(Serialize)]
pub struct ReposResponse {
    /// Whether this workspace has granted the engine authority to run git at
    /// all (`gate_enabled`).
    enabled: bool,
    /// Whether this workspace has additionally granted PUBLISH authority
    /// (`push_enabled`) — reported alongside `enabled`, not only on a
    /// single-repo read, so the client can hide the Push action on its own
    /// without a round trip per repo.
    push_enabled: bool,
    repos: Vec<repos::RepoState>,
    /// Git checkouts on disk that no manifest declares (see
    /// `repos::undeclared_checkouts`) — surfaced, never swept.
    not_composed: Vec<String>,
}

/// `GET /repos` — every composed repo's local state, one status fork per
/// repo, no network call.
///
/// `repos::list` does the actual per-repo work (`with_commit: false` — see
/// this module's own header; twenty-four repos times a second fork each is
/// the exact cost this design exists to avoid paying on the list read). This
/// handler enumerates `declared_modules` a second time only for
/// `not_composed`, which needs the SAME declared set `undeclared_checkouts`
/// diffs against but that `repos::list` does not itself return — a TOML
/// parse and two `read_dir`s, not a git subprocess, so paying it twice costs
/// nothing near what a second `status_v2` fork per repo would.
pub async fn list(State(state): State<SharedState>) -> Json<ReposResponse> {
    let root = state.root.clone();
    let snap = snapshot_or_default(&root);
    let declared = repos::declared_modules(&root, &snap.composition);
    let not_composed = repos::undeclared_checkouts(&root, &declared);

    Json(ReposResponse {
        enabled: repos::gate_enabled(&snap.composition),
        push_enabled: repos::push_enabled(&snap.composition),
        repos: repos::list(&root, &snap.composition).await,
        not_composed,
    })
}

/// The one manifest read a request performs. On an unloadable manifest the
/// default (nothing composed) reproduces each consumer's standing posture —
/// gates read false, module resolution 404s, the listing sweeps the root only
/// (design decision 5). Every answer a handler derives from the returned
/// snapshot agrees with every other, which is the point: gate-check and act
/// can no longer see different manifests within one request.
fn snapshot_or_default(root: &std::path::Path) -> crate::snapshot::WorkspaceSnapshot {
    crate::snapshot::WorkspaceSnapshot::load(root).unwrap_or_default()
}

/// `GET /repos/{name}` — one repo's full state, including `newest_commit`
/// (D5's `with_commit: true`), still no network call.
pub async fn one(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> Result<Json<repos::RepoState>, (StatusCode, String)> {
    let root = state.root.clone();
    let snap = snapshot_or_default(&root);
    let module = repos::resolve(&root, &snap.composition, &name)
        .ok_or((StatusCode::NOT_FOUND, "unknown_repo".to_string()))?;
    Ok(Json(repos::read_one(&root, &module, true).await))
}

/// `limit` is `Option<String>`, deliberately NOT `Option<u32>`. Axum's
/// derived `Query` extractor only defaults a field on a MISSING key — a
/// present-but-unparseable value (`?limit=abc`, `?limit=-1`) fails
/// deserialization and 400s before this handler's body ever runs, which is
/// the opposite of "clamp, never reject". Parsing has to happen past the
/// extractor, in `clamp_limit`, where a failure can fall back to the
/// default instead of refusing the request.
#[derive(Deserialize)]
pub struct LogQuery {
    limit: Option<String>,
}

/// Default page size for `GET /repos/{name}/log`, and its ceiling. Clamped,
/// never rejected — an oversized `limit` is a client mistake worth
/// tolerating, not a 400 worth returning, since the intent ("give me as much
/// as you reasonably can") is legible either way.
const DEFAULT_LOG_LIMIT: u32 = 50;
const MAX_LOG_LIMIT: u32 = 200;

/// The requested `limit` -> the number actually used.
///
/// Total, not partial: every input produces a `u32`, none produce an error.
/// A missing key (`None`) and an unparseable one (`Some("abc")`,
/// `Some("-1")` — `-1` cannot parse as `u32` at all, not merely "out of
/// range") both fall back to `DEFAULT_LOG_LIMIT` identically; anything that
/// DOES parse is capped at `MAX_LOG_LIMIT` rather than rejected. A pure
/// function on purpose — the earlier defect here (axum's `Option<u32>`
/// extractor 400ing on a malformed value before the handler ran) is exactly
/// the kind of thing that stays invisible behind a test asserting only a
/// status code on a fixture too small to need clamping; testing this
/// directly, argument in and `u32` out, is what actually discriminates it.
fn clamp_limit(raw: Option<&str>) -> u32 {
    raw.and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(DEFAULT_LOG_LIMIT)
        .min(MAX_LOG_LIMIT)
}

#[derive(Serialize)]
pub struct Commit {
    sha: String,
    subject: String,
    author: String,
    date: String,
    /// **Positional, not derived from which ref actually contains the
    /// commit.** The first `ahead` entries in `git log`'s (newest-first)
    /// order are marked unpushed — correct for a linear branch, and WRONG
    /// after a merge commit brings in commits that already exist on
    /// `@{u}`: those can sort ahead of a genuinely unpushed commit in log
    /// order, and this marking has no way to tell the two apart. Accepted
    /// for v1 (design's own limitation, restated here because this is where
    /// a future reader will meet it): the alternative is a second `git
    /// rev-list @{u}..HEAD` fork on every page load, on top of the one
    /// `git log` already pays and the one `git status --porcelain=v2
    /// --branch` `read_one` pays beneath it.
    unpushed: bool,
}

/// `GET /repos/{name}/log?limit=N` — the most recent `limit` commits
/// (default 50, capped at 200), each marked `unpushed` per the positional
/// rule documented on `Commit::unpushed`.
pub async fn log(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Query(q): Query<LogQuery>,
) -> Result<Json<Vec<Commit>>, (StatusCode, String)> {
    let root = state.root.clone();
    let snap = snapshot_or_default(&root);
    let module = repos::resolve(&root, &snap.composition, &name)
        .ok_or((StatusCode::NOT_FOUND, "unknown_repo".to_string()))?;
    let limit = clamp_limit(q.limit.as_deref());
    let abs = root.join(&module.path);

    let entries = git::log(&abs, limit, git::DEFAULT_TIMEOUT)
        .await
        .map_err(git_error_response)?;

    // `with_commit: false` — this route only needs `position.ahead`, which
    // `status_v2` (inside `read_one`) already carries; paying for the extra
    // `newest_commit` fork here would be strictly wasted, since `log`'s own
    // first entry already answers "what is the newest commit".
    let ahead = match repos::read_one(&root, &module, false).await.position {
        git::Position::Tracking { ahead, .. } => ahead,
        _ => 0,
    };

    let commits = entries
        .into_iter()
        .enumerate()
        .map(|(i, e)| Commit {
            sha: e.sha,
            subject: e.subject,
            author: e.author,
            date: e.date,
            unpushed: (i as u32) < ahead,
        })
        .collect();

    Ok(Json(commits))
}

/// `GET /repos/{name}/changes` — every changed path from ONE `git status
/// --porcelain=v2 --branch`, reusing `status_v2`/`parse_status_v2` rather
/// than forking a second `git status` just to list what `read_one` already
/// counted.
pub async fn changes(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> Result<Json<Vec<git::ChangedFile>>, (StatusCode, String)> {
    let root = state.root.clone();
    let snap = snapshot_or_default(&root);
    let module = repos::resolve(&root, &snap.composition, &name)
        .ok_or((StatusCode::NOT_FOUND, "unknown_repo".to_string()))?;
    let abs = root.join(&module.path);

    let status = git::status_v2(&abs, git::DEFAULT_TIMEOUT)
        .await
        .map_err(git_error_response)?;

    Ok(Json(status.files))
}

/// `POST /repos/{name}/fetch` — `git fetch --prune` on one repo, returning
/// its state AFTER the fetch. Gated on `sync`: fetch touches no working tree
/// and no branch, so it carries the same authority `POST /repos/fetch`
/// (the group sweep) already requires.
pub async fn fetch_one(
    State(state): State<SharedState>,
    caller: Caller,
    Path(name): Path<String>,
) -> Result<Json<repos::RepoState>, (StatusCode, String)> {
    crate::auth::require(&caller, Grant::Sync)?;
    let root = state.root.clone();
    let snap = snapshot_or_default(&root);
    let module = repos::resolve(&root, &snap.composition, &name)
        .ok_or((StatusCode::NOT_FOUND, "unknown_repo".to_string()))?;
    if !repos::gate_enabled(&snap.composition) {
        return Err(sync_gate_denied());
    }

    let abs = root.join(&module.path);
    let action_error =
        git::fetch(&abs, git::DEFAULT_TIMEOUT).await.err().map(repos::fetch_action_error);
    crate::repo_events::record_fetch(&state.sessions, &caller, &name, action_error.as_ref());

    // Read LAST, after the fetch landed (or failed to) — the caller must
    // never render a row it just acted on from state computed before acting.
    let mut new_state = repos::read_one(&root, &module, true).await;
    new_state.action_error = action_error;
    Ok(Json(new_state))
}

/// `POST /repos/{name}/pull` — `git merge --ff-only @{u}` on one repo,
/// returning its state AFTER the attempt. Gated on `sync`, same as fetch:
/// pull can only ever fast-forward (never merge, never `--force`), so it is
/// the same authority tier as fetch, not push's.
///
/// **A dirty working tree is refused before `git merge` ever runs**, not
/// left for git to catch. `git merge --ff-only` only refuses when the
/// incoming commits actually conflict with the uncommitted change; an
/// uncommitted edit to a file the remote never touched would sail through a
/// bare `--ff-only` call, silently carrying someone's in-progress edit
/// forward under a commit that never saw it. The explicit `is_dirty` check
/// closes that gap by refusing on ANY uncommitted change, not just a
/// conflicting one.
pub async fn pull(
    State(state): State<SharedState>,
    caller: Caller,
    Path(name): Path<String>,
) -> Result<Json<repos::RepoState>, (StatusCode, String)> {
    crate::auth::require(&caller, Grant::Sync)?;
    let root = state.root.clone();
    let snap = snapshot_or_default(&root);
    let module = repos::resolve(&root, &snap.composition, &name)
        .ok_or((StatusCode::NOT_FOUND, "unknown_repo".to_string()))?;
    if !repos::gate_enabled(&snap.composition) {
        return Err(sync_gate_denied());
    }

    let abs = root.join(&module.path);
    let Pulled { error: pull_error, before, after } = locked_dirty_check_then_pull(&abs).await;
    crate::repo_events::record_pull(
        &state.sessions,
        &caller,
        &name,
        before.as_deref(),
        after.as_deref(),
        pull_error.as_ref(),
    );

    let mut new_state = repos::read_one(&root, &module, true).await;
    new_state.action_error = pull_error;
    Ok(Json(new_state))
}

/// What one pull attempt did: its outcome and the shas it moved between.
///
/// `before` and `after` are `head_sha`, never `newest_commit` — the latter is
/// a committer DATE, and a date in a field every reader takes for a revision
/// is the "name promises more than the wire asserts" defect this build is
/// meant to stop shipping.
///
/// On a refusal `after == before`: the attempt is recorded and it moved
/// nothing, which is a different and more honest statement than no event.
struct Pulled {
    error: Option<repos::ActionError>,
    before: Option<String>,
    after: Option<String>,
}

/// The dirty check and the merge, under the process write lock.
///
/// **WD-15 extended over pull's check-then-merge.** `is_dirty` and
/// `git merge` are two subprocesses; without the lock, a `write_file` landing
/// between them defeats the refusal `pull`'s doc presents as the point — the
/// cross-session hazard the lock was built for, previously unextended here.
/// The guard spans only check+merge: `read_one` after it is a read and must
/// not serialize against writes.
///
/// **The HEAD reads are inside the lock too**, and that is not incidental
/// tidiness. `before` read outside it would be a sha another writer could
/// invalidate before the merge ran, so the event would name a transition that
/// never happened — the same "this machine's belief, reported as the remote's
/// account" error `--porcelain` exists to avoid, one layer down.
async fn locked_dirty_check_then_pull(abs: &std::path::Path) -> Pulled {
    let _write_guard = crate::write::write_lock_async().await;
    let before = git::head_sha(abs, git::DEFAULT_TIMEOUT).await.ok().flatten();
    let error = match git::is_dirty(abs, git::DEFAULT_TIMEOUT).await {
        Ok(true) => Some(repos::ActionError {
            kind: "dirty",
            message: "refusing to pull: the working tree has uncommitted changes".to_string(),
        }),
        Ok(false) => git::pull_ff(abs, git::DEFAULT_TIMEOUT).await.err().map(pull_action_error),
        Err(e) => Some(dirty_check_action_error(e)),
    };
    let after = git::head_sha(abs, git::DEFAULT_TIMEOUT).await.ok().flatten();
    Pulled { error, before, after }
}

/// `POST /repos/{name}/push` — `git push` on one repo, returning its state
/// AFTER the attempt. Gated on `push`, not `sync` (D7): publishing under the
/// host user's own credential, with no undo, is a strictly bigger authority
/// than letting a device make this host fetch or fast-forward.
pub async fn push(
    State(state): State<SharedState>,
    caller: Caller,
    Path(name): Path<String>,
) -> Result<Json<repos::RepoState>, (StatusCode, String)> {
    crate::auth::require(&caller, Grant::Push)?;
    let root = state.root.clone();
    let snap = snapshot_or_default(&root);
    let module = repos::resolve(&root, &snap.composition, &name)
        .ok_or((StatusCode::NOT_FOUND, "unknown_repo".to_string()))?;
    if !repos::push_enabled(&snap.composition) {
        return Err(push_gate_denied());
    }

    let abs = root.join(&module.path);
    let outcome = git::push(&abs, git::DEFAULT_TIMEOUT).await;
    let push_error = outcome.as_ref().err().cloned().map(push_action_error);

    // `commits` only where there is a range to count. A new branch and an
    // up-to-date push both report no shas, and a count derived from anything
    // else — the ahead-count `status_v2` holds, say — would be this machine's
    // belief before the push rather than what the push moved.
    let report = outcome.as_ref().ok();
    let commits = match report.and_then(|r| r.before.as_ref().zip(r.after.as_ref())) {
        Some((b, a)) => git::count_range(&abs, b, a, git::DEFAULT_TIMEOUT).await.ok(),
        None => None,
    };
    crate::repo_events::record_push(
        &state.sessions,
        &caller,
        &name,
        report,
        commits,
        &state.hostname,
        push_error.as_ref(),
    );

    let mut new_state = repos::read_one(&root, &module, true).await;
    new_state.action_error = push_error;
    Ok(Json(new_state))
}

/// `POST /repos/fetch` — fetch EVERY composed repo, bounded concurrency
/// (`repos::CONCURRENCY`). Gated on `sync`, same as the single-repo fetch —
/// a group fetch touches no working tree and no branch, so it carries no
/// more authority than fetching one repo at a time would. `repos::fetch_all`
/// is where "twenty-three good answers and one failure" is actually
/// implemented (each repo's own `action_error`); this handler is just the
/// gate in front of it.
pub async fn fetch_all(
    State(state): State<SharedState>,
    caller: Caller,
) -> Result<Json<Vec<repos::RepoState>>, (StatusCode, String)> {
    crate::auth::require(&caller, Grant::Sync)?;
    let root = state.root.clone();
    let snap = snapshot_or_default(&root);
    if !repos::gate_enabled(&snap.composition) {
        return Err(sync_gate_denied());
    }
    let states = repos::fetch_all(&root, &snap.composition).await;
    // One event per repo, not one for the sweep. A group fetch is twenty-five
    // fetches that each succeeded or failed on their own — the promise this
    // route already makes in its response ("twenty-three good answers and one
    // failure") has to hold in the record too, or the log flattens exactly the
    // distinction the response preserves.
    for s in &states {
        crate::repo_events::record_fetch(&state.sessions, &caller, &s.name, s.action_error.as_ref());
    }
    Ok(Json(states))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;

    /// BUILD RISK 1, discharged at the seam rather than at either end.
    ///
    /// `git.rs` proves the `[rejected]` marker survives `--porcelain`, and the
    /// classifier above is a pure function over that marker. Both can be green
    /// while the assembly is broken, which is the defect class this build kept
    /// finding — so this drives a REAL failing push of each classified kind
    /// through `push` into `push_action_error` and asserts the kind that comes
    /// out.
    ///
    /// The no-upstream case is included deliberately: it is the case the plan
    /// originally chose as its guard, and it is `transport` with or without the
    /// flag, so it could never have observed the regression it was written for.
    #[tokio::test]
    async fn each_rejection_keeps_its_kind_under_the_porcelain_flag() {
        use crate::git::{self, DEFAULT_TIMEOUT};
        use crate::testgit::{advance_remote, commit, git_sync, remote_and_clone};

        // 1 — a diverged branch: the remote moved, we did too.
        let (remote, clone) = remote_and_clone();
        advance_remote(&remote);
        commit(&clone, "mine.md", "local work");
        git::fetch(&clone, DEFAULT_TIMEOUT).await.unwrap();
        let err = push_action_error(git::push(&clone, DEFAULT_TIMEOUT).await.unwrap_err());
        assert_eq!(
            err.kind, "not-fast-forwardable",
            "a non-fast-forward must not degrade to transport; got {err:?}"
        );

        // 2 — a policy refusal: the remote was reached and declined.
        let (remote2, clone2) = remote_and_clone();
        let hooks = remote2.join("hooks");
        std::fs::create_dir_all(&hooks).unwrap();
        let hook = hooks.join("pre-receive");
        std::fs::write(&hook, "#!/bin/sh\nexit 1\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        commit(&clone2, "mine.md", "local work");
        let err = push_action_error(git::push(&clone2, DEFAULT_TIMEOUT).await.unwrap_err());
        assert_eq!(
            err.kind, "refused-by-remote",
            "a declining hook is not a transport failure; got {err:?}"
        );

        // 3 — no upstream: genuinely `transport`, and the reason the plan's
        // original guard was blind. Kept so that stays visible rather than
        // being rediscovered.
        let (_remote3, clone3) = remote_and_clone();
        git_sync(&clone3, &["checkout", "-b", "orphan"]);
        commit(&clone3, "mine.md", "local work");
        let err = push_action_error(git::push(&clone3, DEFAULT_TIMEOUT).await.unwrap_err());
        assert_eq!(err.kind, "transport", "got {err:?}");
    }

    #[test]
    fn a_mid_request_manifest_write_cannot_split_gate_from_act() {
        // The incoherence this whole task exists to kill: gate-check and act
        // reading different manifests within one request. The snapshot is
        // loaded, the manifest is then gutted on disk, and every answer
        // derived from the held snapshot must still agree with what was
        // loaded.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("repos/a/.git")).unwrap();
        std::fs::write(
            dir.path().join("modules.toml"),
            "[router]\nsync = true\n[[repos]]\nname = \"a\"\npath = \"repos/a\"\n",
        )
        .unwrap();

        let snap = snapshot_or_default(dir.path());
        std::fs::write(dir.path().join("modules.toml"), "").unwrap();

        assert!(crate::repos::gate_enabled(&snap.composition));
        assert!(crate::repos::resolve(dir.path(), &snap.composition, "a").is_some());
    }

    #[tokio::test]
    async fn the_pull_path_waits_on_a_held_write_lock() {
        // Discriminates the extension itself: with the guard removed from
        // locked_dirty_check_then_pull, the timeout branch completes and this
        // fails. 100ms is the safe direction — a false PASS would need the
        // whole check+merge to take longer than the timeout while unblocked,
        // and both subprocesses run against a local tempdir clone.
        let (_remote, clone) = crate::testgit::remote_and_clone();

        let guard = crate::write::write_lock_async().await;
        let blocked = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            locked_dirty_check_then_pull(&clone),
        )
        .await;
        assert!(blocked.is_err(), "pull must wait while the write lock is held");

        drop(guard);
        let outcome = locked_dirty_check_then_pull(&clone).await;
        assert!(
            outcome.error.is_none(),
            "an up-to-date clone pulls clean: {:?}",
            outcome.error
        );
        // And an up-to-date pull moved nothing, which the shas must say rather
        // than the event's absence saying it for them.
        assert_eq!(outcome.before, outcome.after, "nothing to fast-forward");
        assert!(outcome.before.is_some(), "a cloned repo has a HEAD");
    }

    #[test]
    fn clamp_limit_defaults_when_the_query_key_is_absent() {
        assert_eq!(clamp_limit(None), DEFAULT_LOG_LIMIT);
    }

    #[test]
    fn clamp_limit_accepts_an_explicit_zero() {
        assert_eq!(clamp_limit(Some("0")), 0);
    }

    #[test]
    fn clamp_limit_passes_an_ordinary_value_through() {
        assert_eq!(clamp_limit(Some("5")), 5);
    }

    #[test]
    fn clamp_limit_caps_an_oversized_value_rather_than_rejecting_it() {
        assert_eq!(clamp_limit(Some("99999")), MAX_LOG_LIMIT);
    }

    #[test]
    fn clamp_limit_falls_back_to_the_default_on_a_negative_value() {
        // "-1" cannot parse as a `u32` at all — this is the exact input
        // that made axum's derived `Option<u32>` extractor 400 before the
        // handler body ever ran, which `Option<String>` + `clamp_limit`
        // exists to fix.
        assert_eq!(clamp_limit(Some("-1")), DEFAULT_LOG_LIMIT);
    }

    #[test]
    fn clamp_limit_falls_back_to_the_default_on_garbage() {
        assert_eq!(clamp_limit(Some("abc")), DEFAULT_LOG_LIMIT);
    }

    // `registry ∧ manifest` at the three write verbs (2026-08-07). These
    // exercise the ASSEMBLED path through the real router — `crate::app` —
    // rather than calling handlers directly, so the extractor's ordering
    // (auth, then registry, then manifest) is what actually runs, the same
    // lesson Task 6's review drew about `Caller` itself: the pieces can be
    // tested while the assembly is not.

    /// A registry directory with one principal holding exactly `grants`,
    /// plus the token that authenticates as it.
    ///
    /// Returns the directory so the caller can put it on `AppState`. It does
    /// NOT touch `$HOME`: `std::env::set_var` mutates process-global state
    /// and Rust runs tests in parallel threads, so several tests doing it
    /// concurrently produce failures that appear only under load — passing
    /// alone, red in CI, and blaming the wrong test.
    fn enrolled(grants: Vec<crate::principals::Grant>) -> (tempfile::TempDir, String) {
        use crate::principals::{hash_token, mint_token, write_principal, Principal};
        let dir = tempfile::tempdir().unwrap();
        let token = mint_token();
        write_principal(
            dir.path(),
            &Principal {
                name: "iphone".to_string(),
                token_hash: hash_token(&token),
                grants,
                minted: "2026-08-07T00:00:00Z".to_string(),
            },
        )
        .unwrap();
        (dir, token)
    }

    /// A workspace whose only declared repo is "a" — a bare `.git` directory
    /// stand-in, mirroring `a_mid_request_manifest_write_cannot_split_gate_from_act`'s
    /// fixture above. These tests exercise the GATE (`registry ∧ manifest`),
    /// never git itself — `tests/repos.rs` is where a real clone earns its
    /// cost, for the verbs that actually need one to run to completion.
    fn workspace_with(modules_toml: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("repos/a/.git")).unwrap();
        std::fs::write(dir.path().join("modules.toml"), modules_toml).unwrap();
        dir
    }

    /// `POST /repos/a/push` against `ws`, authenticating against
    /// `registry_dir` with `token` (or unauthenticated when `None`), through
    /// the real axum router (`crate::app`) rather than the handler called
    /// directly — the extractor ordering is exactly what a live request
    /// hits, and calling the handler in-process would bypass it.
    async fn call_push(
        ws: &tempfile::TempDir,
        registry_dir: &std::path::Path,
        token: Option<&str>,
    ) -> (StatusCode, String) {
        let data_dir = tempfile::tempdir().unwrap();
        let store = crate::log::store::LogStore::open(data_dir.path()).unwrap();
        let state = crate::state::AppState {
            root: ws.path().to_path_buf(),
            sessions: std::sync::Arc::new(crate::sessions::Sessions::new(store)),
            model: crate::state::ModelConfig { model: "claude-sonnet-5".to_string() },
            providers: crate::provider::fixed(std::sync::Arc::new(crate::provider::KeylessProvider)),
            models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
            registry_dir: registry_dir.to_path_buf(),
            anthropic_key_present: false,
            zen_key_present: false,
            boot: None,
        };

        let mut req = axum::http::Request::builder().method("POST").uri("/repos/a/push");
        if let Some(t) = token {
            req = req.header(axum::http::header::AUTHORIZATION, format!("Bearer {t}"));
        }
        let response = crate::app(state)
            .oneshot(req.body(axum::body::Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 8 * 1024 * 1024).await.unwrap();
        (status, String::from_utf8_lossy(&bytes).to_string())
    }

    #[tokio::test]
    async fn a_push_without_a_token_is_401_before_the_manifest_is_consulted() {
        // The ordering claim, asserted rather than assumed: an
        // unauthenticated caller must learn NOTHING about the workspace's
        // grants, so the refusal must not mention the manifest.
        let (_home, _t) = enrolled(vec![]);
        let ws = workspace_with("[router]\npush = true\n[[repos]]\nname = \"a\"\npath = \"repos/a\"\n");
        let res = call_push(&ws, _home.path(), None).await;
        assert_eq!(res.0, StatusCode::UNAUTHORIZED);
        assert!(res.1.contains("no_principal"), "{}", res.1);
        assert!(!res.1.contains("modules.local.toml"), "must not leak the ceiling: {}", res.1);
    }

    #[tokio::test]
    async fn an_ungranted_device_is_403_and_does_not_learn_the_ceiling() {
        let (_home, token) = enrolled(vec![crate::principals::Grant::Sync]);
        let ws = workspace_with("[router]\npush = true\n[[repos]]\nname = \"a\"\npath = \"repos/a\"\n");
        let res = call_push(&ws, _home.path(), Some(&token)).await;
        assert_eq!(res.0, StatusCode::FORBIDDEN);
        assert!(res.1.contains("principal_not_granted"), "{}", res.1);
        assert!(!res.1.contains("modules.local.toml"), "{}", res.1);
    }

    #[tokio::test]
    async fn a_granted_device_still_hits_the_manifest_ceiling() {
        // The AND, in the direction that proves the ceiling is real: full
        // registry grant, manifest denies, request refused. Without this
        // test the ceiling could be deleted and every test would stay green.
        let (_home, token) = enrolled(vec![crate::principals::Grant::Push]);
        let ws = workspace_with("[router]\npush = false\n[[repos]]\nname = \"a\"\npath = \"repos/a\"\n");
        let res = call_push(&ws, _home.path(), Some(&token)).await;
        assert_eq!(res.0, StatusCode::FORBIDDEN);
        assert!(res.1.contains("modules.local.toml"), "the ceiling's own message: {}", res.1);
    }

    #[tokio::test]
    async fn a_full_host_grant_changes_nothing_the_manifest_allowed() {
        // `full ∧ manifest ≡ manifest` — the property that makes Stage 1 a
        // no-op for host behavior. With both grants held, the answer is
        // whatever the manifest alone would have said.
        let (_home, token) = enrolled(vec![
            crate::principals::Grant::Sync,
            crate::principals::Grant::Push,
        ]);
        for (manifest_push, expect_gate_refusal) in [("true", false), ("false", true)] {
            let ws = workspace_with(&format!(
                "[router]\npush = {manifest_push}\n[[repos]]\nname = \"a\"\npath = \"repos/a\"\n"
            ));
            let res = call_push(&ws, _home.path(), Some(&token)).await;
            assert_eq!(
                res.0 == StatusCode::FORBIDDEN && res.1.contains("modules.local.toml"),
                expect_gate_refusal,
                "manifest push = {manifest_push}"
            );
        }
    }
}
