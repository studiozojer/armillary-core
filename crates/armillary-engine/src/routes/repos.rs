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
//! anything. Each follows the same shape: resolve the name (404 on a miss,
//! before the gate is even checked — an unknown repo is not this workspace's
//! business to refuse or permit), check the one gate that verb requires (403
//! naming the exact table and key, worded identically across every verb this
//! module gates), act, then `read_one(root, &module, true)` for the response
//! — **never** the state computed before acting. A git failure during the
//! act step does not 500; it lands in the response's `fetch_error`, because
//! twenty-three good answers and one error is the useful outcome (repos.rs's
//! `fetch_all` already makes this promise for the group verb; the per-repo
//! verbs make it too, reusing the same field rather than inventing a second
//! one RepoState would need to carry for no reader that needs to tell them
//! apart).

use crate::git::{self, GitError};
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

/// A `GitError` folded to the string that lands in `RepoState::fetch_error`.
///
/// Distinct from `git_error_response` above: that one is for a READ route,
/// where the failure of the read itself is the response. Here the git call
/// is the ACT step of a verb whose response is always 200 (or 403/404 from
/// the gate/resolve steps, which never reach this far) — the failure is data
/// riding inside an otherwise-successful `RepoState`, not a status code.
fn verb_error_message(e: GitError) -> String {
    match e {
        GitError::Timeout => "timed out".to_string(),
        GitError::Failed(msg) => msg,
        // Unreachable in practice, same as `repos::read_one`'s identical
        // fold: no argument reaching `git::fetch`/`pull_ff`/`push` is
        // request-derived. Carried like `Failed` rather than panicking.
        GitError::InvalidArg(msg) => msg,
    }
}

/// 403 for a missing `[router] sync = true` grant. One function shared by
/// every verb this gate covers (fetch, fetch-one, pull) — `Router.extra` is
/// unvalidated by design (C-5), so a misspelled `snyc` must read exactly
/// like a deliberate off, and a second, differently-worded refusal for the
/// same gate would make that indistinguishable from a genuine second gate.
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
pub async fn list(State(state): State<SharedState>) -> Json<ReposResponse> {
    let root = state.root.clone();
    let declared = repos::declared_modules(&root);
    let not_composed = repos::undeclared_checkouts(&root, &declared);

    let mut out = Vec::with_capacity(declared.len());
    for module in &declared {
        // `with_commit: false` — see this module's own header. Twenty-four
        // repos times a second fork each is the exact cost this design
        // exists to avoid paying on the list read.
        out.push(repos::read_one(&root, module, false).await);
    }

    Json(ReposResponse {
        enabled: repos::gate_enabled(&root),
        push_enabled: repos::push_enabled(&root),
        repos: out,
        not_composed,
    })
}

/// `GET /repos/{name}` — one repo's full state, including `newest_commit`
/// (D5's `with_commit: true`), still no network call.
pub async fn one(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> Result<Json<repos::RepoState>, (StatusCode, String)> {
    let root = state.root.clone();
    let module = repos::resolve(&root, &name)
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
    let module = repos::resolve(&root, &name)
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
    let module = repos::resolve(&root, &name)
        .ok_or((StatusCode::NOT_FOUND, "unknown_repo".to_string()))?;
    let abs = root.join(&module.path);

    let status = git::status_v2(&abs, git::DEFAULT_TIMEOUT)
        .await
        .map_err(git_error_response)?;

    Ok(Json(status.files))
}

/// `POST /repos/{name}/fetch` — `git fetch --prune` on one repo, returning
/// its state AFTER the fetch. Gated on `sync`: fetch touches no working tree
/// and no branch, so it carries the same authority `POST /sync` already
/// requires for the group sweep.
pub async fn fetch_one(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> Result<Json<repos::RepoState>, (StatusCode, String)> {
    let root = state.root.clone();
    let module = repos::resolve(&root, &name)
        .ok_or((StatusCode::NOT_FOUND, "unknown_repo".to_string()))?;
    if !repos::gate_enabled(&root) {
        return Err(sync_gate_denied());
    }

    let abs = root.join(&module.path);
    let fetch_error = git::fetch(&abs, git::DEFAULT_TIMEOUT)
        .await
        .err()
        .map(verb_error_message);

    // Read LAST, after the fetch landed (or failed to) — the caller must
    // never render a row it just acted on from state computed before acting.
    let mut new_state = repos::read_one(&root, &module, true).await;
    new_state.fetch_error = fetch_error;
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
/// conflicting one — dirty blocks a fast-forward outright, full stop, the
/// same precedence this workspace has always given it.
pub async fn pull(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> Result<Json<repos::RepoState>, (StatusCode, String)> {
    let root = state.root.clone();
    let module = repos::resolve(&root, &name)
        .ok_or((StatusCode::NOT_FOUND, "unknown_repo".to_string()))?;
    if !repos::gate_enabled(&root) {
        return Err(sync_gate_denied());
    }

    let abs = root.join(&module.path);
    let pull_error = match git::is_dirty(&abs, git::DEFAULT_TIMEOUT).await {
        Ok(true) => Some(
            "refusing to pull: the working tree has uncommitted changes".to_string(),
        ),
        Ok(false) => git::pull_ff(&abs, git::DEFAULT_TIMEOUT)
            .await
            .err()
            .map(verb_error_message),
        Err(e) => Some(verb_error_message(e)),
    };

    let mut new_state = repos::read_one(&root, &module, true).await;
    new_state.fetch_error = pull_error;
    Ok(Json(new_state))
}

/// `POST /repos/{name}/push` — `git push` on one repo, returning its state
/// AFTER the attempt. Gated on `push`, not `sync` (D7): publishing under the
/// host user's own credential, with no undo, is a strictly bigger authority
/// than letting a device make this host fetch or fast-forward.
pub async fn push(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> Result<Json<repos::RepoState>, (StatusCode, String)> {
    let root = state.root.clone();
    let module = repos::resolve(&root, &name)
        .ok_or((StatusCode::NOT_FOUND, "unknown_repo".to_string()))?;
    if !repos::push_enabled(&root) {
        return Err(push_gate_denied());
    }

    let abs = root.join(&module.path);
    let push_error = git::push(&abs, git::DEFAULT_TIMEOUT)
        .await
        .err()
        .map(verb_error_message);

    let mut new_state = repos::read_one(&root, &module, true).await;
    new_state.fetch_error = push_error;
    Ok(Json(new_state))
}

/// `POST /repos/fetch` — fetch EVERY composed repo, bounded concurrency
/// (`repos::CONCURRENCY`). Gated on `sync`, same as the single-repo fetch and
/// `POST /sync` — a group fetch touches no working tree and no branch, so it
/// carries no more authority than the sweep this whole feature set sits
/// beside. `repos::fetch_all` is where "twenty-three good answers and one
/// failure" is actually implemented (each repo's own `fetch_error`); this
/// handler is just the gate in front of it.
pub async fn fetch_all(
    State(state): State<SharedState>,
) -> Result<Json<Vec<repos::RepoState>>, (StatusCode, String)> {
    let root = state.root.clone();
    if !repos::gate_enabled(&root) {
        return Err(sync_gate_denied());
    }
    Ok(Json(repos::fetch_all(&root).await))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
