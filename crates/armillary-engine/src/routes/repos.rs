//! `/repos` — the four READ routes that make a composed repo addressable.
//!
//! **`repos::resolve` is the security boundary, and every handler here calls
//! it FIRST.** A manifest name is a KEY into `declared_modules`, never a path
//! fragment — see that function's own doc comment — so a miss is a 404
//! before any path is joined, canonicalized, or stat'd. There is no fallback
//! path construction anywhere below: a name that does not resolve simply has
//! nothing built from it.
//!
//! These four routes perform no fetch, no merge, and no mutation of any
//! kind — the verbs (`POST /repos/{name}/pull`, `/push`, …) are a later
//! task. `GET /repos` deliberately calls `repos::read_one` with
//! `with_commit: false` (D5): the second fork it gates multiplies by every
//! composed repo, and NOT paying that multiplication on the list route is
//! this whole design's cost argument. `GET /repos/{name}` pays it once,
//! where one extra fork is unmeasurable.

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
