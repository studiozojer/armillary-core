//! The three mutating git verbs, once — the locked act steps and the failure
//! classifiers that turn a `GitError` into the typed `ActionError` a caller
//! records.
//!
//! **Hoisted out of `routes/repos.rs` (2026-08-09), unchanged.** They were
//! private to the route module while a route was the only caller; a model tool
//! is now the second one, and two callers of one verb must reach ONE
//! implementation or the lock discipline, the refusal vocabulary, and the
//! before/after sha honesty each get a second copy that drifts. The routes
//! above are thin callers of what lives here, and the tool bodies below are
//! the other.
//!
//! Not folded into `repos.rs`: that module is the repo SET and each repo's
//! read state — `declared_modules`, `read_one`, the gates — and it performs no
//! mutation at all by its own header's claim. The act steps are a different
//! concern and are already the larger half of what `routes/repos.rs` held.

use crate::git::{self, GitError};
use crate::repos;

/// `git merge --ff-only`'s failure, folded to the typed error that lands in
/// `RepoState::action_error`.
///
/// Distinct from `routes::repos::git_error_response`: that one is for a READ
/// route, where the failure of the read itself is the response. Here the git
/// call is the ACT step of a verb whose response is always 200 (or 403/404
/// from the gate/resolve steps, which never reach this far) — the failure is
/// data riding inside an otherwise-successful `RepoState`, not a status code.
///
/// `pull_ff` touches no network — it is `git merge --ff-only @{u}` against
/// refs already on disk — so every non-timeout failure IS a fast-forward
/// refusal (a diverged branch, a missing upstream, anything else `--ff-only`
/// itself declines) rather than something reachability-shaped. That is a
/// structural fact about what `pull_ff` can fail at, not a guess: there is no
/// `"transport"` failure mode for a command that never dials out.
pub(crate) fn pull_action_error(e: GitError) -> repos::ActionError {
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
/// judgment `routes::repos::git_error_response` makes of an identical
/// `GitError` on a READ route.
pub(crate) fn dirty_check_action_error(e: GitError) -> repos::ActionError {
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
pub(crate) fn push_action_error(e: GitError) -> repos::ActionError {
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

/// Unlike push, git gives a failed commit no stable locale-pinned marker the
/// way `[rejected]` marks a push — a pre-commit hook's decline is whatever
/// the hook printed. So one kind carries the message, and the *typed*
/// refusals (`detached`, `nothing-to-commit`) are engine-side pre-checks that
/// never reach git at all.
pub(crate) fn commit_action_error(e: GitError) -> repos::ActionError {
    match e {
        GitError::Timeout => repos::ActionError { kind: "timeout", message: "timed out".to_string() },
        GitError::Failed(msg) | GitError::InvalidArg(msg) => {
            repos::ActionError { kind: "commit-failed", message: msg }
        }
    }
}

/// The device's name, appended by the engine, LAST — so a client-supplied
/// `Committed-from:` line is superseded rather than trusted (last trailer
/// wins under `git interpret-trailers`, and ours is written after anything
/// the message carried).
pub(crate) fn with_trailer(message: &str, principal: &str) -> String {
    format!("{}\n\nCommitted-from: {}\n", message.trim_end(), principal)
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
pub(crate) struct Pulled {
    pub(crate) error: Option<repos::ActionError>,
    pub(crate) before: Option<String>,
    pub(crate) after: Option<String>,
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
pub(crate) async fn locked_dirty_check_then_pull(abs: &std::path::Path) -> Pulled {
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

/// What one commit attempt did. On a refusal `after == before` — recorded,
/// moved nothing (Pulled's contract, same reasons).
pub(crate) struct Committed {
    pub(crate) error: Option<repos::ActionError>,
    pub(crate) before: Option<String>,
    pub(crate) after: Option<String>,
    pub(crate) files: Option<u32>,
}

/// Status check, stage, and commit under the process write lock — pull's
/// argument, one verb over: without the lock a `write_file` landing between
/// the status read and `git add --all` would be swept into a commit whose
/// message never described it. One `status_v2` fork answers detached, clean,
/// the file count, and `before` at once.
pub(crate) async fn locked_status_then_commit(abs: &std::path::Path, full_message: &str) -> Committed {
    let _write_guard = crate::write::write_lock_async().await;
    let status = match git::status_v2(abs, git::DEFAULT_TIMEOUT).await {
        Ok(s) => s,
        Err(e) => {
            return Committed { error: Some(dirty_check_action_error(e)), before: None, after: None, files: None }
        }
    };
    let before = status.head.clone();
    if status.branch.is_none() {
        return Committed {
            error: Some(repos::ActionError {
                kind: "detached",
                message: "refusing to commit: HEAD is detached — a commit here would belong to no branch".to_string(),
            }),
            before: before.clone(),
            after: before,
            files: None,
        };
    }
    if status.dirty_files == 0 {
        return Committed {
            error: Some(repos::ActionError {
                kind: "nothing-to-commit",
                message: "the working tree is clean".to_string(),
            }),
            before: before.clone(),
            after: before,
            files: Some(0),
        };
    }
    let files = Some(status.dirty_files);
    let error = match git::add_all(abs, git::DEFAULT_TIMEOUT).await {
        Err(e) => Some(commit_action_error(e)),
        Ok(()) => git::commit(abs, full_message, git::DEFAULT_TIMEOUT).await.err().map(commit_action_error),
    };
    let after = git::head_sha(abs, git::DEFAULT_TIMEOUT).await.ok().flatten();
    Committed { error, before, after, files }
}

/// What one push attempt did: git's own report of the ref it moved and the
/// shas it moved between, how many commits that range held, and the outcome.
///
/// `Pulled`/`Committed`'s sibling, and it exists for the same reason they do:
/// two callers, one derivation. The `commits` count in particular has to have
/// exactly one home — it is derived from git's report rather than from
/// anything this machine believed beforehand, and a second copy of that rule
/// is a second chance to derive it from the wrong thing.
pub(crate) struct Pushed {
    pub(crate) error: Option<repos::ActionError>,
    /// git's own account of what moved. `None` when the push failed — there
    /// is no report of a push that did not happen.
    pub(crate) report: Option<git::PushReport>,
    pub(crate) commits: Option<u32>,
}

/// `git push`, then count what it moved.
///
/// No write lock, unlike pull and commit: a push reads the working tree not at
/// all and changes nothing on this side that a concurrent `write_file` could
/// corrupt or be swept into.
pub(crate) async fn push_then_count(abs: &std::path::Path) -> Pushed {
    let outcome = git::push(abs, git::DEFAULT_TIMEOUT).await;
    let error = outcome.as_ref().err().cloned().map(push_action_error);
    let report = outcome.ok();

    // `commits` only where there is a range to count. A new branch and an
    // up-to-date push both report no shas, and a count derived from anything
    // else — the ahead-count `status_v2` holds, say — would be this machine's
    // belief before the push rather than what the push moved.
    let commits = match report.as_ref().and_then(|r| r.before.as_ref().zip(r.after.as_ref())) {
        Some((b, a)) => git::count_range(abs, b, a, git::DEFAULT_TIMEOUT).await.ok(),
        None => None,
    };
    Pushed { error, report, commits }
}

use crate::tools::{Effect, RepoVerb, ToolCtx, ToolError, ToolOutcome};

/// Cap chosen in the design (§ 6): far above any honest message, far below
/// anything that could stress the log or the stdin pipe.
pub(crate) const MAX_MESSAGE_BYTES: usize = 8 * 1024;

/// Reject before anything runs. Control characters other than newline and tab
/// have no place in a commit message; NUL in particular would truncate what
/// git records vs what the event records.
///
/// Returns the REASON, not a response: the route wears it as a 400 body under
/// its `invalid_message:` prefix and the tool wears it as a `ToolError`
/// detail, and neither spelling is this function's business. One set of rules
/// either way — a message a device may not send is not one a model may send.
pub(crate) fn validate_message(message: &str) -> Result<(), String> {
    if message.trim().is_empty() {
        return Err("empty commit message".to_string());
    }
    if message.len() > MAX_MESSAGE_BYTES {
        return Err(format!("message exceeds {MAX_MESSAGE_BYTES} bytes"));
    }
    if message.chars().any(|c| c.is_control() && c != '\n' && c != '\t') {
        return Err("control characters are not allowed".to_string());
    }
    Ok(())
}

/// David's thumb writes one trailer line; an operator's turn writes two.
/// `Committed-by` names the acting operator and the model piloting it, so a
/// git-only reader can tell consent-gated autonomy from a tap (D6).
fn with_tool_trailer(message: &str, device: &str, operator: &str, model: &str) -> String {
    format!("{}\n\nCommitted-from: {}\nCommitted-by: {}/{}\n", message.trim_end(), device, operator, model)
}

/// Reach one verb's async act step from a tool body.
///
/// Tool bodies are sync by design (`tools::RunFn`) and already run on a thread
/// that is allowed to block — `dispatch` is called inside `spawn_blocking`.
/// The verbs are `async` because every git call is a supervised subprocess
/// with a timeout, so blocking this thread on the runtime that spawned it is
/// the one honest way across, and blocking is exactly what that thread is for.
/// Panics if there is no runtime, which cannot happen: the only caller is
/// `dispatch`, and its only caller is the loop.
fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    tokio::runtime::Handle::current().block_on(fut)
}

/// The absolute path of one composed repo, or a refusal naming what IS
/// composed.
///
/// `repos::resolve` is the security boundary and it is called first, exactly
/// as every route calls it first: a manifest name is a KEY into the declared
/// set, never a path fragment, so `../../../etc` is not sanitised — it simply
/// is not a key, and the miss happens before any path is built. The second
/// enumeration is on the MISS path only, and it earns its cost: a model told
/// only "no" cannot recover, and a model told which names exist can (D6').
fn resolve_repo(ctx: &ToolCtx, name: &str) -> Result<std::path::PathBuf, ToolError> {
    let snap = crate::snapshot::WorkspaceSnapshot::load(&ctx.root).unwrap_or_default();
    if let Some(module) = repos::resolve(&ctx.root, &snap.composition, name) {
        return Ok(ctx.root.join(&module.path));
    }
    let declared = repos::declared_modules(&ctx.root, &snap.composition);
    let names: Vec<&str> = declared.iter().map(|m| m.name.as_str()).collect();
    Err(ToolError::new(
        "unknown_repo",
        if names.is_empty() {
            format!("no composed repo named `{name}` — this workspace composes no repos at all")
        } else {
            format!(
                "no composed repo named `{name}`. This workspace composes: {}",
                names.join(", ")
            )
        },
    ))
}

/// The first seven characters of a sha — enough to identify a commit, short
/// enough to read.
fn short(sha: &str) -> &str {
    &sha[..sha.len().min(7)]
}

/// `short`, where an unborn branch means there is no sha at all. Named rather
/// than empty: a blank where a revision belongs reads as a rendering bug.
fn short_or_none(sha: Option<&String>) -> &str {
    sha.map(|s| short(s)).unwrap_or("(none)")
}

/// A failed act step in the words a model can act on: what stopped it, and
/// the machine kind — the same word the event and the HTTP response carry, so
/// a refusal read in a transcript can be grepped in the log (D6').
fn refused(e: &repos::ActionError) -> String {
    format!("{} ({})", e.message, e.kind)
}

/// `commit_repo` — stage everything and commit once, on the branch the repo is
/// already on.
///
/// **A refusal is an `Ok` carrying its record, not an `Err`.** Two reasons,
/// both structural. `repo_events`'s own rule is that a result is a FIELD of the
/// event and never the event's existence — and an `Err` carries no effects, so
/// every refused verb would vanish from the host record and absence would
/// again mean both "nothing happened" and "it failed". And the loop drops a
/// tool that returned `Err` from the offered set for the rest of the turn: one
/// `nothing-to-commit` would take committing away from a turn that had just
/// learned it needed to write something first. An argument this verb cannot
/// act on at all is still an `Err` — nothing ran, so there is nothing to
/// record.
pub(crate) fn commit_repo(ctx: &ToolCtx, name: &str, message: &str) -> Result<ToolOutcome, ToolError> {
    validate_message(message).map_err(|why| ToolError::new("invalid_input", why))?;
    let abs = resolve_repo(ctx, name)?;

    let full_message =
        with_tool_trailer(message, &ctx.turn.device, &ctx.turn.operator, &ctx.turn.model);
    let done = block_on(locked_status_then_commit(&abs, &full_message));
    let subject = message.lines().next().map(str::to_string);

    let text = match &done.error {
        Some(e) => format!("commit refused for {name}: {}\n", refused(e)),
        None => format!(
            "committed {}..{} in {name}: {} ({} staged)\n",
            short_or_none(done.before.as_ref()),
            short_or_none(done.after.as_ref()),
            subject.as_deref().unwrap_or(""),
            done.files.unwrap_or(0),
        ),
    };

    Ok(ToolOutcome {
        text,
        effects: vec![Effect::RepoActed {
            verb: RepoVerb::Commit,
            repo: name.to_string(),
            before: done.before,
            after: done.after,
            subject,
            files: done.files,
            reference: None,
            commits: None,
            error: done.error,
        }],
    })
}

/// `sync_repo` — fetch, then fast-forward. Two verbs, two records.
///
/// The pull runs whether or not the fetch reached the remote: a failed fetch
/// leaves whatever refs were already on disk, and fast-forwarding onto those
/// is still the right thing. Each half carries its own outcome, so "fetched
/// but could not merge" and "could not reach the remote" stay different facts
/// in the record rather than collapsing into one verdict — the same
/// twenty-three-good-answers reasoning `fetch_all` makes across repos, made
/// here across the two halves of one sync.
pub(crate) fn sync_repo(ctx: &ToolCtx, name: &str) -> Result<ToolOutcome, ToolError> {
    let abs = resolve_repo(ctx, name)?;

    let fetch_error = block_on(repos::fetch_repo(&abs));
    let pulled = block_on(locked_dirty_check_then_pull(&abs));

    let fetched = match &fetch_error {
        None => "fetched".to_string(),
        Some(e) => format!("fetch failed: {}", refused(e)),
    };
    let merged = match &pulled.error {
        Some(e) => format!("pull refused: {}", refused(e)),
        None if pulled.before == pulled.after => "already up to date".to_string(),
        None => format!(
            "fast-forwarded {}..{}",
            short_or_none(pulled.before.as_ref()),
            short_or_none(pulled.after.as_ref())
        ),
    };

    Ok(ToolOutcome {
        text: format!("{name}: {fetched}; {merged}\n"),
        effects: vec![
            Effect::RepoActed {
                verb: RepoVerb::Fetch,
                repo: name.to_string(),
                // A fetch moves no branch this repo owns, so it has no
                // before/after to report — `repo_fetched` has never carried
                // them, and inventing a pair here would claim a transition.
                before: None,
                after: None,
                subject: None,
                files: None,
                reference: None,
                commits: None,
                error: fetch_error,
            },
            Effect::RepoActed {
                verb: RepoVerb::Pull,
                repo: name.to_string(),
                before: pulled.before,
                after: pulled.after,
                subject: None,
                files: None,
                reference: None,
                commits: None,
                error: pulled.error,
            },
        ],
    })
}

/// `push_repo` — publish the current branch under the host user's own
/// credential.
pub(crate) fn push_repo(ctx: &ToolCtx, name: &str) -> Result<ToolOutcome, ToolError> {
    let abs = resolve_repo(ctx, name)?;

    let Pushed { error, report, commits } = block_on(push_then_count(&abs));
    let (reference, before, after) = match report {
        Some(r) => (r.reference, r.before, r.after),
        None => (None, None, None),
    };

    let text = match (&error, &before, &after) {
        (Some(e), _, _) => format!("push refused for {name}: {}\n", refused(e)),
        (None, Some(b), Some(a)) => format!(
            "pushed {name} {}: {}..{} ({} commits)\n",
            reference.as_deref().unwrap_or("the current branch"),
            short(b),
            short(a),
            commits.unwrap_or(0),
        ),
        // A new branch and an up-to-date push both land here: git reported no
        // range, and saying which of the two it was would be a guess.
        (None, _, _) => format!(
            "pushed {name} {}: no commit range to report — a new branch, or already up to date\n",
            reference.as_deref().unwrap_or("the current branch"),
        ),
    };

    Ok(ToolOutcome {
        text,
        effects: vec![Effect::RepoActed {
            verb: RepoVerb::Push,
            repo: name.to_string(),
            before,
            after,
            subject: None,
            files: None,
            reference,
            commits,
            error,
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[tokio::test]
    async fn the_commit_path_waits_on_a_held_write_lock() {
        // Same discriminator as the pull path's own test above, one verb
        // over: with the guard removed from `locked_status_then_commit`, the
        // timeout branch completes and this fails. 100ms is the safe
        // direction — a false PASS would need status+add+commit to outrun
        // the timeout while unblocked, against a local tempdir clone.
        let (_remote, clone) = crate::testgit::remote_and_clone();
        std::fs::write(clone.join("dirty.md"), "uncommitted").unwrap();
        let message = "test: from the wait\n\nCommitted-from: test\n";

        let guard = crate::write::write_lock_async().await;
        let blocked = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            locked_status_then_commit(&clone, message),
        )
        .await;
        assert!(blocked.is_err(), "commit must wait while the write lock is held");

        drop(guard);
        let outcome = locked_status_then_commit(&clone, message).await;
        assert!(
            outcome.error.is_none(),
            "a dirty clone with an untracked file commits clean: {:?}",
            outcome.error
        );
        // And a landed commit moved HEAD, which the shas must say rather
        // than the event's absence saying it for them.
        assert_ne!(outcome.before, outcome.after, "a landed commit must move HEAD");
        assert!(outcome.before.is_some(), "a cloned repo has a HEAD");
    }

    #[test]
    fn the_tool_trailer_is_two_lines_and_names_the_model_piloting_the_operator() {
        // D6. One line is a tap; two is a turn. The `Committed-by` half is the
        // one that carries the new fact — which operator acted, and which
        // model was piloting it when it did.
        let out = with_tool_trailer(
            "notes: a line\n",
            "iphone",
            "tycho",
            "zen/deepseek-v4-flash",
        );
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            &lines[lines.len() - 2..],
            ["Committed-from: iphone", "Committed-by: tycho/zen/deepseek-v4-flash"],
            "{out}"
        );
        assert!(out.starts_with("notes: a line\n\n"), "one blank line, no more: {out}");
    }
}
