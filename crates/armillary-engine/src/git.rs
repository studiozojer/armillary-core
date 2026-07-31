//! Git as a subprocess, one repo at a time.
//!
//! **This module is the engine's first `Command::new`.** Before it, the engine
//! read files, served them, and called Anthropic; it had never executed another
//! program. That is the single categorical change in the sync feature, and it is
//! deliberately confined to this file so the whole of the new authority can be
//! read in one sitting.
//!
//! Two rules hold everywhere below. **Never a shell** — every invocation is an
//! argv array, so there is no string for a metacharacter to live in. **No value
//! from a request ever becomes an argument** — the repo comes from the manifest
//! and every other argument is a literal in this file.
//!
//! The rejected alternative was `git2`/libgit2, which avoids the subprocess
//! entirely. It loses on the thing that decides whether a fetch works at all:
//! git's credential path. The SSH agent, the platform keychain helper,
//! `~/.gitconfig` and its `insteadOf` rules are what let a host reach its
//! remotes, and libgit2 reimplements a subset that diverges exactly there.
//! Shelling out is the narrower real risk despite being the scarier word.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

/// The per-invocation cap. A fetch against an unreachable remote is the
/// expected failure here, not the exotic one — a laptop is asleep, a tailnet
/// is down, a host is off — and an uncapped fetch would hold a sweep open
/// until the network stack gave up on its own schedule.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GitError {
    /// The invocation exceeded its cap and the child was killed.
    Timeout,
    /// git could not be spawned at all — not on PATH, or the repo path is
    /// unusable. Distinct from a nonzero exit, which is an ordinary answer.
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct GitOutput {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl GitOutput {
    pub fn ok(&self) -> bool {
        self.code == 0
    }
}

/// Run one git command in `repo` and collect its output.
///
/// **A nonzero exit is `Ok`, not `Err`.** `rev-parse @{u}` exits 128 when a
/// branch tracks nothing, and that is the answer to "does it have an upstream",
/// not a malfunction. `Err` is reserved for the two cases where no answer
/// exists: the child could not be spawned, or it ran past its cap.
///
/// `stdin` is null. A git that inherits a live stdin can block forever on a
/// credential prompt — the one hang a timeout alone would merely convert into a
/// 30-second stall on every repo.
///
/// **This line has no test, deliberately.** A unit test can only observe it by
/// running a git that reads stdin and checking it returns — which proves
/// nothing unless the test binary's own stdin happens to be open, and under CI
/// it is already `/dev/null`, so such a test passes whether or not this line
/// survives. Rather than ship a green light that means nothing, the guarantee
/// is stated here and left unasserted. Delete this and the failure is a hang in
/// production, not a red suite. (David's ruling, 2026-07-31, after a review
/// caught the non-discriminating test.)
///
/// `kill_on_drop` is what makes the timeout real: `tokio::time::timeout` drops
/// the future, and without this the child would outlive it and keep running.
pub async fn run_git(
    repo: &Path,
    args: &[&str],
    timeout: Duration,
) -> Result<GitOutput, GitError> {
    let mut cmd = Command::new("git");
    cmd.arg("-C")
        .arg(repo)
        .args(args)
        .stdin(Stdio::null())
        .kill_on_drop(true);

    match tokio::time::timeout(timeout, cmd.output()).await {
        Err(_) => Err(GitError::Timeout),
        Ok(Err(e)) => Err(GitError::Failed(e.to_string())),
        Ok(Ok(out)) => Ok(GitOutput {
            code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).trim().to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        }),
    }
}

/// The current branch, or `None` when HEAD is detached.
///
/// `--abbrev-ref HEAD` prints the literal string `HEAD` in the detached case,
/// which is why the comparison below is against a name and not against an exit
/// code.
pub async fn branch(repo: &Path, timeout: Duration) -> Result<Option<String>, GitError> {
    let out = run_git(repo, &["rev-parse", "--abbrev-ref", "HEAD"], timeout).await?;
    if !out.ok() || out.stdout.is_empty() || out.stdout == "HEAD" {
        return Ok(None);
    }
    Ok(Some(out.stdout))
}

/// The tracking branch (`origin/main`), or `None` when the current branch
/// tracks nothing.
pub async fn upstream(repo: &Path, timeout: Duration) -> Result<Option<String>, GitError> {
    let out = run_git(
        repo,
        &["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        timeout,
    )
    .await?;
    if !out.ok() || out.stdout.is_empty() {
        return Ok(None);
    }
    Ok(Some(out.stdout))
}

/// True when the working tree has anything uncommitted, **including untracked
/// files**. `--porcelain` reports those by default and that is wanted: a new
/// uncommitted note in the commons is work, and a sweep must not act around it.
///
/// Unlike `branch`/`upstream`/`newest_commit`, a git failure here is `Err`, not
/// folded into `Ok(false)`: `bool` has no natural "no answer" value the way
/// `Option<String>` does, so silently reporting "not dirty" would misrepresent
/// a status call that never actually ran.
pub async fn is_dirty(repo: &Path, timeout: Duration) -> Result<bool, GitError> {
    let out = run_git(repo, &["status", "--porcelain"], timeout).await?;
    if !out.ok() {
        return Err(GitError::Failed(out.stderr));
    }
    Ok(!out.stdout.is_empty())
}

/// The committer date of HEAD in strict ISO 8601 (`%cI`), or `None` in a repo
/// with no commits yet.
///
/// Committer date rather than author date on purpose: a rebased or
/// cherry-picked commit keeps its original author date, which would report a
/// repo as older than the work actually landing in it.
pub async fn newest_commit(repo: &Path, timeout: Duration) -> Result<Option<String>, GitError> {
    let out = run_git(repo, &["log", "-1", "--format=%cI"], timeout).await?;
    if !out.ok() || out.stdout.is_empty() {
        return Ok(None);
    }
    Ok(Some(out.stdout))
}

/// What a repo's local state permits.
///
/// Exactly one verdict per repo, and the precedence below is load-bearing
/// because several are simultaneously true in practice.
///
/// **Detached → NoUpstream → Diverged → Dirty → Behind → Current.**
///
/// `Detached` first because there is no branch to reason about at all.
/// `NoUpstream` next because nothing downstream is computable without one.
/// `Diverged` above `Dirty` because divergence is a durable fact about history
/// that survives cleaning the working tree — reporting "dirty" for a repo that
/// is also diverged sends the reader to fix the thing that was not the problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Current,
    Behind { commits: u32 },
    Diverged,
    Dirty,
    NoUpstream,
    Detached,
}

/// `git fetch --prune`.
///
/// Touches no working tree and no branch, which is why the sweep runs it
/// unconditionally — on the dirty feature branch too. It is what makes the
/// report *true* rather than a reading of whatever the last fetch happened to
/// leave behind.
///
/// A repo with no remote configured is an `Err`, not a silent success: the
/// sweep reports it rather than counting it as fetched.
///
/// The `git remote` pre-check exists because `git fetch --prune` on a repo
/// with zero remotes configured is not itself a failure — it exits 0 with no
/// output, having correctly done nothing. That is the right answer to "did
/// the fetch fail," but the wrong one to "is there anything here to sync,"
/// which is the question this function actually answers for the sweep.
pub async fn fetch(repo: &Path, timeout: Duration) -> Result<(), GitError> {
    let remotes = run_git(repo, &["remote"], timeout).await?;
    if !remotes.ok() {
        return Err(GitError::Failed(remotes.stderr));
    }
    if remotes.stdout.trim().is_empty() {
        return Err(GitError::Failed("no remote configured".to_string()));
    }

    require_ok(
        run_git(repo, &["fetch", "--prune"], timeout).await?,
        "git fetch",
    )
}

/// Turn a nonzero exit into a `Failed`, naming the command when git itself
/// said nothing.
///
/// One implementation, added 2026-07-31 on David's ruling after a review found
/// this block written out verbatim in both `fetch` and `fast_forward`. The
/// module already had an idiom for a hard-fail exit (`is_dirty`, and `fetch`'s
/// own `git remote` pre-check); a second one inlined in two places is how a
/// third caller ends up copying the wrong one.
fn require_ok(out: GitOutput, cmd: &str) -> Result<(), GitError> {
    if out.ok() {
        return Ok(());
    }
    Err(GitError::Failed(if out.stderr.is_empty() {
        format!("{cmd} exited {}", out.code)
    } else {
        out.stderr
    }))
}

/// Classify the repo against its upstream, using only local refs.
///
/// Deliberately does NOT fetch. The caller decides whether the refs it reads
/// are fresh — `POST /sync` fetches first, `GET /sync` does not and says so.
pub async fn verdict(repo: &Path, timeout: Duration) -> Result<Verdict, GitError> {
    if branch(repo, timeout).await?.is_none() {
        return Ok(Verdict::Detached);
    }
    if upstream(repo, timeout).await?.is_none() {
        return Ok(Verdict::NoUpstream);
    }

    // `HEAD...@{u}` with --left-right --count prints "<ahead>\t<behind>":
    // left side is commits in HEAD and not upstream, right side the reverse.
    let out = run_git(
        repo,
        &["rev-list", "--left-right", "--count", "HEAD...@{u}"],
        timeout,
    )
    .await?;
    if !out.ok() {
        return Err(GitError::Failed(out.stderr));
    }
    // Unparseable output is an error, not a zero. Defaulting to 0/0 would
    // report `Current` — "nothing to do, all good" — from a function whose
    // answer gates whether a fast-forward is safe. A silent wrong-and-
    // reassuring reading is the exact failure this feature's report exists to
    // prevent, so it must not appear in the machinery underneath it.
    let mut parts = out.stdout.split_whitespace();
    let ahead = parts.next().and_then(|s| s.parse::<u32>().ok());
    let behind = parts.next().and_then(|s| s.parse::<u32>().ok());
    let (Some(ahead), Some(behind)) = (ahead, behind) else {
        return Err(GitError::Failed(format!(
            "could not parse `rev-list --left-right --count` output: {:?}",
            out.stdout
        )));
    };

    if ahead > 0 && behind > 0 {
        return Ok(Verdict::Diverged);
    }
    if is_dirty(repo, timeout).await? {
        return Ok(Verdict::Dirty);
    }
    if behind > 0 {
        return Ok(Verdict::Behind { commits: behind });
    }
    Ok(Verdict::Current)
}

/// `git merge --ff-only @{u}`.
///
/// The whole safety argument of this feature is this one flag. The merge
/// succeeds only when the local branch is a strict ancestor of upstream, so a
/// conflict is structurally impossible rather than handled, no merge commit is
/// ever created, and a diverged branch is refused with HEAD unmoved.
///
/// Callers must only reach this for `Verdict::Behind`. It is safe if they do
/// not — git refuses — but the report would then carry a failure the sweep
/// could have predicted.
pub async fn fast_forward(repo: &Path, timeout: Duration) -> Result<(), GitError> {
    require_ok(
        run_git(repo, &["merge", "--ff-only", "@{u}"], timeout).await?,
        "git merge --ff-only",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Run git synchronously for test setup, fully isolated from the machine's
    /// own git configuration.
    ///
    /// `GIT_CONFIG_GLOBAL`/`GIT_CONFIG_SYSTEM` are neutralized and the identity
    /// is supplied by environment, so these tests do not depend on the
    /// developer having a `user.email` set, do not inherit a global
    /// `commit.gpgsign = true` (which would hang waiting for a passphrase), and
    /// do not vary with `init.defaultBranch`. Production `run_git` deliberately
    /// does NOT do this — the real engine needs the user's credential helpers,
    /// SSH config and `insteadOf` rules to reach a remote at all.
    fn git_sync(dir: &std::path::Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .env("GIT_AUTHOR_NAME", "test")
            .env("GIT_AUTHOR_EMAIL", "test@example.invalid")
            .env("GIT_COMMITTER_NAME", "test")
            .env("GIT_COMMITTER_EMAIL", "test@example.invalid")
            .output()
            .expect("git must be on PATH for these tests");
        assert!(
            out.status.success(),
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    }

    fn commit(repo: &std::path::Path, name: &str, body: &str) {
        std::fs::write(repo.join(name), body).unwrap();
        git_sync(repo, &["add", name]);
        git_sync(repo, &["commit", "-m", name]);
    }

    /// A bare "remote" with one commit, plus a clone of it. Returned as
    /// (remote, clone). Both leaked via `keep()` — short-lived test processes,
    /// and a live path matters more than reclaiming a tempdir.
    fn remote_and_clone() -> (PathBuf, PathBuf) {
        let remote = tempfile::tempdir().unwrap().keep();
        git_sync(&remote, &["init", "--bare", "--initial-branch=main", "."]);

        let seed = tempfile::tempdir().unwrap().keep();
        git_sync(&seed, &["init", "--initial-branch=main", "."]);
        commit(&seed, "seed.md", "one");
        git_sync(&seed, &["remote", "add", "origin", remote.to_str().unwrap()]);
        git_sync(&seed, &["push", "-u", "origin", "main"]);

        let clone = tempfile::tempdir().unwrap().keep();
        git_sync(
            &clone,
            &["clone", remote.to_str().unwrap(), clone.to_str().unwrap()],
        );
        (remote, clone)
    }

    #[tokio::test]
    async fn run_git_reports_stdout_and_a_zero_code() {
        let (_remote, clone) = remote_and_clone();
        let out = run_git(&clone, &["rev-parse", "--abbrev-ref", "HEAD"], DEFAULT_TIMEOUT)
            .await
            .unwrap();
        assert_eq!(out.code, 0);
        assert_eq!(out.stdout, "main");
    }

    #[tokio::test]
    async fn run_git_reports_a_nonzero_code_rather_than_erroring() {
        // A failing git command is DATA, not a transport failure: `@{u}` on a
        // branch with no upstream exits 128, and that is the answer, not a bug.
        let (_remote, clone) = remote_and_clone();
        git_sync(&clone, &["checkout", "-b", "orphan"]);
        let out = run_git(
            &clone,
            &["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            DEFAULT_TIMEOUT,
        )
        .await
        .unwrap();
        assert_ne!(out.code, 0);
    }

    #[tokio::test]
    async fn run_git_times_out_rather_than_hanging() {
        // The deadline is driven to zero rather than the command being made
        // slow. No git subcommand is reliably slow enough to race against a
        // wall-clock timeout without being flaky, and a genuinely blocking one
        // (a fetch against a black hole) makes the unit suite depend on the
        // network. A 1ns deadline is already elapsed when the timeout is first
        // polled — process spawn costs microseconds at minimum — so this is
        // deterministic in a way a 200ms-vs-fast-command race is not.
        let (_remote, clone) = remote_and_clone();
        let err = run_git(&clone, &["status", "--porcelain"], Duration::from_nanos(1))
            .await
            .unwrap_err();
        assert_eq!(err, GitError::Timeout);
    }

    // There is deliberately NO test for `.stdin(Stdio::null())`.
    //
    // Amended 2026-07-31, David ruling, after a task review found the test that
    // was here could not discriminate. It ran `git hash-object --stdin-paths`
    // and asserted it returned rather than hanging — which only proves anything
    // when the TEST BINARY'S OWN stdin is open. Under CI, or any non-interactive
    // shell, stdin is already `/dev/null`, the child hits EOF either way, and the
    // test passes against a regression that deletes the very line it guards.
    //
    // A green light that means nothing is worse than an absent one, and the
    // honest fix was to stop claiming the coverage. The reasoning lives on the
    // production line instead. (Making it real would mean holding a pipe and
    // `dup2`-ing it onto fd 0 — an `unsafe` block and platform-specific fd
    // handling in a crate that has neither, to guard one setting.)

    #[tokio::test]
    async fn branch_reads_the_current_branch() {
        let (_remote, clone) = remote_and_clone();
        assert_eq!(
            branch(&clone, DEFAULT_TIMEOUT).await.unwrap(),
            Some("main".to_string())
        );
    }

    #[tokio::test]
    async fn branch_is_none_when_head_is_detached() {
        let (_remote, clone) = remote_and_clone();
        let head = run_git(&clone, &["rev-parse", "HEAD"], DEFAULT_TIMEOUT)
            .await
            .unwrap()
            .stdout;
        git_sync(&clone, &["checkout", &head]);
        assert_eq!(branch(&clone, DEFAULT_TIMEOUT).await.unwrap(), None);
    }

    #[tokio::test]
    async fn upstream_reads_the_tracking_branch() {
        let (_remote, clone) = remote_and_clone();
        assert_eq!(
            upstream(&clone, DEFAULT_TIMEOUT).await.unwrap(),
            Some("origin/main".to_string())
        );
    }

    #[tokio::test]
    async fn upstream_is_none_on_a_branch_that_tracks_nothing() {
        let (_remote, clone) = remote_and_clone();
        git_sync(&clone, &["checkout", "-b", "feat/local-only"]);
        assert_eq!(upstream(&clone, DEFAULT_TIMEOUT).await.unwrap(), None);
    }

    #[tokio::test]
    async fn is_dirty_sees_an_uncommitted_change() {
        let (_remote, clone) = remote_and_clone();
        assert!(!is_dirty(&clone, DEFAULT_TIMEOUT).await.unwrap());
        std::fs::write(clone.join("seed.md"), "edited").unwrap();
        assert!(is_dirty(&clone, DEFAULT_TIMEOUT).await.unwrap());
    }

    #[tokio::test]
    async fn is_dirty_sees_an_untracked_file() {
        // `--porcelain` reports untracked files by default, and that is wanted:
        // a new uncommitted note in the commons is work that a fast-forward
        // could disturb, so it counts.
        let (_remote, clone) = remote_and_clone();
        std::fs::write(clone.join("new-note.md"), "draft").unwrap();
        assert!(is_dirty(&clone, DEFAULT_TIMEOUT).await.unwrap());
    }

    #[tokio::test]
    async fn newest_commit_is_an_iso_timestamp() {
        let (_remote, clone) = remote_and_clone();
        let ts = newest_commit(&clone, DEFAULT_TIMEOUT).await.unwrap().unwrap();
        // `%cI` is strict ISO 8601: 2026-07-30T14:22:07-07:00
        assert!(ts.len() >= 20, "expected an ISO timestamp, got {ts:?}");
        assert_eq!(&ts[4..5], "-");
        assert!(ts.contains('T'));
    }

    /// Push a new commit into the bare remote from a second clone, so the
    /// first clone becomes genuinely behind — as opposed to being told it is.
    fn advance_remote(remote: &std::path::Path) {
        let other = tempfile::tempdir().unwrap().keep();
        git_sync(
            &other,
            &["clone", remote.to_str().unwrap(), other.to_str().unwrap()],
        );
        commit(&other, "from-elsewhere.md", "two");
        git_sync(&other, &["push", "origin", "main"]);
    }

    #[tokio::test]
    async fn verdict_is_current_on_a_clone_nobody_moved() {
        let (_remote, clone) = remote_and_clone();
        fetch(&clone, DEFAULT_TIMEOUT).await.unwrap();
        assert_eq!(verdict(&clone, DEFAULT_TIMEOUT).await.unwrap(), Verdict::Current);
    }

    #[tokio::test]
    async fn verdict_is_behind_after_the_remote_moves() {
        let (remote, clone) = remote_and_clone();
        advance_remote(&remote);
        fetch(&clone, DEFAULT_TIMEOUT).await.unwrap();
        assert_eq!(
            verdict(&clone, DEFAULT_TIMEOUT).await.unwrap(),
            Verdict::Behind { commits: 1 }
        );
    }

    #[tokio::test]
    async fn verdict_is_diverged_when_both_sides_moved() {
        let (remote, clone) = remote_and_clone();
        advance_remote(&remote);
        commit(&clone, "local-only.md", "mine");
        fetch(&clone, DEFAULT_TIMEOUT).await.unwrap();
        assert_eq!(verdict(&clone, DEFAULT_TIMEOUT).await.unwrap(), Verdict::Diverged);
    }

    #[tokio::test]
    async fn verdict_is_dirty_and_dirty_outranks_behind() {
        let (remote, clone) = remote_and_clone();
        advance_remote(&remote);
        std::fs::write(clone.join("seed.md"), "edited").unwrap();
        fetch(&clone, DEFAULT_TIMEOUT).await.unwrap();
        // Behind AND dirty. Dirty is the verdict, because it is the one that
        // blocks the fast-forward.
        assert_eq!(verdict(&clone, DEFAULT_TIMEOUT).await.unwrap(), Verdict::Dirty);
    }

    #[tokio::test]
    async fn verdict_is_diverged_and_diverged_outranks_dirty() {
        // Both true. Diverged wins: it is a durable fact about history that
        // will still be there after the working tree is cleaned, so reporting
        // "dirty" would send you to fix the wrong thing.
        let (remote, clone) = remote_and_clone();
        advance_remote(&remote);
        commit(&clone, "local-only.md", "mine");
        std::fs::write(clone.join("seed.md"), "edited").unwrap();
        fetch(&clone, DEFAULT_TIMEOUT).await.unwrap();
        assert_eq!(verdict(&clone, DEFAULT_TIMEOUT).await.unwrap(), Verdict::Diverged);
    }

    #[tokio::test]
    async fn verdict_is_no_upstream_on_an_untracked_branch() {
        let (_remote, clone) = remote_and_clone();
        git_sync(&clone, &["checkout", "-b", "feat/local-only"]);
        assert_eq!(verdict(&clone, DEFAULT_TIMEOUT).await.unwrap(), Verdict::NoUpstream);
    }

    #[tokio::test]
    async fn verdict_is_detached_before_it_is_anything_else() {
        let (_remote, clone) = remote_and_clone();
        let head = run_git(&clone, &["rev-parse", "HEAD"], DEFAULT_TIMEOUT)
            .await
            .unwrap()
            .stdout;
        std::fs::write(clone.join("seed.md"), "edited").unwrap();
        git_sync(&clone, &["checkout", "--detach", &head]);
        assert_eq!(verdict(&clone, DEFAULT_TIMEOUT).await.unwrap(), Verdict::Detached);
    }

    #[tokio::test]
    async fn fast_forward_applies_the_remote_commits() {
        let (remote, clone) = remote_and_clone();
        advance_remote(&remote);
        fetch(&clone, DEFAULT_TIMEOUT).await.unwrap();
        fast_forward(&clone, DEFAULT_TIMEOUT).await.unwrap();
        assert!(clone.join("from-elsewhere.md").exists());
        assert_eq!(verdict(&clone, DEFAULT_TIMEOUT).await.unwrap(), Verdict::Current);
    }

    #[tokio::test]
    async fn fast_forward_refuses_a_diverged_branch_and_leaves_head_unmoved() {
        let (remote, clone) = remote_and_clone();
        advance_remote(&remote);
        commit(&clone, "local-only.md", "mine");
        fetch(&clone, DEFAULT_TIMEOUT).await.unwrap();

        let before = run_git(&clone, &["rev-parse", "HEAD"], DEFAULT_TIMEOUT)
            .await
            .unwrap()
            .stdout;
        assert!(fast_forward(&clone, DEFAULT_TIMEOUT).await.is_err());
        let after = run_git(&clone, &["rev-parse", "HEAD"], DEFAULT_TIMEOUT)
            .await
            .unwrap()
            .stdout;

        assert_eq!(before, after, "a refused fast-forward must not move HEAD");
        // And no merge was created behind our back.
        assert!(!clone.join("from-elsewhere.md").exists());
    }

    #[tokio::test]
    async fn fetch_is_an_error_on_a_repo_with_no_remote() {
        // Not a panic and not a silent success — the sweep needs to report it.
        let solo = tempfile::tempdir().unwrap().keep();
        git_sync(&solo, &["init", "--initial-branch=main", "."]);
        commit(&solo, "alone.md", "solo");
        assert!(fetch(&solo, DEFAULT_TIMEOUT).await.is_err());
    }
}
