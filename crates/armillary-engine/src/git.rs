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

    #[tokio::test]
    async fn run_git_does_not_inherit_stdin() {
        // The production hang this guards is a git that stops to ask for a
        // credential. With stdin null it reads EOF and exits instead of
        // waiting forever; with stdin inherited a timeout would only convert
        // the hang into a 30-second stall on every repo.
        //
        // `hash-object --stdin-paths` reads paths from stdin until EOF, so it
        // terminates immediately iff stdin is null — and would otherwise
        // outlive this generous deadline.
        let (_remote, clone) = remote_and_clone();
        let out = run_git(
            &clone,
            &["hash-object", "--stdin-paths"],
            Duration::from_secs(5),
        )
        .await
        .expect("must not time out — stdin should be null, not inherited");
        assert_eq!(out.code, 0);
    }

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
}
