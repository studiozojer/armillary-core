//! Real-git fixtures, shared by `git.rs`'s and `repos.rs`'s test modules.
//!
//! `#[cfg(test)]`-gated at the module declaration in `lib.rs`, so none of this
//! is compiled into a release binary.

use std::path::{Path, PathBuf};

/// Run git synchronously for test setup, fully isolated from the machine's own
/// git configuration — no global `user.email`, no inherited `commit.gpgsign`
/// (which would hang on a passphrase prompt), no `init.defaultBranch` drift.
/// Production `git::run_git` deliberately does NOT isolate: the real engine
/// needs the user's credential helpers and SSH config to reach a remote.
pub fn git_sync(dir: &Path, args: &[&str]) {
    git_sync_env(dir, args, &[])
}

/// `git_sync` with extra environment. One implementation of the isolation
/// block, because the alternative is a second copy of it that drifts.
pub fn git_sync_env(dir: &Path, args: &[&str], extra_env: &[(&str, &str)]) {
    let mut cmd = std::process::Command::new("git");
    cmd.arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .env("GIT_AUTHOR_NAME", "test")
        .env("GIT_AUTHOR_EMAIL", "test@example.invalid")
        .env("GIT_COMMITTER_NAME", "test")
        .env("GIT_COMMITTER_EMAIL", "test@example.invalid");
    for (key, value) in extra_env {
        cmd.env(key, value);
    }
    let out = cmd.output().expect("git must be on PATH for these tests");
    assert!(
        out.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
}

pub fn commit(repo: &Path, name: &str, body: &str) {
    std::fs::write(repo.join(name), body).unwrap();
    git_sync(repo, &["add", name]);
    git_sync(repo, &["commit", "-m", name]);
}

/// The committer date stamped on the remote's commit by `advance_remote`.
///
/// **Only its distinctness from "now" is load-bearing**, never its ordering —
/// the assertion it serves is an inequality. A fixed date in the past can never
/// collide with a test run's wall clock, which a future one eventually could.
///
/// Amended 2026-07-31: `%cI` has ONE-SECOND resolution, and the whole fixture
/// completes inside a single second, so the local and remote commits printed
/// byte-identical timestamps and
/// `the_newest_commit_timestamp_is_read_after_the_fast_forward` could not
/// discriminate — the third test in this plan to have that defect, and the one
/// guarding the feature's keystone ordering. Rejected alternative: sleeping past
/// a second boundary, which costs real time on every run and is only
/// probabilistically distinct.
pub const REMOTE_COMMIT_DATE: &str = "1999-12-31T23:59:59+00:00";

/// `commit`, with an explicit committer date so `%cI` is deterministic.
pub fn commit_at(repo: &Path, name: &str, body: &str, committer_date: &str) {
    std::fs::write(repo.join(name), body).unwrap();
    git_sync(repo, &["add", name]);
    git_sync_env(
        repo,
        &["commit", "-m", name],
        &[("GIT_COMMITTER_DATE", committer_date)],
    );
}

/// A bare remote with one commit, plus a clone of it: `(remote, clone)`.
pub fn remote_and_clone() -> (PathBuf, PathBuf) {
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

/// Corrupt the loose object HEAD currently points to, in place.
///
/// Used to prove a genuine read failure (a damaged object) is told apart
/// from an unborn branch (no commits yet) — both make `git log` exit
/// nonzero with empty stdout, so a caller that folds every nonzero exit to
/// "no commits" cannot distinguish a repo with unreadable history from one
/// with none. Objects are written read-only by git, so the mode is loosened
/// before overwriting.
pub fn corrupt_head_object(repo: &Path) {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("git must be on PATH for these tests");
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert_eq!(sha.len(), 40, "expected a full sha from rev-parse HEAD, got {sha:?}");

    let path = repo.join(".git/objects").join(&sha[..2]).join(&sha[2..]);
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    perms.set_readonly(false);
    std::fs::set_permissions(&path, perms).unwrap();
    std::fs::write(&path, b"not a valid zlib stream").unwrap();
}

/// Push a new commit into `remote` from a third clone, so an existing clone
/// becomes genuinely behind rather than being told it is.
pub fn advance_remote(remote: &Path) {
    let other = tempfile::tempdir().unwrap().keep();
    git_sync(
        &other,
        &["clone", remote.to_str().unwrap(), other.to_str().unwrap()],
    );
    commit_at(&other, "from-elsewhere.md", "two", REMOTE_COMMIT_DATE);
    git_sync(&other, &["push", "origin", "main"]);
}
