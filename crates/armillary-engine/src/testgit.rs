//! Real-git fixtures, shared by `git.rs`'s and `sync.rs`'s test modules.
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

pub fn commit(repo: &Path, name: &str, body: &str) {
    std::fs::write(repo.join(name), body).unwrap();
    git_sync(repo, &["add", name]);
    git_sync(repo, &["commit", "-m", name]);
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

/// Push a new commit into `remote` from a third clone, so an existing clone
/// becomes genuinely behind rather than being told it is.
pub fn advance_remote(remote: &Path) {
    let other = tempfile::tempdir().unwrap().keep();
    git_sync(
        &other,
        &["clone", remote.to_str().unwrap(), other.to_str().unwrap()],
    );
    commit(&other, "from-elsewhere.md", "two");
    git_sync(&other, &["push", "origin", "main"]);
}
