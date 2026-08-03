//! Bake the build's own identity into the binary.
//!
//! **Why this exists.** `/health` reported `version` — the crate version from
//! `Cargo.toml`, which has not changed in months and is not going to. So a
//! verb that was merged but not rebuilt answered `unknown_tool`, and from a
//! phone that is indistinguishable from the verb never having shipped. Absence
//! and refusal printing the same sentence, one layer below the branch built to
//! stop exactly that.
//!
//! **The staleness trap, which is the whole difficulty.** A build script's
//! output is cached. Emit `GIT_COMMIT` without telling cargo when to look
//! again and the stamp freezes at whatever commit was checked out the first
//! time it ran — the same defect, moved down one layer and made harder to see.
//! `emit_rerun_triggers` is therefore the load-bearing half of this file, not
//! the bookkeeping half.
//!
//! **What this deliberately does not report: dirtiness.** `--dirty` is
//! tempting and would lie. Editing a tracked file changes neither `HEAD` nor
//! the branch ref, so nothing here would fire and the marker would report the
//! working tree as it stood at the last commit. A stamp that lies is worse
//! than a stamp that is silent, so the contract is narrower and true: this
//! names the commit, not the tree.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let commit = git_commit().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=GIT_COMMIT={commit}");
    emit_rerun_triggers();
}

/// Ask git — never the filesystem — what this build is.
///
/// Returns `None` when there is no git or no checkout: a source tarball, a
/// container built without `.git`. The caller stamps `unknown`, which is the
/// honest answer. Guessing a plausible-looking revision would reintroduce the
/// very failure this file exists to close.
fn git_commit() -> Option<String> {
    let out = git(&["rev-parse", "--short", "HEAD"])?;
    // A revision that is not hex is not a revision. `/health`'s contract test
    // pins the same two shapes.
    out.chars()
        .all(|c| c.is_ascii_hexdigit())
        .then_some(out)
        .filter(|s| !s.is_empty())
}

/// Tell cargo what invalidates the stamp.
///
/// Two events move `HEAD`'s commit, and they touch different files:
///
/// - **switching branches** rewrites `HEAD` itself;
/// - **committing on the current branch** rewrites the branch's ref, and
///   leaves `HEAD` untouched.
///
/// Watching only `HEAD` catches the first and misses the second — which is the
/// common case and the one that matters here, since the scenario that prompted
/// this file is "a branch merged, and the engine was not rebuilt".
///
/// Paths come from `git rev-parse`, not from `../../.git`, because that guess
/// is wrong in a git worktree — where `.git` is a *file* pointing elsewhere —
/// and this workspace runs seven of them.
fn emit_rerun_triggers() {
    // `--git-dir` is per-worktree (it holds this worktree's HEAD).
    // `--git-common-dir` is shared (it holds refs/ and packed-refs). In an
    // ordinary checkout they are the same directory; in a worktree they differ,
    // and using one for both is how a worktree build silently stops updating.
    let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]).map(PathBuf::from) else {
        return;
    };
    let common_dir = git(&["rev-parse", "--path-format=absolute", "--git-common-dir"])
        .map(PathBuf::from)
        .unwrap_or_else(|| git_dir.clone());

    rerun_if_present(&git_dir.join("HEAD"));

    // The branch ref, when HEAD is symbolic. A detached HEAD has no ref to
    // watch and does not need one — every move rewrites `HEAD` directly.
    if let Some(ref_name) = git(&["symbolic-ref", "--quiet", "HEAD"]) {
        // Loose ref. Absent when the ref is packed, which is why packed-refs
        // is watched too rather than instead.
        rerun_if_present(&common_dir.join(&ref_name));
    }
    rerun_if_present(&common_dir.join("packed-refs"));
}

/// Emit a rerun trigger only for a path that exists.
///
/// A nonexistent path makes cargo rerun this script on every build, which is
/// cheap here but would rebuild the crate constantly. Absence is normal — a
/// packed ref has no loose file, a detached HEAD has no branch — so it is
/// skipped rather than treated as an error.
fn rerun_if_present(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

/// One git invocation, trimmed, or `None` if git is absent or the command failed.
///
/// Runs in the crate's own directory so the answer is about *this* checkout
/// even when cargo was invoked from somewhere else.
fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;

    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}
