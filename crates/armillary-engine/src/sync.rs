//! The sweep: fetch (or don't), classify, fast-forward what a classification
//! permits — one pass over every repo `repos::declared_modules` finds on
//! disk, and what was found and deliberately not touched.
//!
//! Enumeration itself — which modules are declared, which checkouts exist on
//! disk, undeclared-checkout detection, and the git-authority gate — lives in
//! `repos.rs`; this module consumes those and adds the network and mutation
//! half: `git::fetch`, `git::verdict`, `git::pull_ff`.

use std::path::Path;

use crate::git::{self, GitError, Verdict};
use crate::repos::{declared_modules, gate_enabled, undeclared_checkouts, DeclaredModule};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// How many repos are in flight at once.
///
/// Twenty-four serial fetches over a sleeping tailnet is a minute of spinner,
/// and one unreachable remote must not hold up the other twenty-three. Bounded
/// rather than unbounded because two dozen simultaneous SSH handshakes is a
/// different kind of rude.
const CONCURRENCY: usize = 8;

#[derive(Debug, Clone, Serialize)]
pub struct RepoReport {
    pub name: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    /// `synced` · `current` · `behind` · `skipped` · `error`.
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commits: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub newest_commit: Option<String>,
    /// Present and `true` only when the repo has a `.gitmodules`.
    ///
    /// D5: submodules are fetched but never updated, so a fast-forward moves
    /// the submodule POINTER and leaves the submodule checkout where it was.
    /// That is a deliberate v0 limit — `repos/armillary-site` is the only repo
    /// with any, and it is a deploy artifact rather than something read from a
    /// phone — but a limit nobody can see is indistinguishable from a bug, so
    /// the report names it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub submodules: Option<bool>,
    /// A fetch that failed while the repo remained readable. Reported beside
    /// the verdict rather than replacing it: a repo with no remote still has a
    /// branch and a timestamp worth printing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fetch_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NotComposed {
    pub path: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SyncReport {
    /// The gate, so the client can hide the action rather than provoke a 403.
    pub enabled: bool,
    /// False for a status read. The client says "as of the last sync" when it
    /// is false — without this, a stale `current` and a fresh one print the
    /// same word.
    pub fetched: bool,
    pub repos: Vec<RepoReport>,
    pub not_composed: Vec<NotComposed>,
}

/// Sweep the workspace.
///
/// `perform = true` fetches and fast-forwards (`POST /sync`); `perform = false`
/// reads local refs and touches nothing (`GET /sync`).
///
/// **Never returns `Err`.** A repo that fails is a line in the report, not a
/// failed request — twenty-three good answers and one error is the useful
/// outcome, and a 500 would throw all of it away.
pub async fn sweep(root: &Path, perform: bool) -> SyncReport {
    let declared = declared_modules(root);
    let not_composed = undeclared_checkouts(root, &declared)
        .into_iter()
        .map(|path| NotComposed { path })
        .collect();

    let permits = Arc::new(Semaphore::new(CONCURRENCY));
    let mut handles = Vec::with_capacity(declared.len());
    let mut labels = Vec::with_capacity(declared.len());

    for module in declared {
        // Kept out of the spawn so a task that dies still has a name and a path
        // to be reported under.
        labels.push(module.clone());
        let permits = permits.clone();
        let abs = root.join(&module.path);
        handles.push(tokio::spawn(async move {
            // The semaphore is never closed, so acquire cannot fail.
            let _permit = permits.acquire().await.expect("semaphore is open");
            one_repo(&abs, module, perform).await
        }));
    }

    let mut repos = Vec::with_capacity(handles.len());
    for (module, handle) in labels.into_iter().zip(handles) {
        match handle.await {
            Ok(report) => repos.push(report),
            // A panicked or cancelled task must not disappear. The contract is
            // that a repo which fails is a LINE, and a silently absent row
            // reads as "not composed" — the exact confusion section four of the
            // report exists to prevent.
            //
            // Unreachable by construction today: `one_repo` has no panic sites,
            // and the semaphore is never closed. It therefore carries NO test
            // rather than a faked one — the same call made about
            // `.stdin(Stdio::null())` in git.rs.
            Err(_) => repos.push(RepoReport {
                name: module.name,
                path: module.path,
                branch: None,
                status: "error",
                reason: Some("task-failed"),
                commits: None,
                newest_commit: None,
                submodules: None,
                fetch_error: None,
            }),
        }
    }

    // No sort. `handles` was pushed in `declared_modules` order and is awaited
    // by index, so `repos` is already in manifest-declared order — operators,
    // then commons, then repos, which is the order the app renders. The
    // `sort_by_key(|r| r.path.clone())` that used to sit here re-ordered
    // alphabetically and silently replaced the right answer with a plausible
    // one, while its own comment claimed to be restoring it.

    SyncReport {
        enabled: gate_enabled(root),
        fetched: perform,
        repos,
        not_composed,
    }
}

/// One repo, start to finish. Fetch (if performing), classify, fast-forward if
/// the verdict permits, then read the timestamp.
async fn one_repo(abs: &Path, module: DeclaredModule, perform: bool) -> RepoReport {
    let t = git::DEFAULT_TIMEOUT;

    let mut fetch_error = None;
    if perform {
        if let Err(e) = git::fetch(abs, t).await {
            fetch_error = Some(match e {
                GitError::Timeout => "timed out".to_string(),
                GitError::Failed(msg) => msg,
                // Unreachable in practice: no request value reaches `fetch`.
                // Carried like `Failed` rather than panicking or dropping it.
                GitError::InvalidArg(msg) => msg,
            });
        }
    }

    let branch = git::branch(abs, t).await.ok().flatten();

    let mut report = RepoReport {
        name: module.name,
        path: module.path,
        branch,
        status: "error",
        reason: Some("git-error"),
        commits: None,
        newest_commit: None,
        submodules: None,
        fetch_error,
    };

    let verdict = match git::verdict(abs, t).await {
        Ok(v) => v,
        Err(GitError::Timeout) => {
            report.status = "skipped";
            report.reason = Some("timeout");
            return report;
        }
        Err(GitError::Failed(_)) => return report,
        // Unreachable in practice: no request value reaches `verdict`.
        // Treated like `Failed`.
        Err(GitError::InvalidArg(_)) => return report,
    };

    match verdict {
        Verdict::Behind { commits } if perform => match git::pull_ff(abs, t).await {
            Ok(()) => {
                report.status = "synced";
                report.reason = None;
                report.commits = Some(commits);
            }
            // git refused a fast-forward we predicted would apply. Report it
            // rather than claiming success — the prediction, not the merge, is
            // what was wrong.
            Err(_) => {
                report.status = "skipped";
                report.reason = Some("diverged");
            }
        },
        Verdict::Behind { commits } => {
            report.status = "behind";
            report.reason = None;
            report.commits = Some(commits);
        }
        Verdict::Current => {
            report.status = "current";
            report.reason = None;
        }
        Verdict::Diverged => {
            report.status = "skipped";
            report.reason = Some("diverged");
        }
        Verdict::Dirty => {
            report.status = "skipped";
            report.reason = Some("dirty");
        }
        Verdict::NoUpstream => {
            report.status = "skipped";
            report.reason = Some("no-upstream");
        }
        Verdict::Detached => {
            report.status = "skipped";
            report.reason = Some("detached");
        }
    }

    // Both read LAST, and for the same reason. `.gitmodules` can ARRIVE in the
    // fast-forward, so a pre-merge read reports `None` for a repo that now has
    // an un-updated submodule pointer — the same read-before-mutation shape
    // this function is built to get right for `newest_commit`.
    report.submodules = abs.join(".gitmodules").exists().then_some(true);
    report.newest_commit = git::newest_commit(abs, t).await.ok().flatten();

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    use crate::testgit::{advance_remote, git_sync, remote_and_clone};

    /// A workspace whose declared modules are REAL clones of a shared bare
    /// remote. Returns (root, remote).
    fn live_workspace() -> (PathBuf, PathBuf) {
        let root = tempfile::tempdir().unwrap().keep();
        git_sync(&root, &["init", "--initial-branch=main", "."]);

        let (remote, _first) = remote_and_clone();
        let module = root.join("repos").join("jianyi");
        fs::create_dir_all(root.join("repos")).unwrap();
        git_sync(
            &root,
            &["clone", remote.to_str().unwrap(), module.to_str().unwrap()],
        );

        fs::write(
            root.join("modules.local.toml"),
            "[router]\nsync = true\n\n\
             [[repos]]\nname = \"jianyi\"\npath = \"repos/jianyi\"\n",
        )
        .unwrap();
        (root, remote)
    }

    fn repo<'a>(report: &'a SyncReport, path: &str) -> &'a RepoReport {
        report
            .repos
            .iter()
            .find(|r| r.path == path)
            .unwrap_or_else(|| panic!("no report for {path}; got {:?}", report.repos))
    }

    #[tokio::test]
    async fn a_sweep_fast_forwards_a_behind_repo_and_reports_the_count() {
        let (root, remote) = live_workspace();
        advance_remote(&remote);

        let report = sweep(&root, true).await;
        let r = repo(&report, "repos/jianyi");

        assert!(report.fetched);
        assert_eq!(r.status, "synced");
        assert_eq!(r.commits, Some(1));
        assert!(root.join("repos/jianyi/from-elsewhere.md").exists());
    }

    #[tokio::test]
    async fn the_newest_commit_timestamp_is_read_after_the_fast_forward() {
        // THE case this feature's report can fail plausibly rather than
        // loudly. Read on the wrong side of the merge, the timestamp is the
        // commit you already had — reassuring, wrong, and shaped identically
        // to a correct answer. Nothing else in the suite can see the
        // difference, so the ordering is pinned here explicitly.
        let (root, remote) = live_workspace();
        let before = crate::git::newest_commit(
            &root.join("repos/jianyi"),
            crate::git::DEFAULT_TIMEOUT,
        )
        .await
        .unwrap()
        .unwrap();

        advance_remote(&remote);
        let report = sweep(&root, true).await;
        let r = repo(&report, "repos/jianyi");

        assert_eq!(r.status, "synced");
        assert_ne!(
            r.newest_commit.as_deref(),
            Some(before.as_str()),
            "the timestamp was read before the fast-forward"
        );
    }

    #[tokio::test]
    async fn a_status_read_never_fetches_and_never_moves_anything() {
        let (root, remote) = live_workspace();
        advance_remote(&remote);

        let report = sweep(&root, false).await;
        let r = repo(&report, "repos/jianyi");

        assert!(!report.fetched);
        // Local refs only, and nothing has fetched them, so the remote's new
        // commit is invisible — correctly, and the `fetched: false` flag is how
        // the client knows to say "as of the last sync".
        assert_eq!(r.status, "current");
        assert!(!root.join("repos/jianyi/from-elsewhere.md").exists());
    }

    #[tokio::test]
    async fn a_status_read_reports_behind_without_applying_it() {
        let (root, remote) = live_workspace();
        advance_remote(&remote);
        // Fetch out of band so the local refs know, without sweeping.
        crate::git::fetch(&root.join("repos/jianyi"), crate::git::DEFAULT_TIMEOUT)
            .await
            .unwrap();

        let report = sweep(&root, false).await;
        let r = repo(&report, "repos/jianyi");

        assert_eq!(r.status, "behind");
        assert_eq!(r.commits, Some(1));
        assert!(!root.join("repos/jianyi/from-elsewhere.md").exists());
    }

    #[tokio::test]
    async fn a_dirty_repo_is_skipped_and_its_working_tree_is_untouched() {
        let (root, remote) = live_workspace();
        advance_remote(&remote);
        let seed = root.join("repos/jianyi/seed.md");
        fs::write(&seed, "my uncommitted edit").unwrap();

        let report = sweep(&root, true).await;
        let r = repo(&report, "repos/jianyi");

        assert_eq!(r.status, "skipped");
        assert_eq!(r.reason, Some("dirty"));
        assert_eq!(fs::read_to_string(&seed).unwrap(), "my uncommitted edit");
        assert!(!root.join("repos/jianyi/from-elsewhere.md").exists());
    }

    #[tokio::test]
    async fn a_repo_with_no_upstream_is_skipped_with_that_reason() {
        let (root, _remote) = live_workspace();
        git_sync(&root.join("repos/jianyi"), &["checkout", "-b", "feat/local"]);

        let report = sweep(&root, true).await;
        let r = repo(&report, "repos/jianyi");

        assert_eq!(r.status, "skipped");
        assert_eq!(r.reason, Some("no-upstream"));
        assert_eq!(r.branch.as_deref(), Some("feat/local"));
    }

    #[tokio::test]
    async fn the_router_root_is_swept_too() {
        let (root, _remote) = live_workspace();
        let report = sweep(&root, true).await;
        let r = repo(&report, ".");
        // No remote on the test root, so fetch fails and it is reported —
        // never silently dropped.
        assert_eq!(r.name, "armillary");
        assert!(r.fetch_error.is_some());
    }

    #[tokio::test]
    async fn undeclared_checkouts_reach_the_report() {
        let (root, remote) = live_workspace();
        fs::create_dir_all(root.join("operators")).unwrap();
        git_sync(
            &root,
            &[
                "clone",
                remote.to_str().unwrap(),
                root.join("operators/ariadne").to_str().unwrap(),
            ],
        );

        let report = sweep(&root, true).await;
        assert_eq!(
            report.not_composed.iter().map(|n| n.path.as_str()).collect::<Vec<_>>(),
            vec!["operators/ariadne"]
        );
        // And it was NOT swept.
        assert!(report.repos.iter().all(|r| r.path != "operators/ariadne"));
    }

    #[tokio::test]
    async fn the_report_is_in_manifest_order_not_alphabetical() {
        // Declared order is operators -> commons -> repos. Alphabetical by path
        // is a DIFFERENT order (zojercommons sorts last, but is declared
        // second), so this fails against a sort and passes against declaration
        // order.
        let (root, _remote) = live_workspace();
        fs::create_dir_all(root.join("operators/tycho/.git")).unwrap();
        fs::create_dir_all(root.join("zojercommons/.git")).unwrap();
        fs::write(
            root.join("modules.local.toml"),
            "[router]\nsync = true\n\n\
             [[operators]]\nname = \"tycho\"\npath = \"operators/tycho\"\n\n\
             [[commons]]\nname = \"zojercommons\"\npath = \"zojercommons\"\n\n\
             [[repos]]\nname = \"jianyi\"\npath = \"repos/jianyi\"\n",
        )
        .unwrap();

        let report = sweep(&root, false).await;
        let paths: Vec<&str> = report.repos.iter().map(|r| r.path.as_str()).collect();
        assert_eq!(
            paths,
            vec![".", "operators/tycho", "zojercommons", "repos/jianyi"]
        );
    }

    #[tokio::test]
    async fn a_repo_with_submodules_says_so_and_they_are_not_updated() {
        // D5. The flag is what keeps a deliberate limit from reading as a bug:
        // the fast-forward moved the pointer, the submodule checkout did not
        // move, and nothing else in the report would show that.
        let (root, remote) = live_workspace();
        fs::write(
            root.join("repos/jianyi/.gitmodules"),
            "[submodule \"content\"]\n\tpath = content\n\turl = ../content.git\n",
        )
        .unwrap();
        git_sync(&root.join("repos/jianyi"), &["add", ".gitmodules"]);
        git_sync(&root.join("repos/jianyi"), &["commit", "-m", "add submodule decl"]);
        git_sync(&root.join("repos/jianyi"), &["push", "origin", "main"]);
        advance_remote(&remote);

        let report = sweep(&root, true).await;
        assert_eq!(repo(&report, "repos/jianyi").submodules, Some(true));
    }

    #[tokio::test]
    async fn submodules_arriving_in_the_fast_forward_are_still_reported() {
        // Pushes .gitmodules into the REMOTE, so the local checkout learns of
        // it only by fast-forwarding. A pre-merge read of the flag returns None
        // here; a post-merge read returns Some(true).
        let (root, remote) = live_workspace();
        let other = tempfile::tempdir().unwrap().keep();
        git_sync(
            &other,
            &["clone", remote.to_str().unwrap(), other.to_str().unwrap()],
        );
        fs::write(
            other.join(".gitmodules"),
            "[submodule \"content\"]\n\tpath = content\n\turl = ../content.git\n",
        )
        .unwrap();
        git_sync(&other, &["add", ".gitmodules"]);
        git_sync(&other, &["commit", "-m", "add submodule decl"]);
        git_sync(&other, &["push", "origin", "main"]);

        assert!(!root.join("repos/jianyi/.gitmodules").exists());

        let report = sweep(&root, true).await;
        assert_eq!(repo(&report, "repos/jianyi").status, "synced");
        assert_eq!(repo(&report, "repos/jianyi").submodules, Some(true));
    }

    #[tokio::test]
    async fn a_repo_without_submodules_omits_the_flag_entirely() {
        let (root, _remote) = live_workspace();
        let report = sweep(&root, false).await;
        assert_eq!(repo(&report, "repos/jianyi").submodules, None);
    }

    #[tokio::test]
    async fn the_report_carries_the_gate_state() {
        let (root, _remote) = live_workspace();
        assert!(sweep(&root, false).await.enabled);

        fs::write(
            root.join("modules.local.toml"),
            "[[repos]]\nname = \"jianyi\"\npath = \"repos/jianyi\"\n",
        )
        .unwrap();
        assert!(!sweep(&root, false).await.enabled);
    }
}
