//! The sweep: which repos, what happened to each, and what was found and
//! deliberately not touched.
//!
//! **The repo set is manifest-derived, never discovered.** The same
//! `parse_workspace` call that backs `/composition` produces it, so there is one
//! enumeration rather than two, composing a new repo makes it sync itself, and
//! `repos/external/` stays invisible because it was never declared.
//!
//! Manifest-derivation is correct and *silent*, which is the wrong shape on its
//! own — a sweep that skips an undeclared module without saying so reads
//! identically to a sweep that had nothing to skip. `undeclared_checkouts` is
//! the answer to that, and it paid for itself before it was written: enumerating
//! for this design is what surfaced `operators/ariadne`, on disk since
//! 2026-07-24 and in no manifest.

use std::path::Path;

/// A module the manifest declares AND that exists on disk as a git checkout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredModule {
    pub name: String,
    /// Relative to the workspace root. The router root itself is `"."`.
    pub path: String,
}

/// The directories scanned one level deep for undeclared checkouts.
const MODULE_CONTAINERS: [&str; 2] = ["operators", "repos"];

/// Every composed module that is actually a checkout, router root first.
///
/// A declared module that was never cloned is omitted rather than reported as
/// an error — C-4, the same rule that lets a bare clone be a working host.
/// Presence is tested by `<path>/.git` existing at all, which covers both an
/// ordinary clone (a directory) and a worktree or submodule (a file).
pub fn declared_modules(root: &Path) -> Vec<DeclaredModule> {
    let mut out = Vec::new();

    let is_checkout = |rel: &str| root.join(rel).join(".git").exists();

    // The router root is not a module and is never in the manifest — it is the
    // repo this whole workspace is. Named "armillary" so the report has
    // something to print beside a bare ".".
    if root.join(".git").exists() {
        out.push(DeclaredModule {
            name: "armillary".to_string(),
            path: ".".to_string(),
        });
    }

    let Ok(composition) = armillary_composition::parse_workspace(root) else {
        // A malformed manifest is a warning posture everywhere else in this
        // engine (see main::declared_boot) and is one here too: sweep the root
        // rather than refusing to sweep anything.
        return out;
    };

    for module in composition
        .operators
        .iter()
        .chain(composition.commons.iter())
        .chain(composition.repos.iter())
    {
        if is_checkout(&module.path) {
            out.push(DeclaredModule {
                name: module.name.clone(),
                path: module.path.clone(),
            });
        }
    }

    out
}

/// Git checkouts sitting one level under `operators/` or `repos/` that no
/// manifest declares.
///
/// **Depth 1 only.** `repos/external/` holds reference clones a level deeper,
/// and not descending is what keeps them out of the report — the same promise
/// D2 makes about them never being swept.
pub fn undeclared_checkouts(root: &Path, declared: &[DeclaredModule]) -> Vec<String> {
    let known: std::collections::BTreeSet<&str> =
        declared.iter().map(|m| m.path.as_str()).collect();
    let mut found = Vec::new();

    for container in MODULE_CONTAINERS {
        let Ok(entries) = std::fs::read_dir(root.join(container)) else {
            continue;
        };
        for entry in entries.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(|s| s.to_string()) else {
                continue;
            };
            let rel = format!("{container}/{name}");
            if known.contains(rel.as_str()) {
                continue;
            }
            if root.join(&rel).join(".git").exists() {
                found.push(rel);
            }
        }
    }

    found.sort();
    found
}

/// Whether this workspace has granted the engine authority to run git.
///
/// Read **per request**, not cached in `AppState`. That diverges from
/// `[router] boot` — which is read once at startup because changing WHICH file
/// boots is a restart-level change — and the divergence is deliberate: this
/// costs one small TOML parse on a route that is about to spawn two dozen
/// subprocesses, and it means granting or revoking the authority takes effect
/// when the file is saved rather than when the daemon is next restarted.
///
/// The key rides in `Router.extra`, C-5's escape hatch, so `armillary-composition`
/// needs no new field and `conformance/` is untouched — and a change to
/// `conformance/` is a change to every implementation of the standard.
///
/// **Fails closed everywhere**: absent, misspelled, non-boolean, or an
/// unparseable manifest all mean disabled. The cost is that a typo is silent,
/// which `main.rs`'s startup banner is what pays for.
pub fn gate_enabled(root: &Path) -> bool {
    armillary_composition::parse_workspace(root)
        .ok()
        .and_then(|c| c.router.extra.get("sync").and_then(|v| v.as_bool()))
        .unwrap_or(false)
}

use crate::git::{self, GitError, Verdict};
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
    };

    match verdict {
        Verdict::Behind { commits } if perform => match git::fast_forward(abs, t).await {
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

    /// A workspace whose declared modules exist on disk as git checkouts.
    /// `.git` is created as a plain directory — every function under test asks
    /// only whether it EXISTS, never for its contents, so a real `git init` per
    /// module would cost seconds of test time to prove nothing extra.
    fn workspace() -> PathBuf {
        let root = tempfile::tempdir().unwrap().keep();
        fs::write(
            root.join("modules.toml"),
            "[router]\ncontains = [\"CLAUDE.md\"]\n",
        )
        .unwrap();
        fs::write(
            root.join("modules.local.toml"),
            "[[operators]]\nname = \"tycho\"\npath = \"operators/tycho\"\n\n\
             [[commons]]\nname = \"zojercommons\"\npath = \"zojercommons\"\n\n\
             [[repos]]\nname = \"jianyi\"\npath = \"repos/jianyi\"\n",
        )
        .unwrap();
        for p in ["operators/tycho", "zojercommons", "repos/jianyi"] {
            fs::create_dir_all(root.join(p).join(".git")).unwrap();
        }
        fs::create_dir_all(root.join(".git")).unwrap();
        root
    }

    #[test]
    fn declared_modules_includes_the_router_root_first() {
        let root = workspace();
        let got = declared_modules(&root);
        assert_eq!(got[0].path, ".");
        assert_eq!(got[0].name, "armillary");
    }

    #[test]
    fn declared_modules_reads_operators_commons_and_repos() {
        let root = workspace();
        // Bound to a local first: `.iter()` on the call result directly borrows
        // a temporary that the `let` statement's own type annotation forces to
        // outlive its statement, and the borrow checker rejects that.
        let declared = declared_modules(&root);
        let paths: Vec<&str> = declared.iter().map(|m| m.path.as_str()).collect();
        assert_eq!(paths, vec![".", "operators/tycho", "zojercommons", "repos/jianyi"]);
    }

    #[test]
    fn declared_modules_omits_a_declared_module_that_is_not_cloned() {
        // C-4: a workspace that composes something absent is a working host.
        // A module declared but never cloned is not an error and not a sync
        // target; it is simply not there.
        let root = workspace();
        fs::write(
            root.join("modules.local.toml"),
            "[[repos]]\nname = \"never-cloned\"\npath = \"repos/never-cloned\"\n",
        )
        .unwrap();
        let declared = declared_modules(&root);
        let paths: Vec<&str> = declared.iter().map(|m| m.path.as_str()).collect();
        assert_eq!(paths, vec!["."]);
    }

    #[test]
    fn declared_modules_is_just_the_root_when_nothing_is_composed() {
        let root = tempfile::tempdir().unwrap().keep();
        fs::create_dir_all(root.join(".git")).unwrap();
        assert_eq!(declared_modules(&root).len(), 1);
    }

    #[test]
    fn undeclared_checkouts_finds_a_module_on_disk_and_not_in_the_manifest() {
        let root = workspace();
        fs::create_dir_all(root.join("operators/ariadne").join(".git")).unwrap();
        let declared = declared_modules(&root);
        assert_eq!(undeclared_checkouts(&root, &declared), vec!["operators/ariadne"]);
    }

    #[test]
    fn undeclared_checkouts_is_empty_when_the_manifest_is_complete() {
        let root = workspace();
        let declared = declared_modules(&root);
        assert!(undeclared_checkouts(&root, &declared).is_empty());
    }

    #[test]
    fn undeclared_checkouts_ignores_a_directory_that_is_not_a_checkout() {
        // `operators/blank` is a scratch template, not a clone.
        let root = workspace();
        fs::create_dir_all(root.join("operators/blank")).unwrap();
        let declared = declared_modules(&root);
        assert!(undeclared_checkouts(&root, &declared).is_empty());
    }

    #[test]
    fn undeclared_checkouts_does_not_descend_into_a_container_directory() {
        // `repos/external/` holds reference clones one level deeper. Scanning
        // depth 1 only is what keeps them invisible, matching D2's promise that
        // an undeclared container stays out of the sweep entirely.
        let root = workspace();
        fs::create_dir_all(root.join("repos/external/pi").join(".git")).unwrap();
        let declared = declared_modules(&root);
        assert!(undeclared_checkouts(&root, &declared).is_empty());
    }

    #[test]
    fn gate_is_off_when_nothing_declares_it() {
        let root = workspace();
        assert!(!gate_enabled(&root));
    }

    #[test]
    fn gate_is_on_when_the_router_declares_sync_true() {
        let root = workspace();
        fs::write(
            root.join("modules.local.toml"),
            "[router]\nsync = true\n",
        )
        .unwrap();
        assert!(gate_enabled(&root));
    }

    #[test]
    fn gate_is_off_when_the_key_is_misspelled() {
        // `extra` is unvalidated by design (C-5 forbids deny_unknown_fields),
        // so a typo disables the feature silently. This test exists to make
        // that behaviour deliberate rather than discovered — the startup
        // banner in main.rs is what makes it visible to a human.
        let root = workspace();
        fs::write(root.join("modules.local.toml"), "[router]\nsnyc = true\n").unwrap();
        assert!(!gate_enabled(&root));
    }

    #[test]
    fn gate_is_off_when_the_value_is_not_a_boolean() {
        let root = workspace();
        fs::write(root.join("modules.local.toml"), "[router]\nsync = \"yes\"\n").unwrap();
        assert!(!gate_enabled(&root));
    }

    #[test]
    fn gate_is_off_when_the_manifest_is_malformed() {
        // Fail closed. A broken manifest must not grant an authority.
        let root = workspace();
        fs::write(root.join("modules.local.toml"), "[router\nsync = ").unwrap();
        assert!(!gate_enabled(&root));
    }

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
