//! The repo set: which composed modules are actual git checkouts on disk,
//! the authority gate that permits git to run at all, and each one's live
//! state.
//!
//! **The repo set is manifest-derived, never discovered.** The same
//! `parse_workspace` call that backs `/composition` produces it, so there is
//! one enumeration rather than two, composing a new repo makes it sync
//! itself, and `repos/external/` stays invisible because it was never
//! declared.
//!
//! Manifest-derivation is correct and *silent*, which is the wrong shape on
//! its own — a sweep that skips an undeclared module without saying so reads
//! identically to a sweep that had nothing to skip. `undeclared_checkouts` is
//! the answer to that, and it paid for itself before it was written:
//! enumerating for this design is what surfaced `operators/ariadne`, on disk
//! since 2026-07-24 and in no manifest.
//!
//! `RepoState` and `read_one` are the other half: one repo's full status,
//! read without a fetch and without a mutation — the write verbs live in
//! `git.rs`, and `sync.rs`'s `sweep` is what calls them before asking here.

use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// How many repos are in flight at once.
///
/// Twenty-four serial fetches over a sleeping tailnet is a minute of spinner,
/// and one unreachable remote must not hold up the other twenty-three. Bounded
/// rather than unbounded because two dozen simultaneous SSH handshakes is a
/// different kind of rude.
pub(crate) const CONCURRENCY: usize = 8;

use armillary_composition::Composition;

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
///
/// Takes the request's one parsed `Composition` (a `WorkspaceSnapshot`'s)
/// rather than parsing here: the malformed-manifest posture — sweep the root
/// rather than refuse to sweep anything — now lives at the route boundary,
/// where an unloadable snapshot defaults to nothing composed.
pub fn declared_modules(root: &Path, composition: &Composition) -> Vec<DeclaredModule> {
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
/// **Fails closed everywhere**: absent, misspelled, or non-boolean all mean
/// disabled — and an unparseable manifest reaches here as the route
/// boundary's default (nothing composed), which reads the same way. The cost
/// is that a typo is silent, which `main.rs`'s startup banner is what pays
/// for. The parse now happens once at the route boundary; every answer in a
/// request derives from that one snapshot.
pub fn gate_enabled(composition: &Composition) -> bool {
    composition
        .router
        .extra
        .get("sync")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

use crate::git::{self, GitError, Position};

/// A verb's failure, typed rather than a bare string.
///
/// Replaces `fetch_error: Option<String>` (D-whole-branch-review): a
/// dirty-tree refusal and an unreachable host used to land in the same
/// untyped field, distinguishable only by string-matching the message —
/// exactly the thing Task 3 split `GitError::InvalidArg` out of `Failed` to
/// avoid doing at the `GitError` layer, reintroduced one layer up. `kind` is
/// a small closed vocabulary a client switches on directly: `"dirty"` (a
/// policy refusal — the gate allowed the verb, the working tree did not),
/// `"not-fast-forwardable"` (git refused the merge or push on its own
/// terms — diverged history, no upstream, a non-fast-forward push),
/// `"refused-by-remote"` (the remote deliberately declined a push — a
/// protected branch, a pre-receive hook — distinct from both neighbors: the
/// remote WAS reached, so it isn't `"transport"`, and pulling first will not
/// help, so it isn't `"not-fast-forwardable"`; see
/// `routes::repos::push_action_error`), `"transport"` (the remote could not
/// be reached), `"timeout"` (the invocation exceeded its cap), `"detached"`
/// (a commit was refused because HEAD is detached — it would belong to no
/// branch), `"nothing-to-commit"` (a commit was refused because the working
/// tree is clean), `"commit-failed"` (`git add`/`git commit` itself failed —
/// a declining pre-commit hook, most commonly, whose own text becomes
/// `message` since git gives it no stable locale-pinned marker the way
/// `"[rejected]"` marks a push). `message` is the human-readable detail for
/// display; `kind` is what code branches on.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ActionError {
    pub kind: &'static str,
    pub message: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RepoState {
    pub name: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    pub position: Position,
    pub dirty_files: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_fetch: Option<String>,
    /// Single-repo routes only (D5) — see `read_one`'s `with_commit`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub newest_commit: Option<String>,
    pub worktrees: u32,
    pub submodules: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action_error: Option<ActionError>,
    /// Set when `git status --porcelain=v2` itself failed to run. **Not**
    /// "every field above is a default" — `last_fetch`, `worktrees`, and
    /// `submodules` are all filesystem reads that happen BEFORE `status_v2`
    /// in `read_one`, so they are real measurements even on this path. Only
    /// `head`, `branch`, `position` (stuck at its `Detached` default), and
    /// `dirty_files` (stuck at `0`) are type defaults when this is set —
    /// `status_v2` is the only source for those four. `newest_commit` is
    /// also absent here, but that is `with_commit`'s ordinary meaning (never
    /// attempted), not this field's.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_error: Option<String>,
}

/// Read one repo's full state. Performs no network call and no mutation —
/// the verbs in `git.rs` act first (fetch, pull, push), and callers reach
/// this afterward to build the response.
///
/// `with_commit` gates the second fork for `newest_commit` (D5): `false` on
/// the list route, where the cost multiplies by twenty-four composed repos —
/// the entire savings this design exists to bank — and `true` on the
/// single-repo page, where one extra fork is unmeasurable.
pub async fn read_one(root: &Path, module: &DeclaredModule, with_commit: bool) -> RepoState {
    let abs = root.join(&module.path);
    let t = git::DEFAULT_TIMEOUT;

    let mut state = RepoState {
        name: module.name.clone(),
        path: module.path.clone(),
        head: None,
        branch: None,
        position: Position::Detached,
        dirty_files: 0,
        last_fetch: git::last_fetch(&abs),
        newest_commit: None,
        worktrees: git::worktree_count(&abs),
        submodules: git::has_submodules(&abs),
        action_error: None,
        read_error: None,
    };

    match git::status_v2(&abs, t).await {
        Ok(s) => {
            state.head = s.head;
            state.branch = s.branch;
            state.position = s.position;
            state.dirty_files = s.dirty_files;
        }
        Err(e) => {
            state.read_error = Some(match e {
                GitError::Timeout => "timed out".to_string(),
                GitError::Failed(msg) => msg,
                // Unreachable in practice: `status_v2` passes only literal
                // args, never anything request-derived. Carried like `Failed`
                // rather than panicking or dropping it — the same fold
                // `fetch_action_error` below (and each verb's own
                // `routes::repos::*_action_error`) applies to the identical
                // variant (distinct from that module's `git_error_response`,
                // which maps `InvalidArg` to 400 for a READ route's own
                // failure, not an ACT step's incidental one).
                GitError::InvalidArg(msg) => msg,
            });
            return state;
        }
    }

    if with_commit {
        // Read LAST and only here: a caller that mutated must see the commit
        // its mutation landed, not the one it already had.
        state.newest_commit = git::newest_commit(&abs, t).await.ok().flatten();
    }
    state
}

/// A manifest name to the module it names.
///
/// **The only resolver, and the reason no path from a request ever reaches
/// the filesystem.** `declared_modules` already enumerates exactly what may
/// be acted on; a name is a KEY into that set, so `../../../etc` is not
/// sanitised — it simply is not a key, and the miss happens before any path
/// is constructed.
///
/// This is strictly stronger than cleaning a string, and it is why the routes
/// take a name rather than a path. `guard.rs`'s header records what the other
/// approach costs to get right: it once validated the string the client sent
/// and then opened whatever that resolved to, producing two independent
/// one-request bypasses.
pub fn resolve(root: &Path, composition: &Composition, name: &str) -> Option<DeclaredModule> {
    declared_modules(root, composition)
        .into_iter()
        .find(|m| m.name == name)
}

/// Whether this workspace has granted the engine authority to PUBLISH.
///
/// Separate from `gate_enabled` (`sync`) on purpose (D7): pull lets an
/// enrolled device make a host fetch; push lets it make a host publish, under
/// the host user's own credential, with no undo. Same fail-closed posture —
/// absent, misspelled, non-boolean or unparseable all mean disabled.
pub fn push_enabled(composition: &Composition) -> bool {
    composition
        .router
        .extra
        .get("push")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Whether this workspace has granted the engine authority to COMMIT.
///
/// Separate from `push_enabled` in both directions (design D2): push without
/// commit is today's world — publish what already exists; commit without push
/// is a device that authors locally but cannot publish. Same fail-closed
/// posture — absent, misspelled, non-boolean or unparseable all mean disabled.
pub fn commit_enabled(composition: &Composition) -> bool {
    composition
        .router
        .extra
        .get("commit")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Every composed repo's state, manifest order, no network call.
pub async fn list(root: &Path, composition: &Composition) -> Vec<RepoState> {
    let mut out = Vec::new();
    for module in declared_modules(root, composition) {
        out.push(read_one(root, &module, false).await);
    }
    out
}

/// A `GitError` from `git::fetch`, folded to the typed `ActionError` this
/// design's `RepoState::action_error` carries. Shared by the group fetch
/// below and the single-repo fetch route (`routes::repos::fetch_one`) so the
/// two verbs classify identically.
///
/// `fetch` always talks to a remote — its only failure modes are "could not
/// reach it" and "took too long trying" — so every non-timeout failure is
/// `"transport"`, never `"not-fast-forwardable"` or `"dirty"` (those belong
/// to `pull`/`push`, which act on the working tree or a ref this repo owns
/// rather than merely asking a remote what it has).
pub fn fetch_action_error(e: GitError) -> ActionError {
    match e {
        GitError::Timeout => ActionError { kind: "timeout", message: "timed out".to_string() },
        GitError::Failed(msg) => ActionError { kind: "transport", message: msg },
        // Unreachable in practice: no request value reaches `fetch`. Carried
        // like `Failed` rather than panicking or dropping it.
        GitError::InvalidArg(msg) => ActionError { kind: "transport", message: msg },
    }
}

/// Fetch every composed repo, bounded at CONCURRENCY, then read each state.
///
/// Group FETCH only. Group pull and group push are deliberately absent (design
/// §5, David 2026-08-04): fetch touches no working tree and no branch, so
/// widening it across twenty-four repos carries no risk, and it is what makes
/// the list's counts true rather than a reading of whatever the last fetch
/// left behind. The multi-verb form waits for the per-repo verbs to prove
/// themselves.
///
/// **Never returns `Err`.** A repo that fails is a line in the result carrying
/// its own error — twenty-three good answers and one failure is the useful
/// outcome, and a 500 would throw all of it away for one bad repo.
pub async fn fetch_all(root: &Path, composition: &Composition) -> Vec<RepoState> {
    let permits = Arc::new(Semaphore::new(CONCURRENCY));
    let mut handles = Vec::new();
    let mut labels = Vec::new();

    for module in declared_modules(root, composition) {
        labels.push(module.clone());
        let permits = permits.clone();
        let root = root.to_path_buf();
        handles.push(tokio::spawn(async move {
            // The semaphore is never closed, so acquire cannot fail.
            let _permit = permits.acquire().await.expect("semaphore is open");
            let abs = root.join(&module.path);
            let action_error =
                git::fetch(&abs, git::DEFAULT_TIMEOUT).await.err().map(fetch_action_error);
            let mut state = read_one(&root, &module, false).await;
            state.action_error = action_error;
            state
        }));
    }

    // A panicked or cancelled task must not disappear from the result: the
    // contract is that a failing repo is a ROW, and a silently absent one
    // reads as "not composed" — the exact confusion this whole design exists
    // to prevent.
    //
    // Unreachable by construction today: the spawned future has no panic
    // sites (`git::fetch` and `read_one` both return values, never unwind),
    // and the semaphore is never closed, so `JoinError` cannot occur in
    // practice. This carries NO test rather than a faked one, the same
    // reasoning `git.rs`'s `run_git` states for `.stdin(Stdio::null())` —
    // inducing a real panic here to "prove" the arm would be exactly the
    // manufactured coverage that precedent refuses.
    let mut out = Vec::with_capacity(handles.len());
    for (module, handle) in labels.into_iter().zip(handles) {
        out.push(handle.await.unwrap_or_else(|_| RepoState {
            name: module.name,
            path: module.path,
            head: None,
            branch: None,
            position: Position::Detached,
            dirty_files: 0,
            last_fetch: None,
            newest_commit: None,
            worktrees: 0,
            submodules: false,
            action_error: None,
            read_error: Some("task-failed".to_string()),
        }));
    }
    out
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

    /// The old signatures parsed internally; these tests keep their exact
    /// pre-snapshot behavior by parsing the fixture the same way the route
    /// boundary now does (unparseable -> nothing composed).
    fn comp(root: &Path) -> Composition {
        armillary_composition::parse_workspace(root).unwrap_or_default()
    }

    #[test]
    fn declared_modules_includes_the_router_root_first() {
        let root = workspace();
        let got = declared_modules(&root, &comp(&root));
        assert_eq!(got[0].path, ".");
        assert_eq!(got[0].name, "armillary");
    }

    #[test]
    fn declared_modules_reads_operators_commons_and_repos() {
        let root = workspace();
        // Bound to a local first: `.iter()` on the call result directly borrows
        // a temporary that the `let` statement's own type annotation forces to
        // outlive its statement, and the borrow checker rejects that.
        let declared = declared_modules(&root, &comp(&root));
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
        let declared = declared_modules(&root, &comp(&root));
        let paths: Vec<&str> = declared.iter().map(|m| m.path.as_str()).collect();
        assert_eq!(paths, vec!["."]);
    }

    #[test]
    fn declared_modules_is_just_the_root_when_nothing_is_composed() {
        let root = tempfile::tempdir().unwrap().keep();
        fs::create_dir_all(root.join(".git")).unwrap();
        assert_eq!(declared_modules(&root, &comp(&root)).len(), 1);
    }

    #[test]
    fn undeclared_checkouts_finds_a_module_on_disk_and_not_in_the_manifest() {
        let root = workspace();
        fs::create_dir_all(root.join("operators/ariadne").join(".git")).unwrap();
        let declared = declared_modules(&root, &comp(&root));
        assert_eq!(undeclared_checkouts(&root, &declared), vec!["operators/ariadne"]);
    }

    #[test]
    fn undeclared_checkouts_is_empty_when_the_manifest_is_complete() {
        let root = workspace();
        let declared = declared_modules(&root, &comp(&root));
        assert!(undeclared_checkouts(&root, &declared).is_empty());
    }

    #[test]
    fn undeclared_checkouts_ignores_a_directory_that_is_not_a_checkout() {
        // `operators/blank` is a scratch template, not a clone.
        let root = workspace();
        fs::create_dir_all(root.join("operators/blank")).unwrap();
        let declared = declared_modules(&root, &comp(&root));
        assert!(undeclared_checkouts(&root, &declared).is_empty());
    }

    #[test]
    fn undeclared_checkouts_does_not_descend_into_a_container_directory() {
        // `repos/external/` holds reference clones one level deeper. Scanning
        // depth 1 only is what keeps them invisible, matching D2's promise that
        // an undeclared container stays out of the sweep entirely.
        let root = workspace();
        fs::create_dir_all(root.join("repos/external/pi").join(".git")).unwrap();
        let declared = declared_modules(&root, &comp(&root));
        assert!(undeclared_checkouts(&root, &declared).is_empty());
    }

    #[test]
    fn gate_is_off_when_nothing_declares_it() {
        let root = workspace();
        assert!(!gate_enabled(&comp(&root)));
    }

    #[test]
    fn gate_is_on_when_the_router_declares_sync_true() {
        let root = workspace();
        fs::write(
            root.join("modules.local.toml"),
            "[router]\nsync = true\n",
        )
        .unwrap();
        assert!(gate_enabled(&comp(&root)));
    }

    #[test]
    fn gate_is_off_when_the_key_is_misspelled() {
        // `extra` is unvalidated by design (C-5 forbids deny_unknown_fields),
        // so a typo disables the feature silently. This test exists to make
        // that behaviour deliberate rather than discovered — the startup
        // banner in main.rs is what makes it visible to a human.
        let root = workspace();
        fs::write(root.join("modules.local.toml"), "[router]\nsnyc = true\n").unwrap();
        assert!(!gate_enabled(&comp(&root)));
    }

    #[test]
    fn gate_is_off_when_the_value_is_not_a_boolean() {
        let root = workspace();
        fs::write(root.join("modules.local.toml"), "[router]\nsync = \"yes\"\n").unwrap();
        assert!(!gate_enabled(&comp(&root)));
    }

    #[test]
    fn gate_is_off_when_the_manifest_is_malformed() {
        // Fail closed. A broken manifest must not grant an authority.
        let root = workspace();
        fs::write(root.join("modules.local.toml"), "[router\nsync = ").unwrap();
        assert!(!gate_enabled(&comp(&root)));
    }

    #[test]
    fn resolve_finds_a_declared_module_by_name() {
        let root = workspace();
        assert_eq!(resolve(&root, &comp(&root), "tycho").unwrap().path, "operators/tycho");
        assert_eq!(resolve(&root, &comp(&root), "armillary").unwrap().path, ".");
    }

    #[test]
    fn resolve_refuses_a_traversal_without_constructing_a_path() {
        // D3: the name is a KEY into the manifest, never a path fragment. These
        // are not sanitised — they simply are not in the map, which is why no
        // `..`-stripping exists anywhere in this module.
        let root = workspace();
        for name in ["../../../etc", "..", "/etc/passwd", "operators/tycho", "nope"] {
            assert!(resolve(&root, &comp(&root), name).is_none(), "{name:?} must not resolve");
        }
    }

    #[test]
    fn resolve_is_case_sensitive_and_exact() {
        let root = workspace();
        assert!(resolve(&root, &comp(&root), "Tycho").is_none());
        assert!(resolve(&root, &comp(&root), "tycho ").is_none());
    }

    #[test]
    fn push_and_sync_are_independent_gates() {
        let root = workspace();
        std::fs::write(root.join("modules.local.toml"), "[router]\nsync = true\n").unwrap();
        assert!(gate_enabled(&comp(&root)));
        assert!(!push_enabled(&comp(&root)), "sync must not grant push");

        std::fs::write(root.join("modules.local.toml"), "[router]\npush = true\n").unwrap();
        assert!(push_enabled(&comp(&root)));
    }

    #[test]
    fn push_gate_fails_closed_on_every_malformed_value() {
        let root = workspace();
        for body in ["[router]\npsuh = true\n", "[router]\npush = \"yes\"\n", "[router\npush = "] {
            std::fs::write(root.join("modules.local.toml"), body).unwrap();
            assert!(!push_enabled(&comp(&root)), "must fail closed on {body:?}");
        }
    }

    #[test]
    fn commit_gate_defaults_closed_and_reads_true() {
        let root = workspace();
        assert!(!commit_enabled(&comp(&root)), "absent must fail closed");

        std::fs::write(root.join("modules.local.toml"), "[router]\ncommit = true\n").unwrap();
        assert!(commit_enabled(&comp(&root)));

        std::fs::write(root.join("modules.local.toml"), "[router]\ncommit = \"yes\"\n").unwrap();
        assert!(!commit_enabled(&comp(&root)), "non-boolean must fail closed");
    }

    use crate::testgit::{commit, git_sync, remote_and_clone};

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

    fn module(name: &str, path: &str) -> DeclaredModule {
        DeclaredModule { name: name.to_string(), path: path.to_string() }
    }

    #[tokio::test]
    async fn ahead_survives_to_the_state() {
        // THE regression test for the founding bug. Under the shipped
        // single-verdict design this repo reported `current`: ahead was
        // parsed, used once for the diverged test, and thrown away with no
        // field to carry it.
        let (root, _remote) = live_workspace();
        commit(&root.join("repos/jianyi"), "mine.md", "unpushed");

        let s = read_one(&root, &module("jianyi", "repos/jianyi"), false).await;
        match s.position {
            Position::Tracking { ahead, behind, .. } => {
                assert_eq!(ahead, 1, "unpushed work must reach the wire");
                assert_eq!(behind, 0);
            }
            other => panic!("expected Tracking, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn dirty_and_ahead_are_both_present_at_once() {
        // The collapse this design exists to end: under a precedence ladder
        // one of these erased the other.
        let (root, _remote) = live_workspace();
        let repo = root.join("repos/jianyi");
        commit(&repo, "mine.md", "unpushed");
        std::fs::write(repo.join("seed.md"), "edited").unwrap();

        let s = read_one(&root, &module("jianyi", "repos/jianyi"), false).await;
        assert_eq!(s.dirty_files, 1);
        assert!(matches!(s.position, Position::Tracking { ahead: 1, .. }));
    }

    #[tokio::test]
    async fn newest_commit_is_absent_on_a_list_read_and_present_on_a_page_read() {
        // D5: the list shows last-FETCH, so it must not pay a second fork per
        // repo.
        let (root, _remote) = live_workspace();
        let m = module("jianyi", "repos/jianyi");
        assert_eq!(read_one(&root, &m, false).await.newest_commit, None);
        assert!(read_one(&root, &m, true).await.newest_commit.is_some());
    }

    #[tokio::test]
    async fn an_unreadable_repo_reports_read_error_and_not_a_clean_state() {
        let root = tempfile::tempdir().unwrap().keep();
        fs::create_dir_all(root.join("repos/ghost")).unwrap();
        let s = read_one(&root, &module("ghost", "repos/ghost"), false).await;
        assert!(s.read_error.is_some(), "a non-repo must not read as clean");
        assert_eq!(s.dirty_files, 0);
        assert_eq!(s.branch, None);
    }

    #[tokio::test]
    async fn fetch_all_preserves_manifest_order_not_completion_order() {
        // "gamma" is declared FIRST but has no remote at all, so its fetch
        // fails and returns near-instantly — the fastest possible completion
        // of the three. "alpha" and "beta" are ordinary clones with real
        // remotes and sort alphabetically BEFORE "gamma", so neither
        // "fastest first" nor "alphabetical" matches the declared order —
        // only manifest order does, and the implementation awaits each
        // spawned handle in the order it was labeled, never by whichever
        // finishes first.
        let root = tempfile::tempdir().unwrap().keep();
        git_sync(&root, &["init", "--initial-branch=main", "."]);
        fs::create_dir_all(root.join("repos")).unwrap();

        git_sync(
            &root,
            &["init", "--initial-branch=main", root.join("repos/gamma").to_str().unwrap()],
        );

        let (remote_a, _seed_a) = remote_and_clone();
        git_sync(
            &root,
            &["clone", remote_a.to_str().unwrap(), root.join("repos/alpha").to_str().unwrap()],
        );

        let (remote_b, _seed_b) = remote_and_clone();
        git_sync(
            &root,
            &["clone", remote_b.to_str().unwrap(), root.join("repos/beta").to_str().unwrap()],
        );

        fs::write(
            root.join("modules.local.toml"),
            "[router]\nsync = true\n\n\
             [[repos]]\nname = \"gamma\"\npath = \"repos/gamma\"\n\n\
             [[repos]]\nname = \"alpha\"\npath = \"repos/alpha\"\n\n\
             [[repos]]\nname = \"beta\"\npath = \"repos/beta\"\n",
        )
        .unwrap();

        let out = fetch_all(&root, &comp(&root)).await;
        let names: Vec<&str> = out.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["armillary", "gamma", "alpha", "beta"]);
    }

    #[tokio::test]
    async fn fetch_all_propagates_a_fetch_failure_into_the_row() {
        let (root, _remote) = live_workspace();
        // Strip the remote entirely so `git::fetch` fails rather than
        // succeeding against nothing.
        git_sync(&root.join("repos/jianyi"), &["remote", "remove", "origin"]);

        let out = fetch_all(&root, &comp(&root)).await;
        let jianyi = out.iter().find(|s| s.path == "repos/jianyi").unwrap();
        let err = jianyi.action_error.as_ref().expect("a repo with no remote must carry its fetch error");
        assert_eq!(err.kind, "transport", "fetch's own failures are always transport, never dirty");
    }

    #[tokio::test]
    async fn fetch_all_reports_every_repo_even_when_all_fetches_fail() {
        // The "twenty-three good answers and one failure" contract, pushed to
        // its edge: zero good answers must still be a full-length `Vec`, not
        // a shortened one and not a panic.
        let (root, _remote) = live_workspace();
        // `live_workspace`'s router root (from its own bare `git init`) has no
        // remote either, so both declared checkouts fail their fetch once
        // jianyi's remote is stripped too.
        git_sync(&root.join("repos/jianyi"), &["remote", "remove", "origin"]);
        let declared = declared_modules(&root, &comp(&root));

        let out = fetch_all(&root, &comp(&root)).await;
        assert_eq!(out.len(), declared.len(), "a failing fetch must not shrink the result");
        assert!(
            out.iter().all(|s| s.action_error.is_some()),
            "every row must carry its own error rather than vanish"
        );
    }

    #[tokio::test]
    async fn list_never_pays_the_newest_commit_fork() {
        // D5: the list route must not carry `newest_commit` — the guarantee
        // `with_commit: false` makes to every caller of `read_one` beneath it.
        let (root, _remote) = live_workspace();
        let out = list(&root, &comp(&root)).await;
        assert!(
            out.iter().all(|s| s.newest_commit.is_none()),
            "list must never pay the second fork per repo"
        );
    }
}
