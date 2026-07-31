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
}
