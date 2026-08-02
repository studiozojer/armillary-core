//! Finding things — the two verbs that answer "where is it" rather than
//! "what does it say".
//!
//! **The domain is declared, not discovered** (SD-2). Roots come from the
//! manifest, so `repos/external/` — 69% of this workspace's files, five other
//! harnesses' source — is excluded by nobody declaring it rather than by an
//! exclusion rule someone maintains. That makes this the first substrate verb
//! whose reach is decided by the composition.

use crate::tools::ToolError;
use std::path::{Path, PathBuf};

/// The directories and files a default-domain search walks.
///
/// **C-1 as running code:** declared, not discovered. Nothing here enumerates
/// a directory to find out what a workspace contains — it asks the same parser
/// `get_composition` asks, so the two cannot drift into disagreeing about what
/// this workspace is.
///
/// Each declared path goes through the guard, which canonicalises it and
/// judges the canonical result. A path that escapes, is denied, or is simply
/// not on this machine is **skipped rather than fatal** (C-4): a manifest
/// naming a repo that was never cloned here is the ordinary case.
pub(crate) fn search_roots(root: &Path) -> Result<Vec<PathBuf>, ToolError> {
    let composition = armillary_composition::parse_workspace(root)
        .map_err(|e| ToolError::new("composition_unreadable", e.to_string()))?;

    let mut out: Vec<PathBuf> = Vec::new();

    for m in composition
        .operators
        .iter()
        .chain(&composition.commons)
        .chain(&composition.repos)
    {
        push_root(root, &m.path, &mut out);
    }
    for f in &composition.router.contains {
        push_root(root, f, &mut out);
    }

    Ok(out)
}

/// Resolve one declared path and add it if it survives the guard.
///
/// A free function rather than a closure over `out` on purpose: a closure
/// holding `&mut out` across both loops and then returning `out` is a
/// borrow-checker argument nobody needs to have.
fn push_root(root: &Path, rel: &str, out: &mut Vec<PathBuf>) {
    if let Ok(p) = crate::guard::resolve(root, rel) {
        // Canonical, so two declarations reaching the same directory through
        // different spellings collapse to one root rather than producing
        // every match twice.
        if !out.contains(&p) {
            out.push(p);
        }
    }
}

/// The domain for one call: the composed roots, or the one path the caller named.
///
/// **SD-3.** An explicit `path` overrides the default domain and reaches
/// anywhere `read_file` can, which is what makes SD-2's narrowing safe — a
/// file you can read but not search would otherwise be a trap.
///
/// An **empty** `path` is treated as absent, deliberately. `list_directory`
/// uses `""` for the workspace root, so a model has learned that spelling and
/// will send it here meaning "no scope"; resolving it to the root would
/// silently search every external clone — the opposite of the request.
pub(crate) fn resolve_domain(
    root: &Path,
    path: Option<&str>,
) -> Result<(Vec<PathBuf>, bool), ToolError> {
    match path {
        Some(p) if !p.is_empty() => Ok((vec![crate::guard::resolve(root, p)?], true)),
        _ => Ok((search_roots(root)?, false)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A workspace shaped like the real one: two declared modules, one
    /// declared-but-absent module, and one directory that exists and is
    /// declared by nothing.
    fn workspace() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("modules.toml"),
            "[router]\ncontains = [\"CLAUDE.md\"]\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("modules.local.toml"),
            "[[operators]]\nname = \"tycho\"\npath = \"operators/tycho\"\n\n\
             [[repos]]\nname = \"engine\"\npath = \"repos/engine\"\n\n\
             [[repos]]\nname = \"never-cloned\"\npath = \"repos/never-cloned\"\n",
        )
        .unwrap();
        fs::write(dir.path().join("CLAUDE.md"), "# router\n").unwrap();

        fs::create_dir_all(dir.path().join("operators/tycho")).unwrap();
        fs::write(dir.path().join("operators/tycho/self.md"), "needle\n").unwrap();
        fs::create_dir_all(dir.path().join("repos/engine")).unwrap();
        fs::write(dir.path().join("repos/engine/lib.rs"), "// needle\n").unwrap();

        // Declared by nothing. This is `repos/external/` in miniature.
        fs::create_dir_all(dir.path().join("repos/external/opencode")).unwrap();
        fs::write(dir.path().join("repos/external/opencode/x.ts"), "needle\n").unwrap();

        dir
    }

    fn rels(root: &Path, roots: &[PathBuf]) -> Vec<String> {
        let canon = root.canonicalize().unwrap();
        let mut out: Vec<String> = roots
            .iter()
            .map(|p| p.strip_prefix(&canon).unwrap().to_string_lossy().to_string())
            .collect();
        out.sort();
        out
    }

    #[test]
    fn roots_are_the_declared_modules_and_the_router_files() {
        let dir = workspace();
        let roots = search_roots(dir.path()).unwrap();

        assert_eq!(
            rels(dir.path(), &roots),
            vec!["CLAUDE.md", "operators/tycho", "repos/engine"]
        );
    }

    #[test]
    fn a_declared_root_that_is_not_on_disk_is_skipped_not_an_error() {
        // C-4: presence-gated throughout. A manifest naming a repo this
        // machine has never cloned is the normal case, not a malformed
        // workspace.
        let dir = workspace();
        let roots = search_roots(dir.path()).expect("an absent module must not fail enumeration");

        assert!(!rels(dir.path(), &roots).iter().any(|r| r.contains("never-cloned")));
    }

    #[test]
    fn undeclared_content_is_not_a_root_and_declaring_it_makes_it_one() {
        // MUTATION-CHECKED, and the mutation is the whole test: asserting
        // only that `repos/external` is absent would pass identically if
        // `search_roots` returned an empty vec, or if the domain logic did
        // not exist at all. Declaring it and watching it appear is what
        // proves the exclusion is a consequence of the manifest.
        let dir = workspace();
        assert!(
            !rels(dir.path(), &search_roots(dir.path()).unwrap())
                .iter()
                .any(|r| r.contains("external")),
            "undeclared content must not be a search root"
        );

        fs::write(
            dir.path().join("modules.local.toml"),
            "[[operators]]\nname = \"tycho\"\npath = \"operators/tycho\"\n\n\
             [[repos]]\nname = \"engine\"\npath = \"repos/engine\"\n\n\
             [[repos]]\nname = \"opencode\"\npath = \"repos/external/opencode\"\n",
        )
        .unwrap();

        assert!(
            rels(dir.path(), &search_roots(dir.path()).unwrap())
                .iter()
                .any(|r| r == "repos/external/opencode"),
            "a declared module must become a search root"
        );
    }

    #[test]
    fn an_empty_path_argument_means_the_default_domain_not_the_whole_disk() {
        // `list_directory` uses "" for the workspace root, so a model has
        // learned that idiom and will send it here meaning "no scope". If ""
        // resolved to the root, it would silently defeat SD-2 and search
        // every external clone — the opposite of what the model asked for.
        let dir = workspace();
        let (roots, scoped) = resolve_domain(dir.path(), Some("")).unwrap();

        assert!(!scoped, "\"\" must not count as an explicit scope");
        assert_eq!(
            rels(dir.path(), &roots),
            vec!["CLAUDE.md", "operators/tycho", "repos/engine"]
        );
    }

    #[test]
    fn an_explicit_path_reaches_content_the_default_domain_excludes() {
        // SD-3, David's constraint: reachable when you are specifically
        // looking at it.
        let dir = workspace();
        let (roots, scoped) = resolve_domain(dir.path(), Some("repos/external/opencode")).unwrap();

        assert!(scoped);
        assert_eq!(rels(dir.path(), &roots), vec!["repos/external/opencode"]);
    }

    #[test]
    fn an_explicit_path_still_passes_the_guard() {
        let dir = workspace();
        for (path, status) in [("../escape", "outside_workspace"), ("nope", "not_found")] {
            let err = resolve_domain(dir.path(), Some(path)).unwrap_err();
            assert_eq!(err.status, status, "{path}");
        }
    }
}
