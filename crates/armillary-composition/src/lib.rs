//! Parses an armillary workspace manifest into a [`Composition`].
//!
//! Implements `constitution/composition.md` C-1..C-6. This crate has no
//! filesystem access beyond the manifest files it is handed or told to read,
//! and never enumerates a directory — which is what makes C-1 ("declared, not
//! discovered") a structural property rather than a rule someone remembers.

mod manifest;
mod merge;

pub use manifest::{Composition, Module, Protocol, Router};
pub use merge::merge;

use manifest::RawManifest;
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum CompositionError {
    #[error("could not read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not parse {path}: {source}")]
    Toml {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("name collision in [[{section}]]: '{name}' is declared in both manifests")]
    NameCollision {
        section: &'static str,
        name: String,
    },
    /// The same name twice inside ONE manifest. Distinct from `NameCollision`
    /// because the prose differs and so does the fix — and because the C-2
    /// legacy path (`[[models]]` plus `[[operators]]`) produces this shape
    /// without the author writing anything twice.
    #[error("duplicate name in [[{section}]]: '{name}' is declared twice in the same manifest")]
    DuplicateName {
        section: &'static str,
        name: String,
    },
}

/// The machine-readable shape a conformance runner compares against a
/// `*.expected-error.json` fixture. Deliberately narrow: a stable code plus the
/// fields that identify the offending declaration. Prose messages are for
/// humans and are free to change; this is the part a fixture may depend on.
#[derive(Debug, Serialize, PartialEq)]
pub struct ConformanceError {
    pub error: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl CompositionError {
    pub fn as_conformance_error(&self) -> ConformanceError {
        match self {
            CompositionError::NameCollision { section, name } => ConformanceError {
                error: "name_collision",
                section: Some(section),
                name: Some(name.clone()),
            },
            CompositionError::DuplicateName { section, name } => ConformanceError {
                error: "duplicate_name",
                section: Some(section),
                name: Some(name.clone()),
            },
            CompositionError::Toml { .. } => ConformanceError {
                error: "parse_error",
                section: None,
                name: None,
            },
            CompositionError::Io { .. } => ConformanceError {
                error: "io_error",
                section: None,
                name: None,
            },
        }
    }
}

/// Parse a single manifest's text.
///
/// C-3: commented-out entries are not declarations. That falls out of using a
/// TOML parser at all — comments are not data — and it is the whole point:
/// *byte-derived* means a parser instead of a model. The rule exists because a
/// local model once read commented-out examples as a live composition.
pub fn parse_manifest_str(text: &str) -> Result<Composition, CompositionError> {
    let raw: RawManifest = toml::from_str(text).map_err(|source| CompositionError::Toml {
        path: PathBuf::from("<str>"),
        source,
    })?;
    let composition: Composition = raw.into();
    check_internally_unique(&composition)?;
    Ok(composition)
}

/// C-6 within a single manifest. Runs at every entry point, so a duplicate is
/// caught whether the manifest arrives as text, as a file, or as an overlay.
fn check_internally_unique(c: &Composition) -> Result<(), CompositionError> {
    merge::check_unique("operators", &c.operators)?;
    merge::check_unique("commons", &c.commons)?;
    merge::check_unique("repos", &c.repos)?;
    merge::check_unique("protocols", &c.protocols)?;
    Ok(())
}

/// Read a workspace's manifests and produce its composition.
///
/// C-4 (presence-gating): a missing `modules.toml` yields an empty composition,
/// and a missing *or dangling* `modules.local.toml` means "no overlay". Neither
/// is an error — a bare clone is a working host, not a broken one.
pub fn parse_workspace(root: &Path) -> Result<Composition, CompositionError> {
    let base = read_manifest_text(&root.join("modules.toml"))?;
    let overlay = read_manifest_text(&root.join("modules.local.toml"))?;
    compose_manifest_texts(base.as_deref(), overlay.as_deref())
}

/// Read one optional manifest's text.
///
/// C-4: absent is `None`, not an error — and `metadata` follows symlinks, so a
/// dangling link reads as absent too, which is exactly what C-4 wants: the
/// private overlay is normally a symlink into a commons that may not be cloned
/// on this machine. Public so a caller (the engine's `WorkspaceSnapshot`) can
/// hash exactly the bytes this crate parses, instead of reading the file a
/// second time and hoping nothing changed between the reads.
pub fn read_manifest_text(path: &Path) -> Result<Option<String>, CompositionError> {
    if path.metadata().is_err() {
        return Ok(None);
    }
    std::fs::read_to_string(path)
        .map(Some)
        .map_err(|source| CompositionError::Io {
            path: path.to_path_buf(),
            source,
        })
}

/// Compose from already-read manifest texts — `parse_workspace` minus the
/// reads. Pure, so one read can feed both this and a digest.
///
/// A TOML error here carries the `<str>` path (it came from text, not a file);
/// callers who read the file themselves know which one they handed over.
pub fn compose_manifest_texts(
    base: Option<&str>,
    overlay: Option<&str>,
) -> Result<Composition, CompositionError> {
    let base = base.map(parse_manifest_str).transpose()?.unwrap_or_default();
    let overlay = overlay.map(parse_manifest_str).transpose()?.unwrap_or_default();
    merge(base, overlay)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn compose_from_texts_matches_parse_workspace_on_the_same_content() {
        let dir = tempfile::tempdir().unwrap();
        let base = "[[repos]]\nname = \"a\"\npath = \"repos/a\"\n";
        let overlay = "[router]\nboot = \"getting-started.md\"\n";
        fs::write(dir.path().join("modules.toml"), base).unwrap();
        fs::write(dir.path().join("modules.local.toml"), overlay).unwrap();

        let via_files = parse_workspace(dir.path()).unwrap();
        let via_texts = compose_manifest_texts(Some(base), Some(overlay)).unwrap();

        assert_eq!(via_files, via_texts);
    }

    #[test]
    fn read_manifest_text_treats_absent_and_dangling_as_none() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            read_manifest_text(&dir.path().join("modules.toml")).unwrap(),
            None
        );

        let dangling = dir.path().join("modules.local.toml");
        std::os::unix::fs::symlink(dir.path().join("nowhere"), &dangling).unwrap();
        assert_eq!(read_manifest_text(&dangling).unwrap(), None);

        fs::write(dir.path().join("real.toml"), "x = 1").unwrap();
        assert_eq!(
            read_manifest_text(&dir.path().join("real.toml"))
                .unwrap()
                .as_deref(),
            Some("x = 1")
        );
    }

    #[test]
    fn bare_workspace_yields_empty_composition() {
        let dir = tempfile::tempdir().unwrap();
        let c = parse_workspace(dir.path()).expect("a bare clone is a working host");
        assert_eq!(c, Composition::default());
    }

    #[test]
    fn dangling_overlay_symlink_is_treated_as_absent() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("modules.toml"),
            "[[repos]]\nname='a'\npath='repos/a'\n",
        )
        .unwrap();
        std::os::unix::fs::symlink(
            dir.path().join("nowhere/modules.local.toml"),
            dir.path().join("modules.local.toml"),
        )
        .unwrap();

        let c = parse_workspace(dir.path()).expect("a dangling overlay is not an error");
        assert_eq!(c.repos.len(), 1);
    }

    #[test]
    fn unknown_protocol_fields_are_tolerated_and_preserved() {
        // C-5: engines MUST NOT reject manifests carrying unknown protocol
        // fields. `kind` is the specific field the constitution names as
        // deliberately unmodeled, so it is the honest test case.
        let c = parse_manifest_str(
            "[[protocols]]\nname='board'\nsource='x.md'\nload='boot'\nkind='lens'\n",
        )
        .expect("an unknown protocol field must not be an error");
        assert_eq!(c.protocols[0].extra.get("kind").unwrap().as_str(), Some("lens"));
    }

    #[test]
    fn duplicate_within_one_manifest_is_an_error() {
        // C-6 was half-implemented: the cross-file case was fatal while the
        // identical mistake inside one file passed in silence.
        let err = parse_manifest_str(
            "[[operators]]\nname='tycho'\npath='a'\n\n[[operators]]\nname='tycho'\npath='b'\n",
        )
        .expect_err("two declarations of one name in one manifest is ambiguous");
        assert_eq!(err.as_conformance_error().error, "duplicate_name");
    }

    #[test]
    fn the_legacy_migration_path_cannot_boot_an_operator_twice() {
        // The sharp case: nobody writes a name twice. They add [[operators]]
        // during the rename and forget to delete [[models]] — the exact
        // mid-migration state C-2 exists to support — and the compat symlink
        // makes both paths the same directory.
        let err = parse_manifest_str(
            "[[operators]]\nname='tycho'\npath='operators/tycho'\n\n             [[models]]\nname='tycho'\npath='models/tycho'\n",
        )
        .expect_err("legacy plus canonical declaring one operator is ambiguous");
        assert_eq!(err.as_conformance_error().error, "duplicate_name");
    }

    #[test]
    fn distinct_names_across_legacy_sections_still_merge() {
        let c = parse_manifest_str(
            "[[operators]]\nname='tycho'\npath='operators/tycho'\n\n             [[models]]\nname='kepler'\npath='models/kepler'\n",
        )
        .expect("different names are not a duplicate");
        assert_eq!(c.operators.len(), 2);
    }

    #[test]
    fn legacy_sections_normalize_to_operators_keeping_declared_paths() {
        let c = parse_manifest_str("[[models]]\nname='tycho'\npath='models/tycho'\n").unwrap();
        assert_eq!(c.operators.len(), 1);
        // C-2: the *section name* normalizes; the declared path is honored as
        // written, so a workspace mid-migration keeps resolving.
        assert_eq!(c.operators[0].path, "models/tycho");
    }

    #[test]
    fn router_table_parses_contains_and_boot() {
        let text = r#"
[router]
contains = ["CLAUDE.md", "README.md"]
boot = "getting-started.md"
"#;
        let c = parse_manifest_str(text).unwrap();
        assert_eq!(c.router.boot.as_deref(), Some("getting-started.md"));
        assert_eq!(c.router.contains, vec!["CLAUDE.md", "README.md"]);
    }

    #[test]
    fn an_absent_router_table_is_the_default_and_serializes_to_nothing() {
        // C-4: a manifest that declares no router table is a working manifest.
        // The serialization half matters as much as the parse half — every
        // conformance fixture is a serialized Composition, and an always-present
        // `router` key would rewrite all of them.
        let c = parse_manifest_str("").unwrap();
        assert_eq!(c.router.boot, None);
        assert!(c.router.contains.is_empty());
        let json = serde_json::to_string(&c).unwrap();
        assert!(!json.contains("router"), "empty router must not serialize: {json}");
    }

    #[test]
    fn an_unknown_router_key_is_tolerated_not_rejected() {
        // C-5: the shape is provisional. An engine reading a newer manifest must
        // not reject it.
        let c = parse_manifest_str("[router]\nboot = \"a.md\"\nfuture_key = 3\n").unwrap();
        assert_eq!(c.router.boot.as_deref(), Some("a.md"));
        assert!(c.router.extra.contains_key("future_key"));
    }

    #[test]
    fn overlay_router_merges_field_wise_not_wholesale() {
        // A machine-local overlay must be able to set `boot` without restating
        // `contains` — a wholesale replace would silently erase the allowlist.
        let base = parse_manifest_str("[router]\ncontains = [\"CLAUDE.md\"]\n").unwrap();
        let overlay = parse_manifest_str("[router]\nboot = \"getting-started.md\"\n").unwrap();
        let merged = crate::merge(base, overlay).unwrap();
        assert_eq!(merged.router.contains, vec!["CLAUDE.md"]);
        assert_eq!(merged.router.boot.as_deref(), Some("getting-started.md"));
    }

    #[test]
    fn overlay_router_boot_overrides_the_base_boot() {
        let base = parse_manifest_str("[router]\nboot = \"public.md\"\n").unwrap();
        let overlay = parse_manifest_str("[router]\nboot = \"local.md\"\n").unwrap();
        let merged = crate::merge(base, overlay).unwrap();
        assert_eq!(merged.router.boot.as_deref(), Some("local.md"));
    }

    #[test]
    fn overlay_router_contains_does_not_erase_the_base_boot() {
        // The mirror of `overlay_router_merges_field_wise_not_wholesale`. Field-wise
        // is a claim about BOTH fields, and a refactor that made `contains` the
        // trigger for replacing the whole table would pass every other test here.
        let base = parse_manifest_str("[router]\nboot = \"getting-started.md\"\n").unwrap();
        let overlay = parse_manifest_str("[router]\ncontains = [\"CLAUDE.md\"]\n").unwrap();
        let merged = crate::merge(base, overlay).unwrap();
        assert_eq!(merged.router.boot.as_deref(), Some("getting-started.md"));
        assert_eq!(merged.router.contains, vec!["CLAUDE.md"]);
    }

    // ---- B-2: an operator declares its own boot surface ----

    #[test]
    fn an_operator_declares_the_files_that_constitute_its_identity() {
        // Declared, not conventional. A rule like "always load
        // `<path>/CLAUDE.md`" is already wrong in this workspace: ariadne's
        // manifest entry says outright that she has no CLAUDE.md and her boot
        // surface is `self.md`.
        let c = parse_manifest_str(
            "[[operators]]\nname = \"tycho\"\npath = \"operators/tycho\"\n\
             boot = [\"operators/tycho/principles.md\", \"operators/tycho/CLAUDE.md\"]\n",
        )
        .unwrap();

        assert_eq!(
            c.operators[0].boot,
            Some(vec![
                "operators/tycho/principles.md".to_string(),
                "operators/tycho/CLAUDE.md".to_string(),
            ]),
            "order is the declaration — stable-first is the caller's to honour"
        );
    }

    #[test]
    fn an_operator_without_a_declared_boot_is_not_an_error() {
        // C-4 throughout. Most operators will not declare one, and an engine
        // that cannot inject a system prompt ignores the field entirely.
        let c = parse_manifest_str(
            "[[operators]]\nname = \"leavitt\"\npath = \"operators/leavitt\"\n",
        )
        .unwrap();
        assert_eq!(c.operators[0].boot, None);
    }

    #[test]
    fn an_unknown_field_on_a_module_survives_instead_of_vanishing() {
        // C-5, and the gap that made this necessary: `Module` had no `extra`
        // and no `deny_unknown_fields`, so serde's default silently DROPPED
        // anything it did not recognise. Writing `boot = [...]` on an operator
        // before the engine understood it would have parsed clean, done
        // nothing, and said nothing — the exact silent-drop class this
        // codebase keeps closing. `Protocol` and `Router` already carry
        // `extra` for this reason; `Module` did not.
        let c = parse_manifest_str(
            "[[repos]]\nname = \"r\"\npath = \"p\"\nsomething_new = \"kept\"\n",
        )
        .unwrap();

        assert_eq!(
            c.repos[0].extra.get("something_new").and_then(|v| v.as_str()),
            Some("kept")
        );
    }

    #[test]
    fn boot_rides_on_the_operators_single_declaration_because_a_second_one_is_an_error() {
        // C-6: a `name` collision within a section is an ERROR, across the
        // manifest pair as much as within one file. So there is no "declare the
        // operator publicly, add its boot in the overlay" shape — an engine
        // MUST refuse rather than silently override. Boot goes wherever the
        // operator itself is declared, which in a real deployment is the
        // private overlay, where the paths belong anyway.
        let base = parse_manifest_str(
            "[[operators]]\nname = \"tycho\"\npath = \"operators/tycho\"\n",
        )
        .unwrap();
        let overlay = parse_manifest_str(
            "[[operators]]\nname = \"tycho\"\npath = \"operators/tycho\"\n\
             boot = [\"operators/tycho/self.md\"]\n",
        )
        .unwrap();

        assert!(
            matches!(
                crate::merge(base, overlay),
                Err(CompositionError::NameCollision { .. })
            ),
            "silently overriding an operator's boot is exactly what C-6 forbids"
        );
    }

    #[test]
    fn an_operator_declared_only_in_the_overlay_carries_its_boot_through() {
        // The real deployment shape: `modules.toml` ships commented-out
        // examples and `modules.local.toml` carries every real entry.
        let base = parse_manifest_str("[router]\ncontains = [\"CLAUDE.md\"]\n").unwrap();
        let overlay = parse_manifest_str(
            "[[operators]]\nname = \"tycho\"\npath = \"operators/tycho\"\n\
             boot = [\"operators/tycho/self.md\"]\n",
        )
        .unwrap();

        let merged = crate::merge(base, overlay).unwrap();
        assert_eq!(
            merged.operators[0].boot,
            Some(vec!["operators/tycho/self.md".to_string()])
        );
        assert_eq!(merged.router.contains, vec!["CLAUDE.md"]);
    }
}
