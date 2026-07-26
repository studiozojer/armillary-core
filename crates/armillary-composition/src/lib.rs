//! Parses an armillary workspace manifest into a [`Composition`].
//!
//! Implements `constitution/composition.md` C-1..C-6. This crate has no
//! filesystem access beyond the manifest files it is handed or told to read,
//! and never enumerates a directory — which is what makes C-1 ("declared, not
//! discovered") a structural property rather than a rule someone remembers.

mod manifest;
mod merge;

pub use manifest::{Composition, Module, Protocol};
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
    Ok(raw.into())
}

/// Read a workspace's manifests and produce its composition.
///
/// C-4 (presence-gating): a missing `modules.toml` yields an empty composition,
/// and a missing *or dangling* `modules.local.toml` means "no overlay". Neither
/// is an error — a bare clone is a working host, not a broken one.
pub fn parse_workspace(root: &Path) -> Result<Composition, CompositionError> {
    let base = read_optional_manifest(&root.join("modules.toml"))?;
    let overlay = read_optional_manifest(&root.join("modules.local.toml"))?;
    merge(base, overlay)
}

fn read_optional_manifest(path: &Path) -> Result<Composition, CompositionError> {
    // `metadata` follows symlinks, so a dangling link reads as absent — which is
    // exactly what C-4 wants here, because the private overlay is normally a
    // symlink into a commons that may not be cloned on this machine.
    if path.metadata().is_err() {
        return Ok(Composition::default());
    }
    let text = std::fs::read_to_string(path).map_err(|source| CompositionError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let raw: RawManifest = toml::from_str(&text).map_err(|source| CompositionError::Toml {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(raw.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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
    fn legacy_sections_normalize_to_operators_keeping_declared_paths() {
        let c = parse_manifest_str("[[models]]\nname='tycho'\npath='models/tycho'\n").unwrap();
        assert_eq!(c.operators.len(), 1);
        // C-2: the *section name* normalizes; the declared path is honored as
        // written, so a workspace mid-migration keeps resolving.
        assert_eq!(c.operators[0].path, "models/tycho");
    }
}
