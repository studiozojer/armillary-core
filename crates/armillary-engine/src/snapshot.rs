//! The workspace as one request sees it.
//!
//! `parse_workspace` re-reads the manifests at every call, which is the right
//! freshness rule (gates take effect on save) applied at the wrong grain: a
//! request that parses at the gate, again at the act, and again at the re-read
//! can see three different workspaces, and a hash computed from a separate
//! read of a file that may have changed cannot honestly back an event
//! claiming "these bytes were in this window" (sprint-1's debt, the survey's
//! seam #3). A snapshot is loaded once per request and answers everything.
//!
//! **Deliberately not a cache.** Sprint-1 suggested `ArcSwap`; retired: two
//! small TOML parses cost microseconds, and invalidation machinery would risk
//! the take-effect-on-save semantics `repos::gate_enabled` documents as the
//! point. If `/composition`-per-turn or pollers ever measure hot, this struct
//! is what an `ArcSwap` would hold — the seam exists, the machinery does not.

use armillary_composition::{Composition, CompositionError};
use std::path::Path;

/// One manifest as it stood at the snapshot's single read.
#[derive(Debug, Clone)]
pub struct ManifestDigest {
    /// An entry of `tools::MANIFEST_FILES`.
    pub path: &'static str,
    /// Over the same bytes the parse consumed.
    pub sha256: String,
}

/// The workspace as one request sees it: parsed once, hashed once,
/// internally coherent by construction.
#[derive(Debug, Clone, Default)]
pub struct WorkspaceSnapshot {
    pub composition: Composition,
    /// Present files only, in `MANIFEST_FILES` order.
    pub manifests: Vec<ManifestDigest>,
}

impl WorkspaceSnapshot {
    /// One read per manifest: the composition is parsed from, and the digest
    /// computed over, the SAME text. There is no second read to drift — the
    /// honesty the DD-1 event claims is structural here, not disciplined.
    pub fn load(root: &Path) -> Result<WorkspaceSnapshot, CompositionError> {
        // Destructured, not indexed: a third entry in MANIFEST_FILES must be
        // a compile error here, not a silently wrong overlay pairing.
        let [base_name, overlay_name] = crate::tools::MANIFEST_FILES;
        let base = armillary_composition::read_manifest_text(&root.join(base_name))?;
        let overlay = armillary_composition::read_manifest_text(&root.join(overlay_name))?;

        let composition =
            armillary_composition::compose_manifest_texts(base.as_deref(), overlay.as_deref())?;
        let manifests = [(base_name, &base), (overlay_name, &overlay)]
            .into_iter()
            .filter_map(|(name, text)| {
                text.as_ref().map(|t| ManifestDigest {
                    path: name,
                    sha256: crate::hash::sha256_hex(t.as_bytes()),
                })
            })
            .collect();
        Ok(WorkspaceSnapshot {
            composition,
            manifests,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_snapshot_parses_and_hashes_the_same_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let text = "[[repos]]\nname = \"a\"\npath = \"repos/a\"\n";
        std::fs::write(dir.path().join("modules.toml"), text).unwrap();

        let snap = WorkspaceSnapshot::load(dir.path()).unwrap();

        assert_eq!(snap.composition.repos[0].name, "a");
        assert_eq!(snap.manifests.len(), 1);
        assert_eq!(snap.manifests[0].path, "modules.toml");
        assert_eq!(
            snap.manifests[0].sha256,
            crate::hash::sha256_hex(text.as_bytes())
        );
    }

    #[test]
    fn a_bare_clone_snapshots_to_nothing_composed() {
        let dir = tempfile::tempdir().unwrap();
        let snap = WorkspaceSnapshot::load(dir.path()).unwrap();
        assert!(snap.manifests.is_empty());
        assert!(snap.composition.repos.is_empty());
    }

    #[test]
    fn a_malformed_manifest_is_a_load_error_not_a_default() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("modules.toml"), "not [ toml").unwrap();
        assert!(WorkspaceSnapshot::load(dir.path()).is_err());
    }
}
