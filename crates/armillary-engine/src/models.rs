//! The host's model catalog — what THIS machine offers to pilot with.
//!
//! Deliberately not a composition surface. `modules.toml` declares what is
//! composed (operators, repos, protocols), and this workspace's vocabulary
//! holds that operators are MODEL-AGNOSTIC — the model is what pilots them
//! in a given session. So the catalog lives beside the credentials it
//! depends on (`~/.config/armillary/{anthropic-key,zen-key}`), one ritual,
//! and the standard stays out of it.
//!
//! Read on demand, never cached at boot: per request for `GET /models` and
//! per turn when resolving a default, so editing the file takes effect
//! everywhere it is consulted at once and the endpoint can never report a
//! default the resolver disagrees with.

use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Deserialize, PartialEq)]
pub struct Catalog {
    #[serde(default)]
    pub default: Option<String>,
    #[serde(default, rename = "model")]
    pub models: Vec<DeclaredModel>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct DeclaredModel {
    pub id: String,
    #[serde(default)]
    pub label: Option<String>,
}

/// `~/.config/armillary/models.toml`, beside the key files.
pub fn default_path() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
        .join(".config/armillary/models.toml")
}

/// Absent or unparseable is an EMPTY catalog, never an error — C-4's
/// posture, the same reason a workspace composing nothing is a working
/// host. A host with no file still pilots: the picker shows one row and
/// the default carries the session.
pub fn load(path: &Path) -> Catalog {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return Catalog::default();
    };
    match toml::from_str(&contents) {
        Ok(catalog) => catalog,
        Err(e) => {
            eprintln!("models: {} did not parse ({e}) — serving an empty catalog", path.display());
            Catalog::default()
        }
    }
}

/// The catalog's `default`, for `main`'s precedence chain.
pub fn declared_default() -> Option<String> {
    load(&default_path()).default
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_is_an_empty_catalog_not_an_error() {
        let catalog = load(Path::new("/nonexistent/models.toml"));
        assert!(catalog.models.is_empty());
        assert_eq!(catalog.default, None);
    }

    #[test]
    fn an_unparseable_file_is_also_empty_and_never_fatal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.toml");
        std::fs::write(&path, "this is not = = toml").unwrap();
        let catalog = load(&path);
        assert!(catalog.models.is_empty());
    }

    #[test]
    fn declared_order_and_labels_survive_the_parse() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.toml");
        std::fs::write(
            &path,
            r#"
default = "claude-sonnet-5"

[[model]]
id = "claude-sonnet-5"
label = "Sonnet 5"

[[model]]
id = "zen/deepseek-v4-flash"
label = "DeepSeek Flash (free)"
"#,
        )
        .unwrap();

        let catalog = load(&path);
        assert_eq!(catalog.default.as_deref(), Some("claude-sonnet-5"));
        // Order is the host's declaration order — it is what the picker shows,
        // so it must not be sorted or deduped on the way through.
        let ids: Vec<&str> = catalog.models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["claude-sonnet-5", "zen/deepseek-v4-flash"]);
        assert_eq!(catalog.models[1].label.as_deref(), Some("DeepSeek Flash (free)"));
    }

    #[test]
    fn a_model_entry_needs_only_an_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.toml");
        std::fs::write(&path, "[[model]]\nid = \"claude-opus-5\"\n").unwrap();
        let catalog = load(&path);
        assert_eq!(catalog.models.len(), 1);
        assert_eq!(catalog.models[0].label, None);
    }
}
