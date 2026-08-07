//! The host's model catalog — what THIS machine offers to pilot with.
//!
//! Deliberately not a composition surface. `modules.toml` declares what is
//! composed (operators, repos, protocols), and this workspace's vocabulary
//! holds that operators are MODEL-AGNOSTIC — the model is what pilots them
//! in a given session. So the catalog lives beside the credentials it
//! depends on (`~/.config/armillary/{anthropic-key,zen-key}`), one ritual,
//! and the standard stays out of it.
//!
//! The *list* is read on demand, never cached: per request for `GET /models`
//! and per turn when resolving a model absent from an instance's own log
//! (`loop_::model_for`'s fallback), so editing the file's entries takes
//! effect everywhere they're consulted without a restart.
//!
//! The *default*, though, is resolved once — at boot, into
//! `AppState.model.model` (`main.rs`'s precedence chain: `--model`, then
//! this file's `default`, then a literal fallback) — because that resolved
//! value is also what `loop_::run_turn` falls back to for every instance
//! that names no model of its own. `GET /models` reports that SAME
//! boot-resolved value, not a fresh re-read of this file's `default` line,
//! so the two can never disagree: editing `models.toml`'s `default` after
//! boot changes what a future restart will resolve to, not what this run is
//! piloting with, and the endpoint says so honestly.

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

/// The catalog's `default`, for `main`'s precedence chain. `default = ""` in
/// `models.toml` filters to `None` here — the same guard the two read paths
/// in `routes/instances.rs` apply — so an empty declaration falls through to
/// the literal fallback instead of becoming the process default and handing
/// `AnthropicProvider` an empty model string (a 400 on every null-model turn).
pub fn declared_default() -> Option<String> {
    declared_default_at(&default_path())
}

/// `declared_default`'s body, over an explicit path — split out so the
/// filter can be exercised against a real temp file instead of the
/// hard-coded `$HOME` location `declared_default` itself is pinned to.
fn declared_default_at(path: &Path) -> Option<String> {
    load(path).default.filter(|s| !s.trim().is_empty())
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

    #[test]
    fn a_blank_declared_default_filters_to_none_rather_than_piloting_with_an_empty_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.toml");
        std::fs::write(&path, "default = \"\"\n").unwrap();
        assert_eq!(declared_default_at(&path), None);
    }

    #[test]
    fn a_whitespace_only_declared_default_also_filters_to_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.toml");
        std::fs::write(&path, "default = \"   \"\n").unwrap();
        assert_eq!(declared_default_at(&path), None);
    }

    #[test]
    fn a_real_declared_default_survives_the_filter() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.toml");
        std::fs::write(&path, "default = \"claude-sonnet-5\"\n").unwrap();
        assert_eq!(declared_default_at(&path).as_deref(), Some("claude-sonnet-5"));
    }
}
