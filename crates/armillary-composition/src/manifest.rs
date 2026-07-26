use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Module {
    pub name: String,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// C-5: the protocol interface is deliberately provisional — the manifest models
/// exactly one axis (`load`), and *kind* is unmodeled until earned. `extra`
/// captures unknown fields so an engine tolerates them rather than rejecting the
/// manifest. Never add `deny_unknown_fields` here; the constitution forbids
/// building anything that assumes the current shape is final.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Protocol {
    pub name: String,
    pub source: String,
    pub load: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires: Option<Vec<String>>,
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, toml::Value>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Composition {
    #[serde(default)]
    pub operators: Vec<Module>,
    #[serde(default)]
    pub commons: Vec<Module>,
    #[serde(default)]
    pub repos: Vec<Module>,
    #[serde(default)]
    pub protocols: Vec<Protocol>,
}

/// The raw shape as it appears on disk, before legacy normalization.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct RawManifest {
    #[serde(default)]
    pub operators: Vec<Module>,
    /// C-2 legacy: `[[models]]` normalizes to operators (the 2026-07-24 rename).
    #[serde(default)]
    pub models: Vec<Module>,
    /// C-2 legacy: `[[agents]]` normalizes to operators (older prose vocabulary).
    #[serde(default)]
    pub agents: Vec<Module>,
    #[serde(default)]
    pub commons: Vec<Module>,
    #[serde(default)]
    pub repos: Vec<Module>,
    #[serde(default)]
    pub protocols: Vec<Protocol>,
}

impl From<RawManifest> for Composition {
    fn from(raw: RawManifest) -> Self {
        // Declaration order within the file is preserved, and legacy sections
        // append after the canonical one. Declared legacy `models/` paths are
        // honored as written (C-2) — normalization is of the *section name*,
        // never of the path.
        let mut operators = raw.operators;
        operators.extend(raw.models);
        operators.extend(raw.agents);
        Composition {
            operators,
            commons: raw.commons,
            repos: raw.repos,
            protocols: raw.protocols,
        }
    }
}
