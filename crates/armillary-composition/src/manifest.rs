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
    /// **Optional, and interpreted only by engines that inject a system
    /// prompt.** The ordered files that constitute this module's identity —
    /// paths relative to the workspace root, most stable first.
    ///
    /// This is a composition field (every implementation reads manifests) with
    /// an engine-specific *use*. B-2 is explicitly conditional — *"where the
    /// engine can enforce (hooks, system-prompt injection, pre-tool gates)"* —
    /// so an implementation with no system slot ignores this and is still
    /// conformant. It is declared here rather than left to convention because
    /// convention is already wrong: this workspace has an operator with no
    /// `CLAUDE.md` whose boot surface is `self.md`.
    ///
    /// Order is the declaration and the caller must honour it: boot content is
    /// the prefix-cache candidate, and changing an early file invalidates
    /// everything after it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boot: Option<Vec<String>>,
    /// C-5, and the same reason `Protocol` and `Router` carry one: without it
    /// serde's default silently DROPS an unrecognized key. A manifest written
    /// against a newer engine would parse clean here, do nothing, and say
    /// nothing. Never add `deny_unknown_fields`.
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, toml::Value>,
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

/// The `[router]` table — the minimal core that IS this repo, never a module.
///
/// `contains` is the allowlist the workspace's gitignore mirrors. `boot` names
/// the router's OWN boot file, read once at instance creation — distinct from
/// a module-contributed `[[protocols]]` entry with `load = "boot"`, which
/// carries protocol semantics (`requires`, load timing, companion data) that
/// the router's own file does not.
///
/// Like `Protocol`, this carries `extra` and must never gain
/// `deny_unknown_fields` (C-5).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Router {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contains: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boot: Option<String>,
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, toml::Value>,
}

impl Router {
    /// True when nothing was declared. Gates serialization so a manifest with
    /// no `[router]` table produces JSON identical to before this key existed
    /// — which is what keeps every existing conformance fixture passing.
    pub fn is_empty(&self) -> bool {
        self.contains.is_empty() && self.boot.is_none() && self.extra.is_empty()
    }
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
    #[serde(default, skip_serializing_if = "Router::is_empty")]
    pub router: Router,
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
    // in RawManifest — no skip_serializing_if; it is Deserialize-only
    #[serde(default)]
    pub router: Router,
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
            router: raw.router,
        }
    }
}
