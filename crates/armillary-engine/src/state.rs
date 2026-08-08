use crate::provider::ProviderFor;
use crate::sessions::Sessions;
use std::path::PathBuf;
use std::sync::Arc;

/// The engine's configured model — resolved once at boot, and the fallback
/// for any instance whose own log names none (`loop_::run_turn`; an
/// instance's own recorded model, when present, takes precedence over this).
/// No credential
/// here: `KeyedProviders` (`provider.rs`) resolves keys per provider now,
/// not per model, so a copy living here too would be a second thing to leak
/// (see `provider.rs`'s `KeyedProviders` doc and the commit that deleted
/// `api_key` from this struct).
///
/// `Debug` stays hand-written even though there is nothing to redact today:
/// this struct may hold a credential again one day, and re-deriving `Debug`
/// then is exactly how that becomes a leak.
#[derive(Clone)]
pub struct ModelConfig {
    pub model: String,
}

impl std::fmt::Debug for ModelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelConfig").field("model", &self.model).finish()
    }
}

#[derive(Clone)]
pub struct AppState {
    /// Canonicalized workspace root. Every served path resolves under it.
    pub root: PathBuf,
    /// The one writer per stream (A-4) and the live fanout over it.
    pub sessions: Arc<Sessions>,
    pub model: ModelConfig,
    /// Which provider pilots a given model — resolved per turn from the
    /// instance's own recorded model (`loop_::model_for`), not once at
    /// boot. `Arc<dyn ProviderFor>` so tests can inject a fixed provider
    /// via `provider::fixed`, exactly as they injected `ScriptedProvider`
    /// before this seam existed.
    pub providers: Arc<dyn ProviderFor>,
    /// Where the host's model catalog lives. A field rather than a call to
    /// `models::default_path()` inside the route, because a route reading a
    /// hard-coded `$HOME` path is untestable — and a test that only passes
    /// on a machine which happens to lack the file is worse than no test.
    pub models_path: PathBuf,
    /// Key PRESENCE, never the keys. `GET /models` reports usability from
    /// these, so that route can never be the place a credential escapes.
    pub anthropic_key_present: bool,
    pub zen_key_present: bool,
    /// The router's own boot file, as declared by `[router] boot` — a path
    /// RELATIVE to `root`, deliberately unresolved here so the containment
    /// check happens at one place (`projection::resolve_boot_path`) rather
    /// than at two.
    ///
    /// Read once at startup: changing WHICH file boots needs a restart;
    /// changing its CONTENT does not, since the projection re-reads and
    /// re-hashes every turn and `rerecord_boot` handles the drift.
    pub boot: Option<String>,
}

/// Hand-written because `dyn ProviderFor` has no `Debug` impl of its own
/// (nor should it grow one just to satisfy a derive) — mirrors `Sessions`'
/// `finish_non_exhaustive` posture for the same reason.
impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("root", &self.root)
            .field("sessions", &self.sessions)
            .field("model", &self.model)
            .field("boot", &self.boot)
            .finish_non_exhaustive()
    }
}

pub type SharedState = Arc<AppState>;
