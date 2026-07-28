use crate::provider::ModelProvider;
use crate::sessions::Sessions;
use std::path::PathBuf;
use std::sync::Arc;

/// Which model pilots sessions, and its credential. No provider call exists
/// yet (Task 10) — this rides in `AppState` now so nothing downstream has to
/// grow this shape later. `api_key` resolves from, in order, the
/// `ANTHROPIC_API_KEY` environment variable or the per-machine file
/// `~/.config/armillary/anthropic-key` (see `main::resolve_api_key` — never a
/// CLI flag, which would land in shell history and `ps`), and `Debug` below
/// redacts it so it never lands in a log line by accident either.
#[derive(Clone)]
pub struct ModelConfig {
    pub model: String,
    pub api_key: Option<String>,
}

impl std::fmt::Debug for ModelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelConfig")
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

#[derive(Clone)]
pub struct AppState {
    /// Canonicalized workspace root. Every served path resolves under it.
    pub root: PathBuf,
    /// The one writer per stream (A-4) and the live fanout over it.
    pub sessions: Arc<Sessions>,
    pub model: ModelConfig,
    /// The model provider the loop (`loop_.rs`) calls for every turn —
    /// `Arc<dyn ModelProvider>` so tests can swap in `ScriptedProvider` (or a
    /// recording double) without touching `main.rs`'s wiring.
    pub provider: Arc<dyn ModelProvider>,
}

/// Hand-written because `dyn ModelProvider` has no `Debug` impl of its own
/// (nor should it grow one just to satisfy a derive) — mirrors `Sessions`'
/// `finish_non_exhaustive` posture for the same reason.
impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("root", &self.root)
            .field("sessions", &self.sessions)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

pub type SharedState = Arc<AppState>;
