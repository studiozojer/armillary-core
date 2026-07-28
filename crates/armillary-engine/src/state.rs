use crate::sessions::Sessions;
use std::path::PathBuf;
use std::sync::Arc;

/// Which model pilots sessions, and its credential. No provider call exists
/// yet (Task 10) — this rides in `AppState` now so nothing downstream has to
/// grow this shape later. `api_key` is read from the `ANTHROPIC_API_KEY`
/// environment variable (never a CLI flag, which would land in shell
/// history and `ps`), and `Debug` below redacts it so it never lands in a
/// log line by accident either.
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

#[derive(Clone, Debug)]
pub struct AppState {
    /// Canonicalized workspace root. Every served path resolves under it.
    pub root: PathBuf,
    /// The one writer per stream (A-4) and the live fanout over it.
    pub sessions: Arc<Sessions>,
    pub model: ModelConfig,
}

pub type SharedState = Arc<AppState>;
