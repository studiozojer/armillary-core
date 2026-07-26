use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct AppState {
    /// Canonicalized workspace root. Every served path resolves under it.
    pub root: PathBuf,
}

pub type SharedState = Arc<AppState>;
