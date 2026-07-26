use crate::state::SharedState;
use axum::{extract::State, Json};
use serde_json::json;

/// The only evidence a deploy should trust.
///
/// A launchd service can report "loaded" while crash-looping — that is how the
/// voicenote endpoint's python-3.9 failure hid, wearing the costume of a
/// successful install. A 200 here is the thing that actually distinguishes them.
pub async fn health(State(state): State<SharedState>) -> Json<serde_json::Value> {
    Json(json!({
        "ok": true,
        "root": state.root.display().to_string(),
        "version": env!("CARGO_PKG_VERSION"),
    }))
}
