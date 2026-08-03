use crate::state::SharedState;
use axum::{extract::State, Json};
use serde_json::json;

/// The only evidence a deploy should trust.
///
/// A launchd service can report "loaded" while crash-looping — that is how the
/// voicenote endpoint's python-3.9 failure hid, wearing the costume of a
/// successful install. A 200 here is the thing that actually distinguishes them.
///
/// `commit` answers the question `version` cannot. The crate version is frozen
/// at `0.1.0` and says nothing about *which build* is replying, so a verb that
/// merged this morning and a verb that does not exist both come back
/// `unknown_tool`. Stamped at build time by `build.rs`; `unknown` when the
/// build had no git to ask. It names the commit, not the working tree — see
/// `build.rs` for why reporting dirtiness would lie.
pub async fn health(State(state): State<SharedState>) -> Json<serde_json::Value> {
    Json(json!({
        "ok": true,
        "root": state.root.display().to_string(),
        "version": env!("CARGO_PKG_VERSION"),
        "commit": env!("GIT_COMMIT"),
    }))
}
