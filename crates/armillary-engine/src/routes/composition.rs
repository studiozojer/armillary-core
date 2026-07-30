use crate::{blocking, state::SharedState, tools};
use axum::{extract::State, http::StatusCode, Json};

/// C-3 as running code: the composition is byte-derived from the manifests, and
/// a model is never asked to re-derive it. The rule exists because a local model
/// once read commented-out examples as a live composition — a TOML parser makes
/// that structurally impossible, since comments are not data.
///
/// **The body is built by `tools::composition_payload`, which is also what the
/// `get_composition` tool calls.** One implementation on purpose: this route and
/// the tool answer the same question, and two builders would eventually
/// disagree about what a workspace is composed of. The tool strips the sha256
/// digests on its way out (a model cannot verify or act on them); this route
/// keeps them, because drift detection is what they are for.
pub async fn composition(
    State(state): State<SharedState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let root = state.root.clone();
    // Still off the async runtime: this is the heaviest route, one read and one
    // SHA-256 per declared protocol.
    let body = blocking::run(move || {
        tools::composition_payload(&root)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.status.to_string()))
    })
    .await?;
    Ok(Json(body))
}
