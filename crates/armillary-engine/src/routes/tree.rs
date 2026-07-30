use crate::{
    blocking,
    state::SharedState,
    tools::{self, Entry},
};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct TreeQuery {
    #[serde(default)]
    pub path: String,
}

#[derive(Serialize)]
pub struct TreeResponse {
    path: String,
    entries: Vec<Entry>,
    /// How many entries the directory actually holds, before the cap.
    total: usize,
    /// True when `entries` is a prefix of `total`. The client has to be told —
    /// a silently short list reads exactly like a complete one, which is the
    /// same defect as a fixture glob that matches nothing and reports success.
    truncated: bool,
}

/// **The listing itself lives in `tools::list_entries`,** which is also what the
/// `list_directory` tool calls. One implementation on purpose: this route and
/// the tool answer the same question, and two of them would eventually disagree
/// about what a directory contains — most dangerously about what it *hides*,
/// since the credential and noise filtering is part of that body.
///
/// What stays here is the HTTP shape. `truncated`/`total` are the route's way
/// of saying the list is a prefix; the tool renders the same fact as a sentence
/// (D7). Both must say it — a silently short list reads exactly like a complete
/// one.
pub async fn tree(
    State(state): State<SharedState>,
    Query(q): Query<TreeQuery>,
) -> Result<Json<TreeResponse>, (StatusCode, String)> {
    let root = state.root.clone();
    let path = q.path.clone();
    // Off the async runtime: `metadata()` follows symlinks, so one entry
    // pointing at a disconnected volume blocks for the mount timeout.
    let (entries, total) = blocking::run(move || {
        tools::list_entries(&root, &path).map_err(|e| (e.http_status(), e.status.to_string()))
    })
    .await?;

    Ok(Json(TreeResponse {
        truncated: entries.len() < total,
        total,
        path: q.path,
        entries,
    }))
}
