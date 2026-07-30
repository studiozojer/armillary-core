use crate::{blocking, state::SharedState, tools};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct FileQuery {
    #[serde(default)]
    pub path: String,
}

#[derive(Serialize)]
pub struct FileResponse {
    path: String,
    sha256: String,
    bytes: u64,
    text: String,
}

/// **The read itself lives in `tools::read_whole`,** which shares its gate —
/// `guard::resolve`, the directory check, the openable check — with the
/// `read_file` tool. Two copies of that gate would eventually disagree about
/// which of them serves a credential.
///
/// **This route stays all-or-nothing on purpose.** The tool pages (D15); the
/// route does not, because the Explorer has nowhere to put a second page and a
/// silently short file reads exactly like a complete one — the same defect
/// `tree.rs` refuses for listings. Over the ceiling this still answers
/// `too_large`, which is terminal for a human who can open the file another
/// way and no longer terminal for a model, which cannot.
pub async fn file(
    State(state): State<SharedState>,
    Query(q): Query<FileQuery>,
) -> Result<Json<FileResponse>, (StatusCode, String)> {
    let root = state.root.clone();
    let path = q.path.clone();
    let (sha256, bytes, text) = blocking::run(move || {
        tools::read_whole(&root, &path).map_err(|e| (e.http_status(), e.status.to_string()))
    })
    .await?;

    Ok(Json(FileResponse {
        path: q.path,
        sha256,
        bytes,
        text,
    }))
}
