use crate::{guard, hash::sha256_hex, state::SharedState};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};

/// 1 MiB. The inbox holds multi-megabyte voice memos and none of them should be
/// streamed to a phone as text.
const MAX_BYTES: u64 = 1024 * 1024;

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

pub async fn file(
    State(state): State<SharedState>,
    Query(q): Query<FileQuery>,
) -> Result<Json<FileResponse>, (StatusCode, String)> {
    let resolved =
        guard::resolve(&state.root, &q.path).map_err(|e| (e.status(), format!("{e:?}")))?;

    let meta = resolved
        .metadata()
        .map_err(|_| (StatusCode::NOT_FOUND, "not found".to_string()))?;
    if meta.is_dir() {
        return Err((StatusCode::BAD_REQUEST, "path is a directory".to_string()));
    }
    // Checked from metadata before reading, so an oversized file is never loaded
    // in order to be rejected.
    if meta.len() > MAX_BYTES {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("{} bytes exceeds the {MAX_BYTES} byte cap", meta.len()),
        ));
    }

    let raw = std::fs::read(&resolved).map_err(|_| (StatusCode::NOT_FOUND, "not found".to_string()))?;
    let sha256 = sha256_hex(&raw);

    // Binary gets a refusal rather than a guess. /tree still lists the .png;
    // inventing an encoding for it would be less honest than saying no.
    let text = String::from_utf8(raw).map_err(|_| {
        (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "not UTF-8 text; binary files are listed but not served in v0".to_string(),
        )
    })?;

    Ok(Json(FileResponse {
        path: q.path,
        sha256,
        bytes: meta.len(),
        text,
    }))
}
