use crate::{blocking, guard, hash::sha256_hex, state::SharedState};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

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

/// All filesystem work for one file read, synchronous and self-contained so it
/// can be handed to a thread that is allowed to block.
fn read(root: &Path, path: &str) -> Result<(String, u64, String), (StatusCode, String)> {
    let resolved = guard::resolve(root, path).map_err(|e| (e.status(), e.code().to_string()))?;

    let meta = resolved
        .metadata()
        .map_err(|_| (StatusCode::NOT_FOUND, "not_found".to_string()))?;
    if meta.is_dir() {
        return Err((StatusCode::BAD_REQUEST, "is_a_directory".to_string()));
    }
    // Checked from metadata before reading, so an oversized file is never loaded
    // in order to be rejected.
    if meta.len() > MAX_BYTES {
        return Err((StatusCode::PAYLOAD_TOO_LARGE, "too_large".to_string()));
    }

    let raw =
        std::fs::read(&resolved).map_err(|_| (StatusCode::NOT_FOUND, "not_found".to_string()))?;
    let sha256 = sha256_hex(&raw);

    // Binary gets a refusal rather than a guess. /tree still lists the .png;
    // inventing an encoding for it would be less honest than saying no.
    let text = String::from_utf8(raw)
        .map_err(|_| (StatusCode::UNSUPPORTED_MEDIA_TYPE, "not_text".to_string()))?;

    Ok((sha256, meta.len(), text))
}

pub async fn file(
    State(state): State<SharedState>,
    Query(q): Query<FileQuery>,
) -> Result<Json<FileResponse>, (StatusCode, String)> {
    let root = state.root.clone();
    let path = q.path.clone();
    let (sha256, bytes, text) = blocking::run(move || read(&root, &path)).await?;

    Ok(Json(FileResponse {
        path: q.path,
        sha256,
        bytes,
        text,
    }))
}
