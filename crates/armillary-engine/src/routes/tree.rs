use crate::{blocking, guard, state::SharedState};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Deserialize)]
pub struct TreeQuery {
    #[serde(default)]
    pub path: String,
}

#[derive(Serialize)]
struct Entry {
    name: String,
    dir: bool,
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

/// A directory listing is a thing a phone renders. Unbounded, one response can
/// carry every entry of a build-index store — 1,147 in this workspace's largest
/// — and the cost lands on the client rather than here.
const MAX_ENTRIES: usize = 500;

/// All filesystem work for one listing, synchronous and self-contained so it can
/// be handed to a thread that is allowed to block.
fn list(root: &Path, path: &str) -> Result<(Vec<Entry>, usize), (StatusCode, String)> {
    let resolved = guard::resolve(root, path).map_err(|e| (e.status(), e.code().to_string()))?;

    let read = std::fs::read_dir(&resolved)
        .map_err(|_| (StatusCode::BAD_REQUEST, "not_a_directory".to_string()))?;

    let mut entries: Vec<Entry> = Vec::new();
    for item in read.flatten() {
        let name = item.file_name().to_string_lossy().to_string();
        if guard::is_hidden_from_listings(&name) {
            continue;
        }
        // `file_type` does not follow symlinks; `metadata` does. A symlinked
        // directory should browse as a directory — this workspace routes real
        // content through symlinks (`models -> operators`, CLAUDE.local.md into
        // the commons). A dangling link resolves to nothing and is simply not an
        // entry, which is the presence-gated reading.
        //
        // This is also the single most expensive call in the engine: it follows
        // links, so one entry pointing at a disconnected volume blocks for the
        // mount timeout. That is precisely why this whole function runs off the
        // async runtime.
        let Ok(meta) = item.path().metadata() else {
            continue;
        };
        entries.push(Entry {
            name,
            dir: meta.is_dir(),
        });
    }

    entries.sort_by(|a, b| {
        b.dir
            .cmp(&a.dir)
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
            .then_with(|| a.name.cmp(&b.name))
    });

    // Sorted before truncating, so the prefix is stable and meaningful rather
    // than whatever the filesystem happened to return first.
    let total = entries.len();
    entries.truncate(MAX_ENTRIES);
    Ok((entries, total))
}

pub async fn tree(
    State(state): State<SharedState>,
    Query(q): Query<TreeQuery>,
) -> Result<Json<TreeResponse>, (StatusCode, String)> {
    let root = state.root.clone();
    let path = q.path.clone();
    let (entries, total) = blocking::run(move || list(&root, &path)).await?;

    Ok(Json(TreeResponse {
        truncated: entries.len() < total,
        total,
        path: q.path,
        entries,
    }))
}
