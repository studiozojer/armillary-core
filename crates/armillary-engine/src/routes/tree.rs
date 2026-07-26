use crate::{guard, state::SharedState};
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
struct Entry {
    name: String,
    dir: bool,
}

#[derive(Serialize)]
pub struct TreeResponse {
    path: String,
    entries: Vec<Entry>,
}

pub async fn tree(
    State(state): State<SharedState>,
    Query(q): Query<TreeQuery>,
) -> Result<Json<TreeResponse>, (StatusCode, String)> {
    let resolved =
        guard::resolve(&state.root, &q.path).map_err(|e| (e.status(), format!("{e:?}")))?;

    let read = std::fs::read_dir(&resolved)
        .map_err(|_| (StatusCode::BAD_REQUEST, "not a directory".to_string()))?;

    let mut entries: Vec<Entry> = Vec::new();
    for item in read.flatten() {
        let name = item.file_name().to_string_lossy().to_string();
        if guard::is_hidden_from_listings(&name) {
            continue;
        }
        // `file_type` does not follow symlinks; `metadata` does. A symlinked
        // directory should browse as a directory — this workspace routes real
        // content through symlinks (`models -> operators`, CLAUDE.local.md into
        // the commons). A dangling link resolves to nothing and is simply not
        // an entry, which is the presence-gated reading.
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

    Ok(Json(TreeResponse {
        path: q.path,
        entries,
    }))
}
