use crate::{blocking, guard, hash::sha256_hex, state::SharedState};
use axum::{extract::State, http::StatusCode, Json};
use serde::Serialize;
use serde_json::json;

#[derive(Serialize)]
struct HashedFile {
    path: String,
    sha256: String,
}

#[derive(Serialize)]
struct ProtocolSource {
    name: String,
    path: String,
    present: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    sha256: Option<String>,
}

/// C-3 as running code: the composition is byte-derived from the manifests, and
/// a model is never asked to re-derive it. The rule exists because a local model
/// once read commented-out examples as a live composition — a TOML parser makes
/// that structurally impossible, since comments are not data.
///
/// Content hashes ride along — a sprint-1 design decision (D10 in
/// `zojercommons/projects/harness/specs/2026-07-24-sprint-1-explorer-spine-design.md`,
/// which is a decision sheet, NOT a rule of this standard). Every manifest read
/// and every protocol body resolved is
/// hashed. Nothing consumes the hashes yet. They cost one sha256 over a file
/// already in memory, and they mean that the day an event log exists, "which
/// bytes were in that window" is answerable without retrofitting anything.
pub async fn composition(
    State(state): State<SharedState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let root = state.root.clone();
    let body = blocking::run(move || build(&root)).await?;
    Ok(Json(body))
}

/// Parses both manifests and reads every protocol body — all synchronous
/// filesystem work, so it runs off the async runtime. This is the heaviest
/// route: one read and one SHA-256 per declared protocol, and it will sit on
/// the loop's boot-injection hot path.
fn build(root: &std::path::Path) -> Result<serde_json::Value, (StatusCode, String)> {
    let composition = armillary_composition::parse_workspace(root)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut manifests = Vec::new();
    for name in ["modules.toml", "modules.local.toml"] {
        if let Ok(bytes) = std::fs::read(root.join(name)) {
            manifests.push(HashedFile {
                path: name.to_string(),
                sha256: sha256_hex(&bytes),
            });
        }
    }

    // C-4: a protocol whose source is not present is reported absent, not an
    // error. A missing dependency means skip the protocol, never fail the boot.
    let protocol_sources: Vec<ProtocolSource> = composition
        .protocols
        .iter()
        .map(|p| {
            // Through the guard, not `root.join`. A manifest carrying an
            // absolute `source` would otherwise be read verbatim, because
            // `Path::join` with an absolute argument DISCARDS the root — the
            // exact footgun guard.rs names as its first defense. Not reachable
            // by a stranger today (the manifest is owner-authored), but this is
            // the route that B-2/B-4 will grow into returning protocol BODIES,
            // and an unguarded join gets copied forward.
            //
            // C-4 already blesses the reporting: a source that cannot be
            // resolved is absent, not an error.
            match guard::resolve(root, &p.source).and_then(|path| {
                std::fs::read(&path).map_err(|_| guard::GuardError::NotFound)
            }) {
                Ok(bytes) => ProtocolSource {
                    name: p.name.clone(),
                    path: p.source.clone(),
                    present: true,
                    sha256: Some(sha256_hex(&bytes)),
                },
                Err(_) => ProtocolSource {
                    name: p.name.clone(),
                    path: p.source.clone(),
                    present: false,
                    sha256: None,
                },
            }
        })
        .collect();

    let mut body = serde_json::to_value(&composition)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    body["manifests"] = json!(manifests);
    body["protocol_sources"] = json!(protocol_sources);
    Ok(body)
}
