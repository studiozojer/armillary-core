use crate::auth::Caller;
use axum::Json;

/// `GET /whoami` — the presented credential's own facts, and nothing else's.
///
/// The FIRST authenticated read route, deliberately: R1 keeps reads open, but
/// this read is ABOUT the credential, so it cannot exist without one. It
/// answers "what may this device do" so the app stops learning its grants by
/// being refused. Never a roster — `principals` stays a host CLI.
pub async fn whoami(caller: Caller) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "name": caller.0.name,
        "grants": caller.0.grants.iter().map(|g| g.as_str()).collect::<Vec<_>>(),
        "minted": caller.0.minted,
    }))
}
