use crate::{repos, state::SharedState, sync};
use axum::{extract::State, http::StatusCode, Json};

/// `GET /sync` — what the local refs say, without touching the network.
///
/// **Ungated on purpose.** It performs no fetch and no merge; it is a pure read
/// of the same disk `/tree` and `/composition` already serve, so gating it would
/// buy no safety and would stop the app from rendering module statuses on a host
/// that has not granted the sweep.
///
/// Never fails: a repo that cannot be read is a line in the report.
pub async fn status(State(state): State<SharedState>) -> Json<sync::SyncReport> {
    Json(sync::sweep(&state.root, false).await)
}

/// `POST /sync` — fetch everything, fast-forward what can be fast-forwarded.
///
/// Gated on `[router] sync = true`. The refusal names the exact table and key
/// it looked for, because `Router.extra` is unvalidated by design (C-5) and a
/// misspelled `snyc` is otherwise indistinguishable from a deliberate off.
pub async fn sweep(
    State(state): State<SharedState>,
) -> Result<Json<sync::SyncReport>, (StatusCode, String)> {
    if !repos::gate_enabled(&state.root) {
        return Err((
            StatusCode::FORBIDDEN,
            "this workspace has not granted the engine authority to run git. \
             Declare it by adding `sync = true` under `[router]` in \
             modules.local.toml (or modules.toml), then retry. \
             GET /sync reads status without the grant."
                .to_string(),
        ));
    }
    Ok(Json(sync::sweep(&state.root, true).await))
}
