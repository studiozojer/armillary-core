//! The armillary engine: a read-only files service over a composed workspace,
//! plus a chat loop (`loop_.rs`) over sessions logged under it.
//!
//! A harness is roughly 5% loop and 95% edge-of-the-world plumbing. This
//! crate was born loopless — useful on its own as an Explorer before the
//! loop existed — but the loop has since landed: `POST /instances/{id}/send`
//! runs one turn against a single model provider, chat-only, v0 (no
//! dispatch to other operators, no tool use yet). The Explorer surfaces
//! (`/tree`, `/file`, `/composition`) remain exactly what they were; the
//! loop is the organ that accreted onto them.

pub mod blocking;
pub mod git;
pub mod guard;
pub mod hash;
pub mod loop_;
pub mod log;
pub mod projection;
pub mod provider;
pub mod repos;
pub mod routes;
mod search;
pub mod sessions;
pub mod state;
#[cfg(test)]
pub mod testgit;
pub mod tools;
mod write;

use axum::{
    routing::{get, post},
    Router,
};
use state::{AppState, SharedState};
use std::sync::Arc;

pub fn app(state: AppState) -> Router {
    let shared: SharedState = Arc::new(state);
    Router::new()
        .route("/health", get(routes::health::health))
        .route("/composition", get(routes::composition::composition))
        .route("/tree", get(routes::tree::tree))
        .route("/file", get(routes::file::file))
        .route("/voicenotes", get(routes::voicenotes::voicenotes))
        .route("/instances", get(routes::instances::list).post(routes::instances::create))
        .route("/instances/{id}", get(routes::instances::attach))
        .route("/instances/{id}/send", post(routes::session_ops::send))
        .route("/instances/{id}/interrupt", post(routes::session_ops::interrupt))
        .route("/instances/{id}/evict", post(routes::session_ops::evict))
        .route("/streams/{stream}/events", get(routes::subscribe::subscribe))
        // `/repos/fetch` (static) and `/repos/{name}` (dynamic) occupy the
        // same two-segment shape. Verified live 2026-08-04 (see
        // `every_verb_is_403_when_nothing_is_granted`, and a since-reverted
        // experiment that registered `/repos/{name}` FIRST): axum 0.8's
        // `matchit` router resolves the static segment ahead of the dynamic
        // one regardless of registration order, and does not panic on the
        // overlap at startup either. Kept static-first anyway, for a human
        // reader scanning this list top to bottom — the ordering is
        // documentation now, not load-bearing.
        .route("/repos", get(routes::repos::list))
        .route("/repos/fetch", post(routes::repos::fetch_all))
        .route("/repos/{name}", get(routes::repos::one))
        .route("/repos/{name}/fetch", post(routes::repos::fetch_one))
        .route("/repos/{name}/pull", post(routes::repos::pull))
        .route("/repos/{name}/push", post(routes::repos::push))
        .route("/repos/{name}/log", get(routes::repos::log))
        .route("/repos/{name}/changes", get(routes::repos::changes))
        .with_state(shared)
}
