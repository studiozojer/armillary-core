//! The armillary engine: a read-only files service over a composed workspace.
//!
//! Deliberately loopless. A harness is roughly 5% loop and 95% edge-of-the-world
//! plumbing, so this is the plumbing — born without the loop, and useful on its
//! own as an Explorer while the loop does not exist. Organs accrete onto it.

pub mod blocking;
pub mod guard;
pub mod hash;
pub mod routes;
pub mod state;

use axum::{routing::get, Router};
use state::{AppState, SharedState};
use std::sync::Arc;

pub fn app(state: AppState) -> Router {
    let shared: SharedState = Arc::new(state);
    Router::new()
        .route("/health", get(routes::health::health))
        .route("/composition", get(routes::composition::composition))
        .route("/tree", get(routes::tree::tree))
        .route("/file", get(routes::file::file))
        .with_state(shared)
}
