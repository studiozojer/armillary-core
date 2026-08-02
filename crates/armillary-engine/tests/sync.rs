//! Route-level tests for `/sync` against an in-process axum app.

use armillary_engine::{
    app,
    log::store::LogStore,
    provider::KeylessProvider,
    sessions::Sessions,
    state::{AppState, ModelConfig},
};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceExt;

fn app_over(setup: impl FnOnce(&PathBuf)) -> axum::Router {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.keep();
    setup(&root);

    let data_dir = tempfile::tempdir().unwrap().keep();
    let store = LogStore::open(&data_dir).unwrap();

    app(AppState {
        root: root.canonicalize().unwrap(),
        sessions: Arc::new(Sessions::new(store)),
        model: ModelConfig {
            model: "claude-sonnet-5".to_string(),
            api_key: None,
        },
        provider: Arc::new(KeylessProvider),
        boot: None,
    })
}

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn post_sync_is_refused_when_the_workspace_does_not_declare_it() {
    let app = app_over(|root| {
        std::fs::write(root.join("modules.toml"), "[router]\ncontains = []\n").unwrap();
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/sync")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let text = String::from_utf8(
        axum::body::to_bytes(response.into_body(), 8192)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    // The refusal must name the exact key, because `extra` is unvalidated and
    // a misspelling is otherwise indistinguishable from a deliberate off.
    assert!(text.contains("[router]"), "refusal should name the table: {text}");
    assert!(text.contains("sync"), "refusal should name the key: {text}");
}

#[tokio::test]
async fn post_sync_runs_when_the_workspace_declares_it() {
    let app = app_over(|root| {
        std::fs::write(root.join("modules.toml"), "[router]\nsync = true\n").unwrap();
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/sync")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["enabled"], true);
    assert_eq!(body["fetched"], true);
}

#[tokio::test]
async fn get_sync_works_without_the_gate() {
    // A status read performs no fetch and no merge. It is a pure read, on the
    // same footing as /composition, so gating it would only stop the client
    // from rendering what it can already see through /tree.
    let app = app_over(|root| {
        std::fs::write(root.join("modules.toml"), "[router]\ncontains = []\n").unwrap();
    });
    let response = app
        .oneshot(Request::builder().uri("/sync").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["enabled"], false);
    assert_eq!(body["fetched"], false);
}

#[tokio::test]
async fn a_workspace_that_composes_nothing_reports_an_empty_sweep() {
    // C-4: a bare host is a working host.
    let app = app_over(|_root| {});
    let response = app
        .oneshot(Request::builder().uri("/sync").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert!(body["repos"].as_array().unwrap().is_empty());
    assert!(body["not_composed"].as_array().unwrap().is_empty());
}
