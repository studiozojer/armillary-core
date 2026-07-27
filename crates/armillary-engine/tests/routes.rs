//! Route-level tests against an in-process axum app — no sockets, no network.

use armillary_engine::{app, state::AppState};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::path::PathBuf;
use tower::ServiceExt;

const ONE_MIB: usize = 1024 * 1024;

/// Build an app over a temp root. The `TempDir` is leaked deliberately: these
/// are short-lived test processes and keeping the guard's canonicalization
/// honest matters more than reclaiming a few directories.
fn app_over(setup: impl FnOnce(&PathBuf)) -> axum::Router {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.keep();
    setup(&root);
    app(AppState {
        root: root.canonicalize().unwrap(),
    })
}

async fn get_json(router: axum::Router, uri: &str) -> (StatusCode, serde_json::Value) {
    let response = router
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 8 * ONE_MIB)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

#[tokio::test]
async fn health_reports_ok() {
    let (status, json) = get_json(app_over(|_| {}), "/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["ok"], true);
    assert!(json["version"].is_string());
    assert!(json["root"].is_string());
}

#[tokio::test]
async fn composition_is_byte_derived_and_hashed() {
    let router = app_over(|root| {
        std::fs::write(
            root.join("modules.toml"),
            "# [[operators]]\n# name = \"commented\"\n# path = \"nope\"\n\n\
             [[operators]]\nname = \"tycho\"\npath = \"operators/tycho\"\n",
        )
        .unwrap();
    });

    let (status, json) = get_json(router, "/composition").await;
    assert_eq!(status, StatusCode::OK);

    // C-3: a commented entry is not a declaration.
    assert_eq!(json["operators"].as_array().unwrap().len(), 1);
    assert_eq!(json["operators"][0]["name"], "tycho");

    // The manifest it actually parsed is hashed (sprint-1 design sheet D10).
    let manifests = json["manifests"].as_array().unwrap();
    assert_eq!(manifests.len(), 1);
    assert_eq!(manifests[0]["path"], "modules.toml");
    assert_eq!(manifests[0]["sha256"].as_str().unwrap().len(), 64);
}

#[tokio::test]
async fn composition_reports_an_absent_protocol_source_without_failing() {
    let router = app_over(|root| {
        std::fs::write(
            root.join("modules.toml"),
            "[[protocols]]\nname = \"board\"\nsource = \"nowhere/practice.md\"\nload = \"boot\"\n",
        )
        .unwrap();
    });

    let (status, json) = get_json(router, "/composition").await;
    assert_eq!(status, StatusCode::OK, "C-4: a missing source is not an error");
    assert_eq!(json["protocol_sources"][0]["present"], false);
    assert!(json["protocol_sources"][0]["sha256"].is_null());
}

#[tokio::test]
async fn tree_lists_directories_first_and_hides_noise() {
    let router = app_over(|root| {
        std::fs::create_dir(root.join(".git")).unwrap();
        std::fs::create_dir(root.join("node_modules")).unwrap();
        std::fs::create_dir(root.join("athanor")).unwrap();
        std::fs::write(root.join("BOARD.md"), "# board").unwrap();
        std::fs::write(root.join(".env"), "TOKEN=hunter2").unwrap();
    });

    let (status, json) = get_json(router, "/tree?path=").await;
    assert_eq!(status, StatusCode::OK);
    let entries = json["entries"].as_array().unwrap();

    assert_eq!(entries.len(), 2, ".git, node_modules and .env must be hidden");
    assert_eq!(entries[0]["name"], "athanor");
    assert_eq!(entries[0]["dir"], true);
    assert_eq!(entries[1]["name"], "BOARD.md");
    assert_eq!(entries[1]["dir"], false);
}

#[tokio::test]
async fn tree_refuses_traversal_with_403() {
    let (status, _) = get_json(app_over(|_| {}), "/tree?path=../..").await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn file_returns_text_with_its_hash() {
    let router = app_over(|root| {
        std::fs::write(root.join("note.md"), "# hello").unwrap();
    });

    let (status, json) = get_json(router, "/file?path=note.md").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["text"], "# hello");
    assert_eq!(json["bytes"], 7);
    // printf '# hello' | shasum -a 256
    assert_eq!(
        json["sha256"],
        "ea67f39f2a707e536439ee31e49fdd586b4a8437d3408f0466112d040cd06681"
    );
}

#[tokio::test]
async fn file_over_the_cap_is_413() {
    let router = app_over(|root| {
        std::fs::write(root.join("big.bin"), vec![b'a'; ONE_MIB + 1]).unwrap();
    });
    let (status, _) = get_json(router, "/file?path=big.bin").await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn non_utf8_file_is_415() {
    let router = app_over(|root| {
        std::fs::write(root.join("image.png"), [0x89u8, 0x50, 0x4E, 0x47, 0xFF, 0xFE]).unwrap();
    });
    let (status, _) = get_json(router, "/file?path=image.png").await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[tokio::test]
async fn dotenv_is_refused_even_when_guessed_directly() {
    let router = app_over(|root| {
        std::fs::create_dir_all(root.join("repos/kairos-engine")).unwrap();
        std::fs::write(root.join("repos/kairos-engine/.env"), "SECRET=1").unwrap();
    });
    let (status, _) = get_json(router, "/file?path=repos/kairos-engine/.env").await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "the denylist must hold for a path no listing ever revealed"
    );
}
