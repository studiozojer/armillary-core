//! Route-level tests against an in-process axum app — no sockets, no network.

use armillary_engine::{
    app,
    log::store::LogStore,
    sessions::Sessions,
    state::{AppState, ModelConfig},
};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceExt;

const ONE_MIB: usize = 1024 * 1024;

fn model_config() -> ModelConfig {
    ModelConfig {
        model: "claude-sonnet-5".to_string(),
        api_key: None,
    }
}

/// Build an app over a temp root. The `TempDir` is leaked deliberately: these
/// are short-lived test processes and keeping the guard's canonicalization
/// honest matters more than reclaiming a few directories.
///
/// The sessions data dir is a SEPARATE tempdir, outside the served root — so
/// the existing `/tree`/`/file` expectations (fixed entry counts, cap tests)
/// stay unchanged. `app_with_data_dir_under_root` below is the fixture for
/// the one test that needs the data dir INSIDE the root.
fn app_over(setup: impl FnOnce(&PathBuf)) -> axum::Router {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.keep();
    setup(&root);

    let data_dir = tempfile::tempdir().unwrap().keep();
    let store = LogStore::open(&data_dir).unwrap();

    app(AppState {
        root: root.canonicalize().unwrap(),
        sessions: Arc::new(Sessions::new(store)),
        model: model_config(),
    })
}

/// Like `app_over`, but the data dir lives INSIDE the served root at
/// `.armillary` — the one arrangement that actually exercises `guard.rs`'s
/// denial of it through `/tree` and `/file` (A-5's sibling concern: this
/// engine serves the disk, so anything under the root is reachable unless
/// the guard says otherwise).
fn app_with_data_dir_under_root(setup: impl FnOnce(&PathBuf)) -> axum::Router {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.keep();
    setup(&root);
    let root = root.canonicalize().unwrap();

    let data_dir = root.join(".armillary");
    let store = LogStore::open(&data_dir).unwrap();

    app(AppState {
        root,
        sessions: Arc::new(Sessions::new(store)),
        model: model_config(),
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

async fn post_json(
    router: axum::Router,
    uri: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 8 * ONE_MIB)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

/// Error responses are `(StatusCode, String)` — plain text, not JSON — so
/// `get_json` reads them back as `Null`. Two different refusals can share a
/// status code (`not_openable` and `not_text` are both 415), so the body is
/// what actually says which branch fired.
async fn get_text(router: axum::Router, uri: &str) -> (StatusCode, String) {
    let response = router
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 8 * ONE_MIB)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
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
async fn tree_caps_a_large_directory_and_says_so() {
    // A silently short list reads exactly like a complete one. The client has
    // to be told, or the phone shows 500 of 1,147 entries and looks correct.
    let router = app_over(|root| {
        std::fs::create_dir(root.join("many")).unwrap();
        for i in 0..600 {
            std::fs::write(root.join(format!("many/f{i:04}.md")), "x").unwrap();
        }
    });

    let (status, json) = get_json(router, "/tree?path=many").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["entries"].as_array().unwrap().len(), 500);
    assert_eq!(json["total"], 600);
    assert_eq!(json["truncated"], true);
}

#[tokio::test]
async fn tree_reports_a_small_directory_as_complete() {
    let router = app_over(|root| {
        std::fs::create_dir(root.join("few")).unwrap();
        std::fs::write(root.join("few/a.md"), "x").unwrap();
    });
    let (_, json) = get_json(router, "/tree?path=few").await;
    assert_eq!(json["total"], 1);
    assert_eq!(json["truncated"], false);
}

#[tokio::test]
async fn swift_build_directories_are_denied() {
    // `.build` — with the dot — is where this workspace's thousand-entry
    // directories live. `build` alone missed every one of them.
    let router = app_over(|root| {
        std::fs::create_dir_all(root.join(".build/records")).unwrap();
    });
    let (status, _) = get_json(router, "/tree?path=.build").await;
    assert_eq!(status, StatusCode::FORBIDDEN);
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
    // `.md` so the openable check (now checked first — see
    // `unopenable_type_is_415_and_still_lists` below) doesn't mask the size
    // check this test exists to exercise.
    let router = app_over(|root| {
        std::fs::write(root.join("big.md"), vec![b'a'; ONE_MIB + 1]).unwrap();
    });
    let (status, _) = get_json(router, "/file?path=big.md").await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn non_utf8_bytes_in_an_openable_file_are_415_not_text() {
    // The openable check runs before this one, so the fixture needs an
    // allowlisted extension — otherwise the request never reaches the
    // `String::from_utf8` branch this test exists to cover, and it would pass
    // for the wrong reason (as `image.png` below now demonstrates).
    let router = app_over(|root| {
        std::fs::write(root.join("notes.md"), [0x23u8, 0x20, 0xFF, 0xFE]).unwrap();
    });
    let (status, body) = get_text(router, "/file?path=notes.md").await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    assert_eq!(body, "not_text");
}

#[tokio::test]
async fn unopenable_extension_is_415_not_openable_before_any_utf8_check() {
    // What `non_utf8_file_is_415` used to (mis)prove: `.png` is refused for
    // its extension alone, before the bytes are ever read, so this is a
    // `not_openable` regardless of what — or whether — the file contains
    // valid text.
    let router = app_over(|root| {
        std::fs::write(root.join("image.png"), [0x89u8, 0x50, 0x4E, 0x47, 0xFF, 0xFE]).unwrap();
    });
    let (status, body) = get_text(router, "/file?path=image.png").await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    assert_eq!(body, "not_openable");
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

#[tokio::test]
async fn credential_is_refused_as_a_credential_not_as_an_unknown_type() {
    // Rule ordering, and it is the one that would silently regress: the
    // denylist must be consulted before the allowlist, so `Secrets.xcconfig`
    // reads as "never served" (403) rather than "can't open this type" (415).
    // The two say different things to whoever is looking at the screen.
    let router = app_over(|root| {
        std::fs::create_dir_all(root.join("repos/app")).unwrap();
        std::fs::write(root.join("repos/app/Secrets.xcconfig"), "KEY=live").unwrap();
    });
    let (status, _) = get_json(router, "/file?path=repos/app/Secrets.xcconfig").await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn voicenotes_is_404_when_the_protocol_is_not_declared() {
    // The default fixture declares no voicenotes protocol at all.
    let (status, _) = get_json(app_over(|_| {}), "/voicenotes").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn unopenable_type_is_415_and_still_lists() {
    let setup = |root: &PathBuf| {
        std::fs::create_dir_all(root.join("local/inbox")).unwrap();
        std::fs::write(root.join("local/inbox/memo.m4a"), [0u8; 4]).unwrap();
    };

    let (status, _) = get_json(app_over(setup), "/file?path=local/inbox/memo.m4a").await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);

    // The allowlist governs opening, not listing. A browser that hides what it
    // cannot open is lying about the filesystem one level down.
    let (_, listing) = get_json(app_over(setup), "/tree?path=local/inbox").await;
    assert!(listing["entries"]
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e["name"] == "memo.m4a"));
}

#[tokio::test]
async fn create_instance_returns_201_and_logs_instance_created_at_seq_1() {
    let router = app_over(|_| {});
    let (status, created) =
        post_json(router.clone(), "/instances", serde_json::json!({ "operator": "tycho" })).await;

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(created["operator"], "tycho");
    assert_eq!(created["lastSeq"], 1);
    assert!(created["startedAt"].as_str().is_some_and(|s| !s.is_empty()));
    let id = created["id"].as_str().unwrap().to_string();
    assert_eq!(created["stream"], id, "stream == id in v0");

    // Proven over HTTP, since that's the only interface this test has: the
    // event actually landed in the log at seq 1, not merely in the response.
    let (status, attach) = get_json(router, &format!("/instances/{id}")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(attach["earliestSeq"], 1);
    assert_eq!(attach["headSeq"], 1);
}

#[tokio::test]
async fn create_instance_with_a_null_operator_is_accepted() {
    let router = app_over(|_| {});
    let (status, created) =
        post_json(router, "/instances", serde_json::json!({ "operator": null })).await;

    assert_eq!(status, StatusCode::CREATED);
    assert!(created["operator"].is_null());
}

#[tokio::test]
async fn list_shows_created_instances() {
    let router = app_over(|_| {});
    let (_, a) = post_json(router.clone(), "/instances", serde_json::json!({ "operator": "tycho" })).await;
    let (_, b) =
        post_json(router.clone(), "/instances", serde_json::json!({ "operator": "kepler" })).await;

    let (status, listed) = get_json(router, "/instances").await;
    assert_eq!(status, StatusCode::OK);

    let entries = listed.as_array().unwrap();
    assert_eq!(entries.len(), 2);
    let ids: Vec<&str> = entries.iter().map(|i| i["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&a["id"].as_str().unwrap()));
    assert!(ids.contains(&b["id"].as_str().unwrap()));
}

#[tokio::test]
async fn list_is_empty_before_any_instance_is_created() {
    let (status, listed) = get_json(app_over(|_| {}), "/instances").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed.as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn attach_returns_earliest_and_head_seq_and_the_instance() {
    let router = app_over(|_| {});
    let (_, created) =
        post_json(router.clone(), "/instances", serde_json::json!({ "operator": "tycho" })).await;
    let id = created["id"].as_str().unwrap();

    let (status, attach) = get_json(router, &format!("/instances/{id}")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(attach["earliestSeq"], 1);
    assert_eq!(attach["headSeq"], 1);
    assert_eq!(attach["instance"]["id"], id);
    assert_eq!(attach["instance"]["operator"], "tycho");
}

#[tokio::test]
async fn attach_on_an_unknown_id_is_404_unknown_instance() {
    let (status, body) = get_text(app_over(|_| {}), "/instances/does-not-exist").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body, "unknown_instance");
}

#[tokio::test]
async fn tree_and_file_deny_the_data_dir_when_it_lives_under_the_served_root() {
    let router = app_with_data_dir_under_root(|_| {});

    let (_, created) =
        post_json(router.clone(), "/instances", serde_json::json!({ "operator": null })).await;
    let id = created["id"].as_str().unwrap();

    let (status, json) = get_json(router.clone(), "/tree?path=").await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        !json["entries"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["name"] == ".armillary"),
        "the data dir must never be listed"
    );

    let (status, body) =
        get_text(router, &format!("/file?path=.armillary/streams/{id}.jsonl")).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(body, "denied_noise");
}
