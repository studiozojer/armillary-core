//! Route-level tests against an in-process axum app — no sockets, no network.

use armillary_engine::{
    app,
    log::store::LogStore,
    principals::{hash_token, write_principal, Grant, Principal},
    provider::{self, KeylessProvider},
    sessions::Sessions,
    state::{AppState, ModelConfig},
};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tower::ServiceExt;

const ONE_MIB: usize = 1024 * 1024;

/// The bearer token every authenticated call in this file presents. Fixed,
/// mirroring `tests/repos.rs`'s `TEST_TOKEN` (Task 7's precedent): nothing
/// here asserts anything about the token itself, only about what a request
/// carrying it is allowed to do, so a literal string keeps `full_grant_registry`
/// and `post_json`'s callers in sync without threading a return value through
/// every call site.
const TEST_TOKEN: &str = "test-fixed-token-for-tests-routes-rs-2026-08-07";

/// A fresh registry directory holding one principal — both grants — that
/// authenticates as `TEST_TOKEN`. Mirrors `tests/repos.rs`'s
/// `full_grant_registry`: a new tempdir per call, since `Registry::load` is
/// read per request regardless and nothing here needs state to persist
/// across calls.
fn full_grant_registry() -> PathBuf {
    let dir = tempfile::tempdir().unwrap().keep();
    write_principal(
        &dir,
        &Principal {
            name: "test-client".to_string(),
            token_hash: hash_token(TEST_TOKEN),
            grants: vec![Grant::Sync, Grant::Push],
            minted: "2026-08-07T00:00:00Z".to_string(),
        },
    )
    .unwrap();
    dir
}

fn model_config() -> ModelConfig {
    ModelConfig {
        model: "claude-sonnet-5".to_string(),
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
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: None,
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
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: None,
    })
}

/// Like `app_over`, but with `[router] boot` declared. The boot file is
/// written into the served root, since a boot path must resolve under it.
///
/// Returns the root and the session data dir alongside the router — `boot`'s
/// stricter tests need to read the stream's raw events back (identity, not
/// just count) or hand them to `project_context` directly (the end-to-end
/// property this task exists to deliver), and `AppState`'s `Sessions` is
/// moved into the router with no way to get it back out. Reopening a fresh
/// `LogStore` over the same `data_dir` afterward — the same pattern
/// `loop_flow.rs`'s crash-resume test uses — reads what's actually durable.
fn app_with_boot(boot_rel: &str, contents: &str) -> (axum::Router, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.keep();
    std::fs::write(root.join(boot_rel), contents).unwrap();
    let root = root.canonicalize().unwrap();

    let data_dir = tempfile::tempdir().unwrap().keep();
    let store = LogStore::open(&data_dir).unwrap();

    let router = app(AppState {
        root: root.clone(),
        sessions: Arc::new(Sessions::new(store)),
        model: model_config(),
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: Some(boot_rel.to_string()),
    });
    (router, root, data_dir)
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

/// `bearer`, when `Some`, is sent as `Authorization: Bearer <token>` —
/// threaded through here rather than a parallel helper (Task 8's precedent),
/// so every call site says explicitly whether it is authenticating or
/// deliberately not.
async fn post_json(
    router: axum::Router,
    uri: &str,
    body: serde_json::Value,
    bearer: Option<&str>,
) -> (StatusCode, serde_json::Value) {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(token) = bearer {
        builder = builder.header(axum::http::header::AUTHORIZATION, format!("Bearer {token}"));
    }
    let response = router
        .oneshot(builder.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 8 * ONE_MIB)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

/// Creates an instance, then attaches to it — returning the attach JSON
/// (`headSeq`/`earliestSeq`/`instance`) so callers can read how many events
/// actually landed on the stream, not just what `create`'s own response
/// claims (which deliberately still reports `lastSeq: 1` — see `create`'s
/// doc comment).
async fn create_and_attach(router: axum::Router) -> serde_json::Value {
    let (status, created) =
        post_json(router.clone(), "/instances", serde_json::json!({ "operator": null }), Some(TEST_TOKEN)).await;
    assert_eq!(status, StatusCode::CREATED);
    let id = created["id"].as_str().unwrap().to_string();

    let (status, attach) = get_json(router, &format!("/instances/{id}")).await;
    assert_eq!(status, StatusCode::OK);
    attach
}

/// Assert a misconfigured boot loaded nothing — the session exists and has no
/// system prompt.
///
/// This used to assert "no boot event was appended", which was the right claim
/// when a boot event could only say `{path, sha256}`: appending one meant
/// CLAIMING a hash for a file that could not be read, and the projection would
/// then fail every turn on the stream forever. B-2's `{files: [{path,
/// present}]}` can say *absent*, so the premise changed — the engine now
/// records the failed path instead of dropping it, and a misconfigured boot
/// stops being invisible from the phone.
///
/// What the four tests below were always about survives intact: a broken boot
/// declaration must not break the session, and must not smuggle content in.
fn assert_boot_loaded_nothing(root: &std::path::Path, data_dir: &std::path::Path, attach: &serde_json::Value) {
    let id = attach["instance"]["id"].as_str().unwrap();
    let store = LogStore::open(data_dir).unwrap();
    let events = store.read_from(id, 0).unwrap();
    let turn = armillary_engine::projection::project_context(&events, root)
        .expect("a broken boot declaration must not make the stream unprojectable");
    assert_eq!(turn.system, None, "{events:?}");
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
async fn health_reports_the_build_it_is_answering_from() {
    // The defect this closes: `version` is the crate version from Cargo.toml,
    // unchanged in months and never going to change. So a tool that has just
    // been merged but not rebuilt answers `unknown_tool`, and from a phone
    // that is indistinguishable from the tool never having shipped —
    // absence-vs-refusal, one layer below the branch built to prevent it.
    let (_, json) = get_json(app_over(|_| {}), "/health").await;

    let commit = json["commit"].as_str().expect("health must report a commit");
    let version = json["version"].as_str().unwrap();

    assert!(!commit.is_empty(), "an empty commit answers nothing");

    // The repair that would defeat the whole point: aliasing `commit` to the
    // constant we are trying to get away from. Pinned so it cannot pass review
    // as a fix.
    assert_ne!(
        commit, version,
        "commit must carry build identity, not restate the frozen crate version"
    );

    // The build script's contract, in the only two shapes it may take: a hex
    // revision, or the honest admission that this build could not tell. A
    // plausible-looking value that is neither is the failure mode worth
    // catching — a stamp that lies is worse than no stamp.
    assert!(
        commit == "unknown" || commit.chars().all(|c| c.is_ascii_hexdigit()),
        "a commit is a hex revision or `unknown`, got {commit:?}"
    );
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
        post_json(router.clone(), "/instances", serde_json::json!({ "operator": "tycho" }), Some(TEST_TOKEN)).await;

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
    // seq 2 is DD-1's composition event.
    assert_eq!(attach["headSeq"], 2);
}

#[tokio::test]
async fn create_instance_with_a_null_operator_is_accepted() {
    let router = app_over(|_| {});
    let (status, created) =
        post_json(router, "/instances", serde_json::json!({ "operator": null }), Some(TEST_TOKEN)).await;

    assert_eq!(status, StatusCode::CREATED);
    assert!(created["operator"].is_null());
}

#[tokio::test]
async fn creating_an_instance_without_a_credential_is_refused() {
    // Instance creation spends the model budget and starts a turn that can
    // write files. It is a mutation, and it is the route a file write
    // inherits its principal through.
    let router = app_over(|_| {});
    let (status, _) = post_json(router, "/instances", serde_json::json!({}), None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn listing_instances_without_a_credential_still_works() {
    // Reads stay open (R1). Pinned so the gate does not creep onto the read
    // surface without the decision being made again.
    let (status, _) = get_json(app_over(|_| {}), "/instances").await;
    assert_eq!(status, StatusCode::OK);
}

// The three tests below use a DELIBERATELY NONEXISTENT instance id
// (`no-such-instance`) rather than a real one. That makes each one assert
// something stronger than "auth is required": it asserts auth runs BEFORE
// the route resolves the instance. A 401 (not 404) proves the `Caller`
// extractor rejects the request before the handler ever looks the instance
// up — `send`/`interrupt`/`evict` all 404 on an unknown id once past the
// gate, so a 404 here would mean extraction is not running first, the
// ordering a future refactor could quietly invert with nothing else in this
// file able to notice (nothing here calls `/send`, `/interrupt`, or
// `/evict` otherwise, and `tests/loop_flow.rs`'s client attaches a
// credential to every request unconditionally).

#[tokio::test]
async fn sending_without_a_credential_is_401_before_the_instance_is_resolved() {
    let router = app_over(|_| {});
    let (status, _) = post_json(
        router,
        "/instances/no-such-instance/send",
        serde_json::json!({ "text": "hi", "clientKey": "c1" }),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn interrupting_without_a_credential_is_401_before_the_instance_is_resolved() {
    let router = app_over(|_| {});
    let (status, _) =
        post_json(router, "/instances/no-such-instance/interrupt", serde_json::json!({}), None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn evicting_without_a_credential_is_401_before_the_instance_is_resolved() {
    let router = app_over(|_| {});
    let (status, _) = post_json(
        router,
        "/instances/no-such-instance/evict",
        serde_json::json!({ "eventId": "whatever" }),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn list_shows_created_instances() {
    let router = app_over(|_| {});
    let (_, a) = post_json(router.clone(), "/instances", serde_json::json!({ "operator": "tycho" }), Some(TEST_TOKEN)).await;
    let (_, b) =
        post_json(router.clone(), "/instances", serde_json::json!({ "operator": "kepler" }), Some(TEST_TOKEN)).await;

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
        post_json(router.clone(), "/instances", serde_json::json!({ "operator": "tycho" }), Some(TEST_TOKEN)).await;
    let id = created["id"].as_str().unwrap();

    let (status, attach) = get_json(router, &format!("/instances/{id}")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(attach["earliestSeq"], 1);
    assert_eq!(attach["headSeq"], 2);
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
        post_json(router.clone(), "/instances", serde_json::json!({ "operator": null }), Some(TEST_TOKEN)).await;
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

#[tokio::test]
async fn create_appends_a_boot_event_after_instance_created() {
    let contents = "# Getting started\n\nAn operator is an identity.\n";
    let (router, _root, data_dir) = app_with_boot("getting-started.md", contents);
    let attach = create_and_attach(router).await;
    // instance_created is seq 1, boot is seq 2, DD-1's composition is seq 3 —
    // the ordering the instance registry depends on
    // (instance_from_first_event requires seq 1 to be instance_created), with
    // composition last so the earlier numbers stay where they were documented.
    assert_eq!(attach["headSeq"], 3);
    assert_eq!(attach["earliestSeq"], 1);

    // `headSeq == 2` alone would pass if `create` appended ANY second event
    // — read the stream back and check what actually landed, and in what
    // order, so an inversion (boot appended before instance_created) fails
    // HERE directly, rather than only incidentally via some other test's
    // 404 surfacing as "404 != 200".
    let id = attach["instance"]["id"].as_str().unwrap().to_string();
    let store = LogStore::open(&data_dir).unwrap();
    let events = store.read_from(&id, 0).unwrap();
    let types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
    assert_eq!(types, vec!["instance_created", "boot", "composition"]);
    assert_eq!(events[1].data["files"][0]["path"], "getting-started.md");
    assert_eq!(
        events[1].data["files"][0]["sha256"],
        armillary_engine::hash::sha256_hex(contents.as_bytes())
    );
}

#[tokio::test]
async fn a_created_instances_boot_event_projects_to_a_system_prompt() {
    // The end-to-end property this task exists to deliver: create → the
    // boot event lands durably → project_context reads it back as
    // `system: Some(...)`. This is the one thing no other test in this file
    // actually connects, and it exercises the relative-path storage, the
    // sha, the event type and the ordering together — a non-UTF-8 boot file
    // (guarded against separately below) would have failed exactly here,
    // as `BootUnreadable` rather than a system prompt.
    let contents = "# Getting started\n\nAn operator is an identity.\n";
    let (router, root, data_dir) = app_with_boot("getting-started.md", contents);
    let attach = create_and_attach(router).await;
    let id = attach["instance"]["id"].as_str().unwrap().to_string();

    let store = LogStore::open(&data_dir).unwrap();
    let events = store.read_from(&id, 0).unwrap();
    let turn = armillary_engine::projection::project_context(&events, &root).unwrap();
    assert_eq!(turn.system, Some(contents.to_string()));
}

#[tokio::test]
async fn create_records_the_composition_so_a_session_knows_what_it_was_booted_into() {
    // DD-1. The resolved manifest is the one thing a session cannot work out
    // for itself: C-3 forbids handing a model raw TOML to re-derive, because a
    // local model once read commented-out examples as a live composition.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.keep();
    std::fs::write(
        root.join("modules.toml"),
        "# [[repos]]\n# name = \"ghost\"\n[[repos]]\nname = \"kairos-engine\"\npath = \"p\"\n",
    )
    .unwrap();
    let data_dir = tempfile::tempdir().unwrap().keep();
    let store = LogStore::open(&data_dir).unwrap();
    let router = app(AppState {
        root: root.canonicalize().unwrap(),
        sessions: Arc::new(Sessions::new(store)),
        model: model_config(),
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: None,
    });

    let (status, created) =
        post_json(router, "/instances", serde_json::json!({ "operator": null }), Some(TEST_TOKEN)).await;
    assert_eq!(status, StatusCode::CREATED);
    let id = created["id"].as_str().unwrap().to_string();

    let store = LogStore::open(&data_dir).unwrap();
    let events = store.read_from(&id, 0).unwrap();
    let types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
    assert_eq!(types, vec!["instance_created", "composition"]);

    // It reaches the model as a turn, not as an event nobody projects.
    let turn = armillary_engine::projection::project_context(&events, &root).unwrap();
    let text = format!("{:?}", turn.messages);
    assert!(text.contains("kairos-engine"), "{text}");
    assert!(!text.contains("ghost"), "a commented-out repo reached the model: {text}");
}

/// A workspace declaring one operator with a two-file boot surface.
///
/// The paths deliberately are NOT `<path>/CLAUDE.md`: a convention would already
/// be wrong in the real workspace, where one operator has no CLAUDE.md at all
/// and boots from `self.md`. Declared, not inferred.
fn app_with_operator_boot() -> (axum::Router, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.keep();
    std::fs::create_dir_all(root.join("operators/tycho")).unwrap();
    std::fs::write(root.join("operators/tycho/principles.md"), "# principles").unwrap();
    std::fs::write(root.join("operators/tycho/voice.md"), "# voice").unwrap();
    std::fs::write(
        root.join("modules.toml"),
        "[[operators]]\nname = \"tycho\"\npath = \"operators/tycho\"\n\
         boot = [\"operators/tycho/principles.md\", \"operators/tycho/voice.md\"]\n\n\
         [[operators]]\nname = \"leavitt\"\npath = \"operators/leavitt\"\n",
    )
    .unwrap();
    let root = root.canonicalize().unwrap();

    let data_dir = tempfile::tempdir().unwrap().keep();
    let store = LogStore::open(&data_dir).unwrap();
    let router = app(AppState {
        root: root.clone(),
        sessions: Arc::new(Sessions::new(store)),
        model: model_config(),
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: None,
    });
    (router, root, data_dir)
}

async fn create_for(router: axum::Router, operator: serde_json::Value) -> String {
    let (status, created) =
        post_json(router, "/instances", serde_json::json!({ "operator": operator }), Some(TEST_TOKEN)).await;
    assert_eq!(status, StatusCode::CREATED);
    created["id"].as_str().unwrap().to_string()
}

#[tokio::test]
async fn summoning_an_operator_boots_the_files_it_declared_in_declared_order() {
    // **B-2.** Before this, every instance got the router's own file whatever
    // was summoned — the rule was in the constitution and nowhere in the
    // engine. The point is speed as much as correctness: pulling these would
    // cost a listing and two reads, each a full round trip with the whole
    // window re-sent, before the operator said a word.
    let (router, root, data_dir) = app_with_operator_boot();
    let id = create_for(router, serde_json::json!("tycho")).await;

    let store = LogStore::open(&data_dir).unwrap();
    let events = store.read_from(&id, 0).unwrap();
    let types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
    assert_eq!(types, vec!["instance_created", "boot", "composition"]);

    let turn = armillary_engine::projection::project_context(&events, &root).unwrap();
    assert_eq!(turn.system.as_deref(), Some("# principles\n\n# voice"));
}

#[tokio::test]
async fn an_operator_that_declares_no_boot_gets_no_boot_event() {
    // C-4: presence-gated. Most operators will never declare one.
    let (router, _root, data_dir) = app_with_operator_boot();
    let id = create_for(router, serde_json::json!("leavitt")).await;

    let store = LogStore::open(&data_dir).unwrap();
    let types: Vec<String> = store
        .read_from(&id, 0)
        .unwrap()
        .iter()
        .map(|e| e.event_type.clone())
        .collect();
    assert_eq!(types, vec!["instance_created", "composition"]);
}

#[tokio::test]
async fn a_dispatcher_instance_does_not_inherit_any_operators_boot() {
    // No operator summoned means no operator identity. An instance that
    // silently booted as tycho because tycho is declared first would be the
    // worst possible failure here.
    let (router, _root, data_dir) = app_with_operator_boot();
    let id = create_for(router, serde_json::Value::Null).await;

    let store = LogStore::open(&data_dir).unwrap();
    let types: Vec<String> = store
        .read_from(&id, 0)
        .unwrap()
        .iter()
        .map(|e| e.event_type.clone())
        .collect();
    assert_eq!(types, vec!["instance_created", "composition"]);
}

#[tokio::test]
async fn a_bare_clone_is_told_that_nothing_is_composed_rather_than_told_nothing() {
    // C-4: presence-gated, and this is what presence-gating means HERE — not
    // "stay silent" but "say what is actually the case". A session given no
    // composition event cannot tell "nothing is composed" from "the engine
    // never said", and the second is a reason to go looking for what isn't
    // there. Same argument as `truncated` on a listing.
    let attach = create_and_attach(app_over(|_root| {})).await;
    assert_eq!(attach["headSeq"], 2);
}

#[tokio::test]
async fn an_unreadable_boot_source_still_creates_the_instance() {
    // Skip-not-fail: an instance that works without identity beats an
    // instance that cannot be created (KeylessProvider's posture).
    let dir = tempfile::tempdir().unwrap();
    let root = dir.keep();
    let data_dir = tempfile::tempdir().unwrap().keep();
    let store = LogStore::open(&data_dir).unwrap();
    let router = app(AppState {
        root: root.canonicalize().unwrap(),
        sessions: Arc::new(Sessions::new(store)),
        model: model_config(),
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: Some("does-not-exist.md".to_string()),
    });
    let attach = create_and_attach(router).await;
    assert_boot_loaded_nothing(&root.canonicalize().unwrap(), &data_dir, &attach);
}

#[tokio::test]
async fn a_non_utf8_boot_source_is_skipped_not_appended() {
    // Appending a boot event over bytes that don't decode would turn a
    // benign misconfiguration into every future turn on this stream failing
    // forever (projection.rs's String::from_utf8 failure routes to
    // fail_turn, not to drift-recovery) — skip-never-fail demands this is
    // caught before the append, not discovered at first use.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.keep();
    std::fs::write(root.join("boot.md"), [0xff, 0xfe, 0x00, 0xff]).unwrap();
    let data_dir = tempfile::tempdir().unwrap().keep();
    let store = LogStore::open(&data_dir).unwrap();
    let router = app(AppState {
        root: root.canonicalize().unwrap(),
        sessions: Arc::new(Sessions::new(store)),
        model: model_config(),
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: Some("boot.md".to_string()),
    });
    let attach = create_and_attach(router).await;
    assert_boot_loaded_nothing(&root.canonicalize().unwrap(), &data_dir, &attach);
}

#[tokio::test]
async fn a_boot_path_declared_absolute_is_refused_not_relativized() {
    // The manifest documents `[router] boot` as relative to root. An
    // absolute path that happens to sit inside root would still pass
    // resolve_boot_path's containment check — refused here instead, rather
    // than silently rewritten, so a misconfiguration doesn't hide.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.keep();
    let root = root.canonicalize().unwrap();
    std::fs::write(root.join("boot.md"), "# hello\n").unwrap();
    let data_dir = tempfile::tempdir().unwrap().keep();
    let store = LogStore::open(&data_dir).unwrap();
    let router = app(AppState {
        root: root.clone(),
        sessions: Arc::new(Sessions::new(store)),
        model: model_config(),
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: Some(root.join("boot.md").to_string_lossy().to_string()),
    });
    let attach = create_and_attach(router).await;
    assert_boot_loaded_nothing(&root.canonicalize().unwrap(), &data_dir, &attach);
}

#[tokio::test]
async fn a_boot_path_escaping_root_is_skipped_not_honored() {
    // A REAL containment escape — mirroring projection.rs's
    // `boot_path_escaping_root_is_boot_unreadable` — not merely a missing
    // file. A file that exists in a SIBLING tempdir, reached via
    // `../<sibling-name>/secret.md`: a nonexistent `../escaped.md` would
    // fail at `canonicalize()` before the `starts_with` containment check
    // in `resolve_boot_path` is ever reached, making that version of this
    // test pass even with the containment check deleted entirely.
    let outside = tempfile::tempdir().unwrap();
    let outside_root = outside.keep();
    std::fs::write(outside_root.join("secret.md"), b"nope").unwrap();
    let outside_name = outside_root.file_name().unwrap().to_string_lossy().to_string();
    let escape_path = format!("../{outside_name}/secret.md");

    let dir = tempfile::tempdir().unwrap();
    let root = dir.keep();
    let data_dir = tempfile::tempdir().unwrap().keep();
    let store = LogStore::open(&data_dir).unwrap();
    let router = app(AppState {
        root: root.canonicalize().unwrap(),
        sessions: Arc::new(Sessions::new(store)),
        model: model_config(),
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: Some(escape_path),
    });
    let attach = create_and_attach(router).await;
    assert_boot_loaded_nothing(&root.canonicalize().unwrap(), &data_dir, &attach);
}

#[tokio::test]
async fn an_instance_records_whether_it_may_write_the_composition() {
    // WD-9, and this test exists to close a SEAM rather than to check a field.
    // The route WRITES `mayWriteComposition` into `instance_created.data` and
    // `loop_::may_write_composition` READS it back; each is correct alone and
    // the pair fails silently if they disagree about the key — which is the
    // shape of defect this repo has already shipped once. So the reader is run
    // against events the writer actually produced.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.keep().canonicalize().unwrap();
    let data_dir = tempfile::tempdir().unwrap().keep();
    let store = LogStore::open(&data_dir).unwrap();
    let router = armillary_engine::app(AppState {
        root,
        sessions: Arc::new(Sessions::new(store)),
        model: model_config(),
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: None,
    });

    let (status, created) = post_json(
        router.clone(),
        "/instances",
        serde_json::json!({ "mayWriteComposition": true }),
        Some(TEST_TOKEN),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(created["mayWriteComposition"], true);

    let id = created["id"].as_str().unwrap().to_string();
    let store = LogStore::open(&data_dir).unwrap();
    let events = store.read_from(&id, 0).unwrap();
    assert!(
        armillary_engine::loop_::may_write_composition(&events),
        "the loop's reader did not see the grant the route wrote"
    );

    // Default OFF, asserted rather than assumed — a grant that defaults on is
    // the whole protection gone.
    let (_, plain) = post_json(router.clone(), "/instances", serde_json::json!({}), Some(TEST_TOKEN)).await;
    assert_eq!(plain["mayWriteComposition"], false);

    let plain_id = plain["id"].as_str().unwrap().to_string();
    let store = LogStore::open(&data_dir).unwrap();
    let plain_events = store.read_from(&plain_id, 0).unwrap();
    assert!(!armillary_engine::loop_::may_write_composition(&plain_events));
}

#[tokio::test]
async fn create_records_the_requested_model_and_returns_it() {
    let router = app_over(|_| {});
    let (status, created) = post_json(
        router,
        "/instances",
        serde_json::json!({ "operator": "tycho", "model": "zen/deepseek-v4-flash" }),
        Some(TEST_TOKEN),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(created["model"], "zen/deepseek-v4-flash");
}

#[tokio::test]
async fn create_without_a_model_is_accepted_and_reports_none() {
    let router = app_over(|_| {});
    let (status, created) =
        post_json(router, "/instances", serde_json::json!({ "operator": "tycho" }), Some(TEST_TOKEN)).await;
    assert_eq!(status, StatusCode::CREATED);
    assert!(created["model"].is_null());
}

/// Fix 4: `{"model": ""}` must not survive into the log as `"model": ""`
/// while every read reports `null` for the same instance — normalization
/// happens on the WRITE path, not only in the two readers'
/// `.filter(|s| !s.is_empty())` guards. Read the raw event back (not just
/// the create response) so a regression that moved the filter back to a
/// reader-only guard would still be caught.
#[tokio::test]
async fn create_with_an_empty_model_string_is_normalized_to_none_in_the_log() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.keep().canonicalize().unwrap();
    let data_dir = tempfile::tempdir().unwrap().keep();
    let store = LogStore::open(&data_dir).unwrap();
    let router = armillary_engine::app(AppState {
        root,
        sessions: Arc::new(Sessions::new(store)),
        model: model_config(),
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
        registry_dir: full_grant_registry(),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: None,
    });

    let (status, created) = post_json(
        router,
        "/instances",
        serde_json::json!({ "operator": "tycho", "model": "" }),
        Some(TEST_TOKEN),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    assert!(created["model"].is_null());

    let id = created["id"].as_str().unwrap().to_string();
    let store = LogStore::open(&data_dir).unwrap();
    let events = store.read_from(&id, 0).unwrap();
    let instance_created = events
        .iter()
        .find(|e| e.event_type == "instance_created")
        .expect("instance_created must be the first event");
    assert!(
        instance_created.data.get("model").unwrap().is_null(),
        "the log recorded {:?}, not null — write path did not normalize the empty string",
        instance_created.data.get("model")
    );
}

#[tokio::test]
async fn models_reports_an_empty_catalog_rather_than_failing() {
    // `app_over` already points `models_path` at `/nonexistent/models.toml` —
    // a guaranteed-absent path, exactly "no models.toml anywhere".
    let app = app_over(|_| {});

    let response = app
        .oneshot(Request::builder().uri("/models").body(Body::empty()).unwrap())
        .await
        .unwrap();

    // 200 with nothing in it, never a 500: a host that has not written the
    // file is a working host, and the app's fallback depends on this.
    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), ONE_MIB).await.unwrap(),
    )
    .unwrap();
    assert_eq!(body["models"].as_array().unwrap().len(), 0);
}

/// Like `app_over`, but differs only in the three fields Task 4 reads:
/// `models_path` points at a REAL file (`app_over`'s literal is deliberately
/// absent), and the two key-presence booleans are set explicitly rather than
/// always `false` — the only way to exercise `usable` for both a keyed and a
/// keyless provider in the same request.
fn app_with_models(root: &Path, models_path: &Path, anthropic: bool, zen: bool) -> axum::Router {
    let root = root.canonicalize().unwrap();
    let data_dir = tempfile::tempdir().unwrap().keep();
    let store = LogStore::open(&data_dir).unwrap();

    app(AppState {
        root,
        hostname: "test-host".to_string(),
        sessions: Arc::new(Sessions::new(store)),
        model: model_config(),
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path: models_path.to_path_buf(),
        registry_dir: full_grant_registry(),
        anthropic_key_present: anthropic,
        zen_key_present: zen,
        boot: None,
    })
}

#[tokio::test]
async fn a_model_whose_provider_has_no_key_is_reported_unusable() {
    let dir = tempfile::tempdir().unwrap();
    let models_path = dir.path().join("models.toml");
    std::fs::write(
        &models_path,
        "[[model]]\nid = \"claude-sonnet-5\"\n\n[[model]]\nid = \"zen/deepseek-v4-flash\"\n",
    )
    .unwrap();
    // A host with an Anthropic key and no Zen key.
    let app = app_with_models(dir.path(), &models_path, true, false);

    let response = app
        .oneshot(Request::builder().uri("/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), ONE_MIB).await.unwrap(),
    )
    .unwrap();

    assert_eq!(body["models"][0]["provider"], "anthropic");
    assert_eq!(body["models"][0]["usable"], true);
    assert_eq!(body["models"][1]["provider"], "zen");
    assert_eq!(body["models"][1]["usable"], false);
}

/// Fix 2: `/models`' `default` must be the engine's EFFECTIVE default — what
/// `AppState.model.model` was resolved to once at boot — not a fresh re-read
/// of the catalog file's `default` line. Here the two are made to disagree
/// on purpose (as `--model` overriding the file, or a post-boot edit to the
/// file, would in real use) so a regression that swaps back to
/// `catalog.default` fails loudly.
#[tokio::test]
async fn models_default_is_the_boot_resolved_value_not_a_fresh_catalog_read() {
    let dir = tempfile::tempdir().unwrap();
    let models_path = dir.path().join("models.toml");
    std::fs::write(&models_path, "default = \"claude-sonnet-5\"\n").unwrap();

    let root = dir.path().canonicalize().unwrap();
    let data_dir = tempfile::tempdir().unwrap().keep();
    let store = LogStore::open(&data_dir).unwrap();
    let app = app(AppState {
        root,
        hostname: "test-host".to_string(),
        sessions: Arc::new(Sessions::new(store)),
        model: ModelConfig {
            model: "claude-opus-5".to_string(),
        },
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path,
        registry_dir: full_grant_registry(),
        anthropic_key_present: true,
        zen_key_present: false,
        boot: None,
    });

    let (status, body) = get_json(app, "/models").await;
    assert_eq!(status, StatusCode::OK);
    // The boot-resolved value wins, even though the file on disk still says
    // "claude-sonnet-5" — the endpoint and the resolver agree by construction.
    assert_eq!(body["default"], "claude-opus-5");
}

/// Fix 8: the branch's highest-value missing test. `models.rs`'s unit tests
/// cover the PARSE layer; nothing asserted the SERIALIZED wire shape of the
/// two fields the app actually reads off `GET /models`. A `skip_serializing_if`
/// or a field rename would go green on both the parse tests and the
/// `provider`/`usable` assertions above, and break only against a real app.
#[tokio::test]
async fn models_default_and_label_survive_serialization_under_their_wire_names() {
    let dir = tempfile::tempdir().unwrap();
    let models_path = dir.path().join("models.toml");
    std::fs::write(
        &models_path,
        "default = \"claude-sonnet-5\"\n\n[[model]]\nid = \"claude-sonnet-5\"\nlabel = \"Sonnet 5\"\n",
    )
    .unwrap();
    let app = app_with_models(dir.path(), &models_path, true, false);

    let (status, body) = get_json(app, "/models").await;
    assert_eq!(status, StatusCode::OK);
    // `default` at the top level, `label` on the entry — exact JSON keys, not
    // just "some truthy field survived".
    assert_eq!(body["default"], "claude-sonnet-5");
    assert_eq!(body["models"][0]["label"], "Sonnet 5");
}
