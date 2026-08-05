//! Route-level tests for `/repos` against an in-process axum app.
//!
//! Real git fixtures, not the fake `.git`-directory stand-ins `repos.rs`'s
//! OWN unit tests use for `declared_modules`: these routes read `git log`
//! and `git status`, which need an actual repository to answer anything.
//! `armillary_engine::testgit` is `#[cfg(test)]`-gated on the LIBRARY crate,
//! so it is invisible from here — an integration test binary links the
//! library built WITHOUT that cfg — and these are a self-contained
//! second copy of exactly the fixtures this task needs, not the whole file.

use armillary_engine::{
    app,
    log::store::LogStore,
    provider::KeylessProvider,
    sessions::Sessions,
    state::{AppState, ModelConfig},
};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tower::ServiceExt;

/// Run git synchronously for test setup, isolated from the machine's own git
/// config — no global `user.email`, no inherited `commit.gpgsign`, no
/// `init.defaultBranch` drift. Mirrors `armillary_engine::testgit::git_sync`.
fn git_sync(dir: &Path, args: &[&str]) {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .env("GIT_AUTHOR_NAME", "test")
        .env("GIT_AUTHOR_EMAIL", "test@example.invalid")
        .env("GIT_COMMITTER_NAME", "test")
        .env("GIT_COMMITTER_EMAIL", "test@example.invalid")
        .output()
        .expect("git must be on PATH for these tests");
    assert!(
        out.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
}

fn commit(repo: &Path, name: &str, body: &str) {
    std::fs::write(repo.join(name), body).unwrap();
    git_sync(repo, &["add", name]);
    git_sync(repo, &["commit", "-m", name]);
}

/// Corrupt the loose object HEAD currently points to, in place. Mirrors
/// `armillary_engine::testgit::corrupt_head_object`, unreachable from here
/// (see this file's header) — used to prove a genuine read failure 502s
/// rather than reading as "no commits yet".
fn corrupt_head_object(repo: &Path) {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("git must be on PATH for these tests");
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert_eq!(sha.len(), 40, "expected a full sha from rev-parse HEAD, got {sha:?}");

    let path = repo.join(".git/objects").join(&sha[..2]).join(&sha[2..]);
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    perms.set_readonly(false);
    std::fs::set_permissions(&path, perms).unwrap();
    std::fs::write(&path, b"not a valid zlib stream").unwrap();
}

/// A bare remote with one commit, plus a clone of it: `(remote, clone)`.
fn remote_and_clone() -> (PathBuf, PathBuf) {
    let remote = tempfile::tempdir().unwrap().keep();
    git_sync(&remote, &["init", "--bare", "--initial-branch=main", "."]);

    let seed = tempfile::tempdir().unwrap().keep();
    git_sync(&seed, &["init", "--initial-branch=main", "."]);
    commit(&seed, "seed.md", "one");
    git_sync(&seed, &["remote", "add", "origin", remote.to_str().unwrap()]);
    git_sync(&seed, &["push", "-u", "origin", "main"]);

    let clone = tempfile::tempdir().unwrap().keep();
    git_sync(&clone, &["clone", remote.to_str().unwrap(), clone.to_str().unwrap()]);
    (remote, clone)
}

/// A workspace whose declared modules are REAL clones of a shared bare
/// remote, with `[router] sync = true` so `enabled` reads `true`. `push` is
/// deliberately left undeclared, so `push_enabled` reads `false` — the
/// independent-gates property `get_repos_reports_both_gates...` checks.
/// Returns `(root, remote)`.
fn live_workspace_with_sync() -> (PathBuf, PathBuf) {
    let root = tempfile::tempdir().unwrap().keep();
    git_sync(&root, &["init", "--initial-branch=main", "."]);

    let (remote, _first) = remote_and_clone();
    let module = root.join("repos").join("jianyi");
    std::fs::create_dir_all(root.join("repos")).unwrap();
    git_sync(&root, &["clone", remote.to_str().unwrap(), module.to_str().unwrap()]);

    std::fs::write(
        root.join("modules.local.toml"),
        "[router]\nsync = true\n\n\
         [[repos]]\nname = \"jianyi\"\npath = \"repos/jianyi\"\n",
    )
    .unwrap();
    (root, remote)
}

/// `live_workspace_with_sync`, but with NO `[router]` gate declared at all —
/// neither `sync` nor `push`. Every verb must refuse against this fixture;
/// it is the "nothing granted" half of the two-gate story.
fn live_workspace() -> (PathBuf, PathBuf) {
    let root = tempfile::tempdir().unwrap().keep();
    git_sync(&root, &["init", "--initial-branch=main", "."]);

    let (remote, _first) = remote_and_clone();
    let module = root.join("repos").join("jianyi");
    std::fs::create_dir_all(root.join("repos")).unwrap();
    git_sync(&root, &["clone", remote.to_str().unwrap(), module.to_str().unwrap()]);

    std::fs::write(
        root.join("modules.local.toml"),
        "[[repos]]\nname = \"jianyi\"\npath = \"repos/jianyi\"\n",
    )
    .unwrap();
    (root, remote)
}

/// Push a new commit into `remote` from a third clone, so an existing clone
/// becomes genuinely behind rather than being told it is. Mirrors
/// `armillary_engine::testgit::advance_remote`, unreachable from here (see
/// this file's header).
fn advance_remote(remote: &Path) {
    let other = tempfile::tempdir().unwrap().keep();
    git_sync(&other, &["clone", remote.to_str().unwrap(), other.to_str().unwrap()]);
    commit(&other, "from-elsewhere.md", "two");
    git_sync(&other, &["push", "origin", "main"]);
}

fn build_app(root: &Path) -> axum::Router {
    let data_dir = tempfile::tempdir().unwrap().keep();
    let store = LogStore::open(&data_dir).unwrap();
    app(AppState {
        root: root.canonicalize().unwrap(),
        sessions: Arc::new(Sessions::new(store)),
        model: ModelConfig { model: "claude-sonnet-5".to_string(), api_key: None },
        provider: Arc::new(KeylessProvider),
        boot: None,
    })
}

async fn get_json(root: &Path, uri: &str) -> serde_json::Value {
    let response = build_app(root)
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(response.into_body(), 8 * 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

async fn get_status(root: &Path, uri: &str) -> u16 {
    let response = build_app(root)
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    response.status().as_u16()
}

async fn post_json(root: &Path, uri: &str) -> serde_json::Value {
    let response = build_app(root)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(response.into_body(), 8 * 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

async fn post_status(root: &Path, uri: &str) -> u16 {
    let response = build_app(root)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    response.status().as_u16()
}

#[tokio::test]
async fn get_repos_reports_both_gates_and_omits_newest_commit() {
    let (root, _remote) = live_workspace_with_sync();
    let body = get_json(&root, "/repos").await;
    assert_eq!(body["enabled"], true);
    assert_eq!(body["push_enabled"], false);
    // The router root itself has no commits in this fixture, so its own
    // `newest_commit` would read `None` either way — checking `jianyi`
    // specifically is what actually discriminates `with_commit: false` from
    // `true`, since jianyi has real history to omit or carry.
    let jianyi = body["repos"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "jianyi")
        .expect("jianyi must be in the list");
    assert!(jianyi.get("newest_commit").is_none());
}

#[tokio::test]
async fn get_one_repo_by_name_carries_newest_commit() {
    let (root, _remote) = live_workspace_with_sync();
    let body = get_json(&root, "/repos/jianyi").await;
    assert_eq!(body["name"], "jianyi");
    assert!(body["newest_commit"].is_string());
}

#[tokio::test]
async fn an_undeclared_name_is_404_not_500() {
    let (root, _remote) = live_workspace_with_sync();
    assert_eq!(get_status(&root, "/repos/../../etc").await, StatusCode::NOT_FOUND.as_u16());
    assert_eq!(get_status(&root, "/repos/nope").await, StatusCode::NOT_FOUND.as_u16());
}

#[tokio::test]
async fn the_log_marks_unpushed_commits() {
    let (root, _remote) = live_workspace_with_sync();
    commit(&root.join("repos/jianyi"), "mine.md", "unpushed");
    let body = get_json(&root, "/repos/jianyi/log?limit=10").await;
    assert_eq!(body[0]["unpushed"], true, "the newest commit is ahead of upstream");
    assert_eq!(body[1]["unpushed"], false);
}

#[tokio::test]
async fn changes_lists_untracked_and_modified_files() {
    let (root, _remote) = live_workspace_with_sync();
    let repo = root.join("repos/jianyi");
    std::fs::write(repo.join("seed.md"), "edited").unwrap();
    std::fs::write(repo.join("new.md"), "fresh").unwrap();
    let body = get_json(&root, "/repos/jianyi/changes").await;
    let kinds: Vec<&str> =
        body.as_array().unwrap().iter().map(|f| f["change"].as_str().unwrap()).collect();
    assert!(kinds.contains(&"modified"));
    assert!(kinds.contains(&"untracked"));
}

#[tokio::test]
async fn log_limit_is_applied_to_the_request() {
    // A body-COUNT assertion, not a status-code one: the fixture has 6
    // commits regardless of what is requested, so a status-only check here
    // would pass identically whether the route's clamp/limit plumbing is
    // wired up or deleted outright. Requesting 2 must return exactly 2.
    let (root, _remote) = live_workspace_with_sync();
    let repo = root.join("repos/jianyi");
    for i in 0..5 {
        commit(&repo, &format!("f{i}.md"), "x");
    }
    let body = get_json(&root, "/repos/jianyi/log?limit=2").await;
    assert_eq!(body.as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn a_malformed_limit_falls_back_to_the_default_rather_than_400() {
    // The defect this closes: axum's derived `Query<LogQuery>` extractor
    // 400s on a present-but-unparseable value when the field type is
    // `Option<u32>`, because `Option` only defaults a MISSING key, not a
    // malformed one. `clamp_limit`'s own unit tests (`routes/repos.rs`)
    // cover the parsing directly; this proves the route wiring actually
    // reaches the request at all, over real HTTP.
    let (root, _remote) = live_workspace_with_sync();
    assert_eq!(get_status(&root, "/repos/jianyi/log?limit=abc").await, 200);
    assert_eq!(get_status(&root, "/repos/jianyi/log?limit=-1").await, 200);
}

#[tokio::test]
async fn pull_fast_forwards_and_returns_the_new_state() {
    let (root, remote) = live_workspace_with_sync();
    advance_remote(&remote);
    post_json(&root, "/repos/jianyi/fetch").await;
    let body = post_json(&root, "/repos/jianyi/pull").await;

    assert!(root.join("repos/jianyi/from-elsewhere.md").exists());
    assert_eq!(body["position"]["behind"], 0, "the response must be POST-mutation state");
}

#[tokio::test]
async fn pull_refuses_a_dirty_repo_and_leaves_the_working_tree_alone() {
    let (root, remote) = live_workspace_with_sync();
    advance_remote(&remote);
    let seed = root.join("repos/jianyi/seed.md");
    std::fs::write(&seed, "my uncommitted edit").unwrap();
    post_json(&root, "/repos/jianyi/fetch").await;
    let body = post_json(&root, "/repos/jianyi/pull").await;

    assert_eq!(std::fs::read_to_string(&seed).unwrap(), "my uncommitted edit");
    assert!(!root.join("repos/jianyi/from-elsewhere.md").exists());
    // A dirty-tree refusal is a typed POLICY error, told apart from a
    // transport or fast-forward failure without string-matching the message.
    assert_eq!(body["action_error"]["kind"], "dirty");
}

#[tokio::test]
async fn pull_on_a_diverged_branch_reports_not_fast_forwardable() {
    // pull_ff never touches the network — a diverged branch is a git-side
    // fast-forward refusal, not a transport failure, and the kind must say so.
    let (root, remote) = live_workspace_with_sync();
    advance_remote(&remote);
    commit(&root.join("repos/jianyi"), "local-only.md", "mine");
    post_json(&root, "/repos/jianyi/fetch").await;
    let body = post_json(&root, "/repos/jianyi/pull").await;
    assert_eq!(body["action_error"]["kind"], "not-fast-forwardable");
}

#[tokio::test]
async fn push_is_403_when_only_sync_is_granted() {
    let (root, _remote) = live_workspace_with_sync(); // sync = true, no push key
    assert_eq!(post_status(&root, "/repos/jianyi/push").await, 403);
}

#[tokio::test]
async fn every_verb_is_403_when_nothing_is_granted() {
    let (root, _remote) = live_workspace(); // no gates at all
    for path in ["/repos/jianyi/fetch", "/repos/jianyi/pull", "/repos/jianyi/push", "/repos/fetch"] {
        assert_eq!(post_status(&root, path).await, 403, "{path} must refuse");
    }
}

#[tokio::test]
async fn ahead_survives_to_the_wire() {
    // THE regression test for the founding bug, taken all the way to the
    // SERDE layer rather than stopping at the Rust struct (repos.rs's OWN
    // `ahead_survives_to_the_state` asserts on `Position` directly, which a
    // `#[serde(rename)]` or `skip_serializing_if` on `Tracking`'s `ahead`
    // could defeat while leaving every existing test green). A repo with N
    // unpushed commits and nothing incoming must serialize `ahead: N` on the
    // actual HTTP response body.
    let (root, _remote) = live_workspace_with_sync();
    let repo = root.join("repos/jianyi");
    for i in 0..3 {
        commit(&repo, &format!("unpushed{i}.md"), "x");
    }
    let body = get_json(&root, "/repos/jianyi").await;
    assert_eq!(body["position"]["kind"], "tracking", "expected a Tracking position");
    assert_eq!(body["position"]["ahead"], 3, "ahead must reach the wire, not merely the struct");
}

#[tokio::test]
async fn a_corrupt_repo_502s_the_log_route_rather_than_reading_as_no_commits() {
    // The whole-branch review's second-worst finding: a repo with a corrupt
    // loose object exits 128 from `git log`, and the SAME shape (empty
    // stdout, nonzero exit) an unborn branch produces. Read naively, that
    // reads a genuine read failure as "no commits yet" — `200 []` from
    // `/log`, alongside `read_error` from `/repos/{name}` and `502` from
    // `/changes` on the identical repo.
    let (root, _remote) = live_workspace_with_sync();
    corrupt_head_object(&root.join("repos/jianyi"));
    assert_eq!(
        get_status(&root, "/repos/jianyi/log?limit=10").await,
        StatusCode::BAD_GATEWAY.as_u16(),
        "a corrupt repo must not report as having no commits"
    );
}

#[tokio::test]
async fn a_total_fetch_failure_does_not_read_as_success() {
    // The whole-branch review's worst finding on the shipped branch: the
    // screen reported every repo `current` having contacted nothing.
    let (root, _remote) = live_workspace_with_sync();
    git_sync(&root.join("repos/jianyi"), &["remote", "remove", "origin"]);
    let body = post_json(&root, "/repos/fetch").await;
    let jianyi = body
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "jianyi")
        .unwrap();
    assert!(jianyi["action_error"]["message"].is_string(), "a failed fetch must be on the wire");
    assert_eq!(jianyi["action_error"]["kind"], "transport");
}
