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
    provider::{self, KeylessProvider},
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

/// Create a linked worktree off `repo`, then delete its gitdir under
/// `.git/worktrees/` so the checkout is left pointing at nothing. Returns the
/// path to the now-orphaned worktree. Mirrors
/// `armillary_engine::testgit::stale_linked_worktree`, unreachable from here
/// (see this file's header) — used to prove a repo git cannot OPEN at all
/// (not merely one with a corrupt object) still 502s the log route rather
/// than reading as "no commits yet".
fn stale_linked_worktree(repo: &Path) -> PathBuf {
    let wt = repo.join(".worktrees").join("stale");
    git_sync(repo, &["worktree", "add", wt.to_str().unwrap(), "-b", "stale-topic"]);
    let name = wt.file_name().unwrap().to_str().unwrap();
    std::fs::remove_dir_all(repo.join(".git/worktrees").join(name)).unwrap();
    wt
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
        model: ModelConfig { model: "claude-sonnet-5".to_string() },
        providers: provider::fixed(Arc::new(KeylessProvider)),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        registry_dir: std::path::PathBuf::from("/nonexistent/registry"),
        anthropic_key_present: false,
        zen_key_present: false,
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
async fn pull_under_a_gone_upstream_reports_not_fast_forwardable_naming_the_upstream() {
    // The previously unpinned arm (survey 2026-08-06, weakness #3): the
    // remote branch is deleted and pruned, `@{u}` no longer resolves, and the
    // merge fails before it begins. The kind stays in the closed vocabulary —
    // "not-fast-forwardable" is what a shipped client can already render.
    // Measured writing this test: git's own message is the cryptic
    // "merge: @{u} - not something we can merge" (C locale), naming nothing —
    // so the HUMAN-readable half of the answer is the same response's
    // read-side `position`, which discriminates upstream-gone and names the
    // branch. The pin is the response as a whole, not the message.
    let (root, remote) = live_workspace_with_sync();
    git_sync(&remote, &["branch", "-m", "main", "elsewhere"]);
    git_sync(&root.join("repos/jianyi"), &["fetch", "--prune"]);
    let body = post_json(&root, "/repos/jianyi/pull").await;

    assert_eq!(body["action_error"]["kind"], "not-fast-forwardable");
    assert_eq!(body["position"]["kind"], "upstream-gone");
    assert_eq!(body["position"]["upstream"], "origin/main");
}

#[tokio::test]
async fn push_is_403_when_only_sync_is_granted() {
    let (root, _remote) = live_workspace_with_sync(); // sync = true, no push key
    assert_eq!(post_status(&root, "/repos/jianyi/push").await, 403);
}

/// Grant `push = true` on top of `live_workspace_with_sync`'s fixture.
fn grant_push(root: &Path) {
    std::fs::write(
        root.join("modules.local.toml"),
        "[router]\nsync = true\npush = true\n\n\
         [[repos]]\nname = \"jianyi\"\npath = \"repos/jianyi\"\n",
    )
    .unwrap();
}

#[tokio::test]
async fn push_sends_the_local_commit_to_the_remote() {
    // THE founding-bug shape, on this branch's widest authority: a handler
    // that resolved, passed the gate, and returned read_one(..., true)
    // WITHOUT ever calling git::push would pass every other test in this
    // suite — a well-formed RepoState with ahead still on it, the client
    // showing the push as done, and the commit sitting local. Verified from
    // a THIRD clone of the remote, never from the pusher's own refs (those
    // move locally whether or not the remote accepted anything).
    let (root, remote) = live_workspace_with_sync();
    grant_push(&root);
    commit(&root.join("repos/jianyi"), "mine.md", "local work");

    let body = post_json(&root, "/repos/jianyi/push").await;
    assert!(body["action_error"].is_null(), "a clean push must not carry an error");

    let verify = tempfile::tempdir().unwrap().keep();
    git_sync(&verify, &["clone", remote.to_str().unwrap(), verify.to_str().unwrap()]);
    assert!(verify.join("mine.md").exists(), "the remote never received the pushed commit");
}

#[tokio::test]
async fn push_on_a_diverged_branch_reports_the_error_and_does_not_land() {
    let (root, remote) = live_workspace_with_sync();
    grant_push(&root);
    advance_remote(&remote);
    commit(&root.join("repos/jianyi"), "mine.md", "local work");
    // Deliberately no fetch first — pushing straight into a remote that has
    // moved is exactly the diverged case `push` must refuse, not force.

    let body = post_json(&root, "/repos/jianyi/push").await;
    assert!(body["action_error"]["message"].is_string(), "the error must be on the wire");
    assert_eq!(body["action_error"]["kind"], "not-fast-forwardable");

    let verify = tempfile::tempdir().unwrap().keep();
    git_sync(&verify, &["clone", remote.to_str().unwrap(), verify.to_str().unwrap()]);
    assert!(!verify.join("mine.md").exists(), "a refused push must not land on the remote");
}

/// Install a `pre-receive` hook in a bare remote that unconditionally
/// declines. The one-line reproduction for N2: a hosted remote's protected
/// branch, or any policy gate, refuses a push through exactly this
/// mechanism, and git prints a DIFFERENT literal (`"! [remote rejected]"`)
/// for it than for an ordinary non-fast-forward (`"! [rejected]"`).
fn install_declining_pre_receive_hook(remote: &Path) {
    let hooks = remote.join("hooks");
    std::fs::create_dir_all(&hooks).unwrap();
    let hook = hooks.join("pre-receive");
    std::fs::write(&hook, "#!/bin/sh\nexit 1\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(&hook).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&hook, perms).unwrap();
}

#[tokio::test]
async fn push_declined_by_a_pre_receive_hook_is_refused_by_remote_not_transport() {
    // N2: a protected-branch or policy refusal must not read as "the remote
    // could not be reached" — that is the false story `action_error` exists
    // to end, on the branch that grants push authority. The remote WAS
    // reached; it deliberately declined.
    let (root, remote) = live_workspace_with_sync();
    grant_push(&root);
    install_declining_pre_receive_hook(&remote);
    commit(&root.join("repos/jianyi"), "mine.md", "local work");

    let body = post_json(&root, "/repos/jianyi/push").await;
    assert!(body["action_error"]["message"].is_string(), "the error must be on the wire");
    assert_eq!(body["action_error"]["kind"], "refused-by-remote");
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
async fn a_stale_linked_worktree_502s_the_log_route_rather_than_reading_as_no_commits() {
    // N1: a repo git cannot OPEN at all — not merely one with a corrupt
    // object — is a different trigger for the same defect. This workspace
    // runs linked worktrees routinely, and `declared_modules` admits a
    // gitfile, so a stale linked worktree (its gitdir removed out from under
    // it) is not exotic here. `git log` exits 128 with empty stdout, the
    // same shape an unborn branch produces, so this must not read as "no
    // commits yet" either.
    let (root, _remote) = live_workspace_with_sync();
    let orphan = stale_linked_worktree(&root.join("repos/jianyi"));
    let rel = orphan.strip_prefix(&root).unwrap().to_str().unwrap().replace('\\', "/");
    std::fs::write(
        root.join("modules.local.toml"),
        format!(
            "[router]\nsync = true\n\n\
             [[repos]]\nname = \"jianyi\"\npath = \"repos/jianyi\"\n\n\
             [[repos]]\nname = \"stale\"\npath = \"{rel}\"\n"
        ),
    )
    .unwrap();

    assert_eq!(
        get_status(&root, "/repos/stale/log?limit=10").await,
        StatusCode::BAD_GATEWAY.as_u16(),
        "a repo git cannot open must not report as having no commits"
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
