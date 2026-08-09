//! Runs `conformance/log/replay/` as data — the log replay/gap-detection
//! suite `conformance/README.md` named as "wanted next" (constitution
//! `instances.md` A-1/A-3).
//!
//! Table-driven on the `crates/armillary-composition/tests/conformance.rs`
//! pattern: adding a fixture here requires no code change, and the suite
//! asserts it discovered at least one fixture so a bad glob can't pass
//! silently.
//!
//! **The inclusive/exclusive seam, named once, here:** the fixture's `from`
//! (`conformance/log/README.md`'s contract) is the client's cursor and is
//! EXCLUSIVE — the last seq the client has already seen, so the first seq
//! it actually needs is `from + 1`. The store's read primitive is INCLUSIVE
//! of the seq it is given (it returns every event with `seq >= from_seq`).
//! `replay_from = from + 1` below is the whole adapter between those two
//! conventions — it is not a redefinition of either one.
//!
//! A second test (`every_fixture_event_validates_against_the_schema` /
//! `every_event_a_scripted_turn_emits_validates_against_the_schema`, in this
//! same file) schema-validates every fixture line AND every event a real
//! scripted turn emits against `schema/event.schema.json`, via the
//! `jsonschema` crate — a DEV-only dependency (see `Cargo.toml`): the seam
//! stays intact because `schema/` and `conformance/` describe the shape and
//! this crate merely proves it emits that shape, never the other way round.

use armillary_engine::log::envelope::{Actor, Role};
use armillary_engine::log::store::LogStore;
use armillary_engine::loop_::run_turn;
use armillary_engine::provider::{self, ScriptedProvider};
use armillary_engine::sessions::{NewEvent, Sessions};
use armillary_engine::state::{AppState, ModelConfig, SharedState};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::sync::watch;

fn replay_fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/log/replay")
        .canonicalize()
        .expect("conformance/log/replay must exist relative to the crate")
}

fn schema_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../schema/event.schema.json")
        .canonicalize()
        .expect("schema/event.schema.json must exist relative to the crate")
}

fn compile_event_schema() -> jsonschema::Validator {
    let text = fs::read_to_string(schema_path()).expect("read schema/event.schema.json");
    let schema: serde_json::Value =
        serde_json::from_str(&text).expect("schema/event.schema.json must parse as JSON");
    jsonschema::validator_for(&schema).expect("schema/event.schema.json must itself be a valid schema")
}

// --- replay/gap fixtures ---

#[derive(serde::Deserialize)]
struct ExpectedGap {
    #[serde(rename = "requestedFrom")]
    requested_from: u64,
    #[serde(rename = "earliestAvailable")]
    earliest_available: u64,
}

#[derive(serde::Deserialize)]
struct Expected {
    from: u64,
    replayed: Vec<String>,
    gap: Option<ExpectedGap>,
}

/// Copies `<stem>.jsonl` into a fresh temp data dir laid out the way the
/// store expects (`<data_dir>/streams/<stream>.jsonl`): every line in a
/// fixture already carries the same `stream` name (`conformance/log/README.md`),
/// so the whole "load" step is reading that name off the first line and
/// copying the file to where it belongs — this suite is about replay/gap
/// semantics, not re-proving what the writer already enforces at append time.
fn load_fixture_into_store(dir: &Path, stem: &str) -> (tempfile::TempDir, LogStore, String) {
    let jsonl_path = dir.join(format!("{stem}.jsonl"));
    let text = fs::read_to_string(&jsonl_path).unwrap_or_else(|e| panic!("read {stem}.jsonl: {e}"));
    let first_line = text
        .lines()
        .next()
        .unwrap_or_else(|| panic!("{stem}.jsonl must have at least one line"));
    let first: serde_json::Value = serde_json::from_str(first_line)
        .unwrap_or_else(|e| panic!("{stem}.jsonl's first line must be valid JSON: {e}"));
    let stream = first["stream"]
        .as_str()
        .unwrap_or_else(|| panic!("{stem}.jsonl's first event must carry a stream name"))
        .to_string();

    let data_dir = tempfile::tempdir().unwrap();
    let streams_dir = data_dir.path().join("streams");
    fs::create_dir_all(&streams_dir).unwrap();
    fs::copy(&jsonl_path, streams_dir.join(format!("{stream}.jsonl"))).unwrap();
    let store = LogStore::open(data_dir.path()).unwrap();
    (data_dir, store, stream)
}

#[test]
fn log_replay_fixtures_match_expected() {
    let dir = replay_fixtures_dir();
    let mut ok = 0;

    for entry in fs::read_dir(&dir).expect("read conformance/log/replay") {
        let path = entry.expect("dir entry").path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if !name.ends_with(".jsonl") {
            continue;
        }
        let stem = name.trim_end_matches(".jsonl").to_string();

        let expected_path = dir.join(format!("{stem}.expected.json"));
        let expected: Expected = serde_json::from_str(
            &fs::read_to_string(&expected_path)
                .unwrap_or_else(|e| panic!("fixture {stem} is missing its .expected.json: {e}")),
        )
        .unwrap_or_else(|e| panic!("fixture {stem}'s expected.json did not parse: {e}"));

        let (_data_dir, store, stream) = load_fixture_into_store(&dir, &stem);

        // ADAPTER: see the module doc — `from` is exclusive, `read_from` is
        // inclusive, so the first seq actually needed is `from + 1`.
        let replay_from = expected.from + 1;
        let replayed = store
            .read_from(&stream, replay_from)
            .unwrap_or_else(|e| panic!("fixture {stem}: read_from failed: {e}"));
        let replayed_ids: Vec<String> = replayed.iter().map(|e| e.id.clone()).collect();

        let earliest = store.earliest_seq(&stream).unwrap();
        let gap_present = earliest != 0 && replay_from < earliest;

        assert_eq!(
            replayed_ids, expected.replayed,
            "fixture {stem}: replayed ids did not match expected"
        );

        match &expected.gap {
            None => assert!(!gap_present, "fixture {stem}: expected no gap but one was detected (earliest {earliest}, replay_from {replay_from})"),
            Some(exp) => {
                assert!(gap_present, "fixture {stem}: expected a gap but none was detected");
                assert_eq!(expected.from, exp.requested_from, "fixture {stem}: gap.requestedFrom");
                assert_eq!(earliest, exp.earliest_available, "fixture {stem}: gap.earliestAvailable");
            }
        }

        ok += 1;
    }

    // Not ceremony — see crates/armillary-composition/tests/conformance.rs's
    // identical assertion: a glob that silently matches nothing reports
    // success, which is the exact failure mode this line rules out.
    assert!(ok > 0, "no replay fixtures were discovered — the glob matched nothing");
    eprintln!("conformance/log/replay: {ok} fixtures");
}

// --- schema validation: every fixture line, and every event a real turn emits ---

#[test]
fn every_fixture_event_validates_against_the_schema() {
    let validator = compile_event_schema();
    let dir = replay_fixtures_dir();
    let mut checked = 0;

    for entry in fs::read_dir(&dir).expect("read conformance/log/replay") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let text = fs::read_to_string(&path).unwrap();
        for (i, line) in text.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("{}:{}: not valid JSON: {e}", path.display(), i + 1));
            if let Err(e) = validator.validate(&value) {
                panic!("{}:{}: failed schema validation: {e}", path.display(), i + 1);
            }
            checked += 1;
        }
    }

    assert!(checked > 0, "no fixture events were schema-checked — the glob matched nothing");
    eprintln!("conformance/log/replay: {checked} fixture events schema-validated");
}

fn model_config() -> ModelConfig {
    ModelConfig {
        model: "scripted".to_string(),
    }
}

#[tokio::test]
async fn every_event_a_scripted_turn_emits_validates_against_the_schema() {
    let validator = compile_event_schema();

    let data_dir = tempfile::tempdir().unwrap();
    let root = tempfile::tempdir().unwrap();
    let store = LogStore::open(data_dir.path()).unwrap();
    let sessions = Arc::new(Sessions::new(store));

    let id = uuid::Uuid::new_v4().to_string();
    sessions
        .append(
            &id,
            NewEvent {
                actor: Actor {
                    role: Role::System,
                    instance: None,
                    principal: None,
                },
                event_type: "instance_created".to_string(),
                data: serde_json::json!({ "operator": "tycho" }),
            },
        )
        .unwrap();
    sessions
        .append(
            &id,
            NewEvent {
                actor: Actor {
                    role: Role::User,
                    instance: None,
                    principal: None,
                },
                event_type: "user_message".to_string(),
                data: serde_json::json!({ "text": "hi", "clientKey": "c1" }),
            },
        )
        .unwrap();

    let state: SharedState = Arc::new(AppState {
        root: root.path().canonicalize().unwrap(),
        sessions: sessions.clone(),
        model: model_config(),
        providers: provider::fixed(Arc::new(ScriptedProvider::new(vec!["Hel", "Hello there"]))),
        models_path: std::path::PathBuf::from("/nonexistent/models.toml"),
        hostname: "test-host".to_string(),
        registry_dir: std::path::PathBuf::from("/nonexistent/registry"),
        anthropic_key_present: false,
        zen_key_present: false,
        boot: None,
    });

    let generation = uuid::Uuid::new_v4().to_string();
    let (_cancel_tx, cancel_rx) = watch::channel(false);
    run_turn(state, id.clone(), generation, cancel_rx, Vec::new()).await;

    let events = sessions.store().read_from(&id, 0).unwrap();
    assert!(
        events.len() >= 3,
        "expected at least instance_created, user_message, assistant_message; got {}",
        events.len()
    );

    for ev in &events {
        let value = serde_json::to_value(ev).unwrap();
        if let Err(e) = validator.validate(&value) {
            panic!("emitted event {:?} (seq {}) failed schema validation: {e}", ev.event_type, ev.seq);
        }
    }
}
