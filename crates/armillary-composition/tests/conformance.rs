//! Runs `conformance/manifest/` as data.
//!
//! Table-driven on purpose: adding a fixture must require no code change here.
//! That property is what stops the suite from drifting into "whatever the
//! parser happens to do" — the fixtures are the specification, and this file is
//! only a runner.

use armillary_composition::{Composition, CompositionError};
use std::{
    fs,
    path::{Path, PathBuf},
};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/manifest")
        .canonicalize()
        .expect("conformance/manifest must exist relative to the crate")
}

/// Parse `<stem>.toml`, merging `<stem>.local.toml` when present (C-6).
fn run_fixture(dir: &Path, stem: &str) -> Result<Composition, CompositionError> {
    let base_text = fs::read_to_string(dir.join(format!("{stem}.toml"))).expect("read base");
    let base = armillary_composition::parse_manifest_str(&base_text)?;

    let overlay_path = dir.join(format!("{stem}.local.toml"));
    if overlay_path.exists() {
        let overlay_text = fs::read_to_string(&overlay_path).expect("read overlay");
        let overlay = armillary_composition::parse_manifest_str(&overlay_text)?;
        armillary_composition::merge(base, overlay)
    } else {
        Ok(base)
    }
}

#[test]
fn manifest_fixtures_match_expected() {
    let dir = fixtures_dir();
    let (mut ok, mut errs) = (0, 0);

    for entry in fs::read_dir(&dir).expect("read conformance/manifest") {
        let path = entry.expect("dir entry").path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if !name.ends_with(".toml") || name.ends_with(".local.toml") {
            continue;
        }
        let stem = name.trim_end_matches(".toml").to_string();

        let expected_ok = dir.join(format!("{stem}.expected.json"));
        let expected_err = dir.join(format!("{stem}.expected-error.json"));

        if expected_ok.exists() {
            let expected: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(&expected_ok).unwrap()).unwrap();
            let composition = run_fixture(&dir, &stem)
                .unwrap_or_else(|e| panic!("fixture {stem} should parse but failed: {e}"));
            let actual = serde_json::to_value(&composition).unwrap();
            assert_eq!(actual, expected, "fixture {stem} did not match expected output");
            ok += 1;
        } else if expected_err.exists() {
            let expected: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(&expected_err).unwrap()).unwrap();
            let err = run_fixture(&dir, &stem)
                .expect_err(&format!("fixture {stem} should have failed but succeeded"));
            let actual = serde_json::to_value(err.as_conformance_error()).unwrap();
            assert_eq!(actual, expected, "fixture {stem} produced the wrong error");
            errs += 1;
        } else {
            panic!("fixture {stem} has neither .expected.json nor .expected-error.json");
        }
    }

    // Not ceremony. A fixture glob that silently matches nothing reports
    // success, which is the same failure as a nullglob making a permission
    // denial indistinguishable from an empty directory. Make blindness loud.
    assert!(ok > 0, "no success fixtures were discovered — the glob matched nothing");
    assert!(errs > 0, "no error fixtures were discovered — the glob matched nothing");

    // Printed rather than asserted against a fixed number, so that adding a
    // fixture still requires no code change here — but a run that quietly
    // covered less than expected is visible under `--nocapture`.
    eprintln!("conformance/manifest: {ok} success fixtures, {errs} error fixtures");
}
