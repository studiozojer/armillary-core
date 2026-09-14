//! `GET /tray` — the machine-local feed of arrivals.
//!
//! The tray is a feed of *pointers*, not a store. An operator that rings a
//! bell about an artifact (the augur's daily field is the first) also drops
//! one JSON file under `<root>/local/tray/`; the artifact itself stays where
//! its writer keeps it. This route lists those files, newest first. It never
//! discovers artifacts on its own — a directory the tab lists would become a
//! sink, which is exactly how the board once silted — so the write path is the
//! ringer's, and this is read-only.
//!
//! Entries are per machine and never synced. The client fans out across the
//! hosts it knows and chips each card with its origin, then opens the artifact
//! against that host. Design:
//! `zojercommons/projects/harness/specs/2026-09-14-tray-design.md`.

use crate::{blocking, state::SharedState};
use axum::{extract::State, http::StatusCode, Json};
use serde::Serialize;
use std::path::Path;

/// Where the ringers write. Relative to the workspace root; untracked by the
/// router's allowlist, like everything else under `local/`.
const TRAY_DIR: &str = "local/tray";

/// One arrival. The fields a card needs are typed; whatever else the ringer
/// put in the file travels through `extra` untouched, so a new sender can
/// carry its own detail without a schema change here.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct TrayEntry {
    /// The file this entry was read from, relative to the root — the handle a
    /// client would use to remove it, once removal exists.
    pub id: String,
    pub sender: String,
    pub kind: String,
    /// RFC 3339, as written by the ringer. The sort key.
    pub created_at: String,
    pub summary: String,
    /// Workspace-relative path to the artifact, resolvable through `/file` on
    /// the host that wrote it.
    pub path: String,
    /// The machine that wrote the artifact, as the ringer names it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize)]
pub struct Tray {
    pub entries: Vec<TrayEntry>,
    /// Files in the tray directory this route could not read as entries.
    /// Counted rather than hidden: a malformed drop is a bug in a ringer, and
    /// a client that sees `skipped > 0` can say so.
    pub skipped: usize,
}

/// A ringer's payload names the artifact `report_path`; the tray calls it
/// `path`. Both are accepted so the augur's existing payload shape needs no
/// migration, and `path` wins if a file carries both.
fn parse_entry(id: String, text: &str) -> Option<TrayEntry> {
    let serde_json::Value::Object(mut map) = serde_json::from_str(text).ok()? else {
        return None;
    };
    let mut take = |key: &str| -> Option<String> {
        match map.remove(key)? {
            serde_json::Value::String(s) => Some(s),
            _ => None,
        }
    };
    let path = take("path").or_else(|| take("report_path"))?;
    let entry = TrayEntry {
        id,
        sender: take("sender")?,
        kind: take("kind")?,
        created_at: take("created_at")?,
        summary: take("summary")?,
        path,
        host: take("host"),
        date: take("date"),
        extra: map,
    };
    Some(entry)
}

fn build(root: &Path) -> Result<Tray, (StatusCode, String)> {
    let dir = root.join(TRAY_DIR);
    // A machine that has never had an arrival has no directory. That is an
    // empty tray, not a missing resource — the tab must render the same on a
    // fresh machine as on one whose tray was emptied.
    let Ok(read) = std::fs::read_dir(&dir) else {
        return Ok(Tray {
            entries: Vec::new(),
            skipped: 0,
        });
    };

    let mut entries = Vec::new();
    let mut skipped = 0;
    for item in read.flatten() {
        let name = item.file_name().to_string_lossy().to_string();
        // Ringers write `<name>.json.tmp` and rename into place; a tmp file is
        // an entry mid-write, not a malformed one.
        if !name.ends_with(".json") {
            continue;
        }
        let text = match std::fs::read_to_string(item.path()) {
            Ok(t) => t,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        match parse_entry(format!("{TRAY_DIR}/{name}"), &text) {
            Some(entry) => entries.push(entry),
            None => skipped += 1,
        }
    }

    // Newest first. `created_at` is RFC 3339 from one writer per machine, so
    // a string sort is a time sort for entries written in the same offset;
    // ties (and cross-offset drift) break on the filename, which the ringer
    // prefixes with a millisecond stamp.
    entries.sort_by(|a, b| {
        b.created_at
            .cmp(&a.created_at)
            .then_with(|| b.id.cmp(&a.id))
    });

    Ok(Tray { entries, skipped })
}

/// Derived on every request, never cached — the directory is the state, and
/// a ringer may have written since the last read.
pub async fn tray(State(state): State<SharedState>) -> Result<Json<Tray>, (StatusCode, String)> {
    let root = state.root.clone();
    let tray = blocking::run(move || build(&root)).await?;
    Ok(Json(tray))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    const AUGUR: &str = r#"{"date": "2026-09-14", "summary": "Mercury trine natal Neptune perfects (+4)", "events_file": "/abs/field/2026-09-14.events.json", "report_path": "operators/augur/field/2026-09-14.md", "host": "stjerneborg", "event_count": 20, "sender": "augur", "kind": "field", "created_at": "2026-09-14T06:30:00-07:00"}"#;

    #[test]
    fn the_augur_payload_shape_parses_with_report_path_as_path() {
        let e = parse_entry("local/tray/x.json".into(), AUGUR).unwrap();
        assert_eq!(e.sender, "augur");
        assert_eq!(e.kind, "field");
        assert_eq!(e.path, "operators/augur/field/2026-09-14.md");
        assert_eq!(e.host.as_deref(), Some("stjerneborg"));
        assert_eq!(e.date.as_deref(), Some("2026-09-14"));
        // Sender-specific detail travels through untouched.
        assert_eq!(e.extra.get("event_count"), Some(&serde_json::json!(20)));
        assert!(e.extra.get("events_file").is_some());
        // The promoted keys do not also appear in `extra`.
        assert!(e.extra.get("report_path").is_none());
        assert!(e.extra.get("summary").is_none());
    }

    #[test]
    fn a_file_missing_a_required_field_is_not_an_entry() {
        let no_summary =
            r#"{"sender":"x","kind":"y","created_at":"2026-01-01T00:00:00Z","path":"p"}"#;
        assert!(parse_entry("id".into(), no_summary).is_none());
        assert!(parse_entry("id".into(), "not json").is_none());
        assert!(parse_entry("id".into(), "[1,2]").is_none());
    }

    #[test]
    fn no_directory_is_an_empty_tray_not_an_error() {
        let root = tempfile::tempdir().unwrap();
        let tray = build(root.path()).unwrap();
        assert!(tray.entries.is_empty());
        assert_eq!(tray.skipped, 0);
    }

    #[test]
    fn newest_first_and_malformed_counted_and_tmp_ignored() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("local/tray");
        fs::create_dir_all(&dir).unwrap();
        let entry = |created: &str, summary: &str| {
            format!(
                r#"{{"sender":"augur","kind":"field","created_at":"{created}","summary":"{summary}","path":"p"}}"#
            )
        };
        fs::write(
            dir.join("1-old.json"),
            entry("2026-09-13T06:30:00-07:00", "old"),
        )
        .unwrap();
        fs::write(
            dir.join("2-new.json"),
            entry("2026-09-14T06:30:00-07:00", "new"),
        )
        .unwrap();
        fs::write(dir.join("3-bad.json"), "{").unwrap();
        fs::write(
            dir.join("4-mid.json.tmp"),
            entry("2026-09-15T06:30:00-07:00", "mid-write"),
        )
        .unwrap();
        fs::write(dir.join(".DS_Store"), b"noise").unwrap();

        let tray = build(root.path()).unwrap();
        let summaries: Vec<&str> = tray.entries.iter().map(|e| e.summary.as_str()).collect();
        assert_eq!(summaries, ["new", "old"]);
        assert_eq!(tray.skipped, 1);
        assert_eq!(tray.entries[0].id, "local/tray/2-new.json");
    }
}
