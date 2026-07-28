use crate::{blocking, guard, state::SharedState};
use axum::{extract::State, http::StatusCode, Json};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

/// Frontmatter is read with a cap rather than in full: a transcript body is
/// unbounded prose and none of it is wanted here.
const MAX_FRONTMATTER_BYTES: usize = 8 * 1024;

/// Extensions the audio-directory listing pass will report as candidates.
///
/// Deliberately the same set the app uses to decide a path is audio, so the
/// client and this engine cannot disagree about what counts as a memo — a
/// wider engine-side set would report a file the app itself refuses to play.
/// The cost, accepted rather than overlooked: a memo saved in a format
/// outside this list goes invisible from `/voicenotes` entirely, rather than
/// surfacing as `untranscribed`. What this closes: on the real inbox,
/// `.DS_Store` and a `.kairosbackup` file were both reported as
/// `untranscribed` memos before this filter existed — noise on disk is not a
/// voice note just because it shares a directory with some.
const AUDIO_EXTENSIONS: [&str; 5] = ["m4a", "mp3", "wav", "m4b", "aac"];

/// True when a listed file's extension is one this engine treats as audio.
/// Matched case-insensitively, like every other name comparison in this
/// engine (see `guard.rs`). Governs the directory-listing pass only — a
/// transcript's own `source:` field is trusted regardless of extension,
/// because a transcript naming a file this engine would not have listed is
/// still a transcript.
fn is_audio(name: &str) -> bool {
    match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => {
            AUDIO_EXTENSIONS.contains(&ext.to_lowercase().as_str())
        }
        _ => false,
    }
}

#[derive(Serialize, Clone)]
pub struct Transcript {
    path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recorded: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_min: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transcribed_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
}

#[derive(Serialize)]
pub struct AudioEntry {
    audio: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes: Option<u64>,
    /// "transcribed" | "untranscribed" | "audio_absent"
    state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    transcript: Option<Transcript>,
}

#[derive(Serialize)]
pub struct VoicenoteIndex {
    audio_root: String,
    /// Present so a client can say *where* it looked when the answer is empty.
    transcript_roots: Vec<String>,
    entries: Vec<AudioEntry>,
}

/// Read the leading `---` block as flat key/value pairs.
///
/// Deliberately not a YAML parser. The frontmatter this reads is emitted by one
/// tool (`practices/voicenotes/transcribe.py`) in one flat shape, and pulling in
/// a YAML dependency to read six string fields would buy generality nobody
/// asked for. A file whose frontmatter this cannot read is skipped, which is the
/// same posture C-4 takes toward a protocol source that is not present.
fn parse_frontmatter(text: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("---") {
        return out;
    }
    // A block is only frontmatter if it closes. Without this, a file that
    // opens with `---` (a markdown horizontal rule is exactly that spelling)
    // but never closes it scans to EOF and absorbs body prose — so an
    // ordinary hand-written document sharing the transcript directory, with
    // a line shaped `Source: see below`, misreads as a transcript.
    let mut closed = false;
    for line in lines {
        let trimmed = line.trim();
        if trimmed == "---" {
            closed = true;
            break;
        }
        let Some((key, value)) = trimmed.split_once(':') else {
            continue;
        };
        let value = value.trim().trim_matches('"').trim_matches('\'');
        if value.is_empty() {
            continue;
        }
        out.insert(key.trim().to_string(), value.to_string());
    }
    if closed {
        out
    } else {
        BTreeMap::new()
    }
}

/// A trailing (or leading) `/` on a manifest-declared directory is an
/// authoring detail, not content. Left in, it doubles the separator every
/// time the directory is joined with a filename (`local/inbox//done.m4a`),
/// which matches nothing — done once here, at the point the value is read,
/// so every later join and every response field sees the same clean string.
fn trim_slashes(s: &str) -> String {
    s.trim_matches('/').to_string()
}

/// The declared locations, read out of the manifest rather than hardcoded.
///
/// `Protocol.extra` already carries unknown keys (C-5 forbids
/// `deny_unknown_fields`), so declaring `audio` and `transcripts` on the
/// voicenotes protocol needs no schema change and no constitution change. That
/// is the whole reason this shape was chosen: the manifest can already say it.
fn declared(root: &Path) -> Result<(String, Vec<String>), (StatusCode, String)> {
    let composition = armillary_composition::parse_workspace(root)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let protocol = composition
        .protocols
        .iter()
        .find(|p| p.name == "voicenotes")
        .ok_or((StatusCode::NOT_FOUND, "not_composed".to_string()))?;

    let audio = protocol
        .extra
        .get("audio")
        .and_then(|v| v.as_str())
        .map(trim_slashes)
        .ok_or((StatusCode::NOT_FOUND, "not_composed".to_string()))?;

    let transcripts = protocol
        .extra
        .get("transcripts")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|v| v.as_str().map(trim_slashes))
                .collect()
        })
        .unwrap_or_default();

    Ok((audio, transcripts))
}

/// Trimmed and lowercased, for comparison only — never stored and never
/// returned. `guard.rs` already treats case-insensitivity as a property of
/// the filesystem to be assumed rather than detected; a join performed off
/// to the side of the guard does not get to disagree with that.
fn normalize_for_match(s: &str) -> String {
    s.trim_matches('/').to_lowercase()
}

fn build(root: &Path) -> Result<VoicenoteIndex, (StatusCode, String)> {
    let (audio_root, transcript_roots) = declared(root)?;

    // source path -> transcript, keyed by the raw `source:` string exactly as
    // transcribe.py wrote it — that is what an `audio_absent` entry reports
    // back verbatim. `match_index` below is the same set of entries again,
    // keyed by a normalized form, and exists only to find a match; a client
    // never sees it.
    let mut by_source: BTreeMap<String, Transcript> = BTreeMap::new();
    let mut match_index: BTreeMap<String, String> = BTreeMap::new();
    for dir in &transcript_roots {
        let Ok(resolved) = guard::resolve(root, dir) else {
            continue; // C-4: a declared location that is not here is absent, not an error.
        };
        let Ok(read) = std::fs::read_dir(&resolved) else {
            continue;
        };
        for item in read.flatten() {
            let path = item.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let head = String::from_utf8_lossy(&bytes[..bytes.len().min(MAX_FRONTMATTER_BYTES)]);
            let fm = parse_frontmatter(&head);
            let Some(source) = fm.get("source") else {
                continue; // Not a transcript, or one that cannot be linked.
            };
            let name = item.file_name().to_string_lossy().to_string();
            match_index.insert(normalize_for_match(source), source.clone());
            by_source.insert(
                source.clone(),
                Transcript {
                    path: format!("{dir}/{name}"),
                    title: fm.get("title").cloned(),
                    recorded: fm.get("recorded").cloned(),
                    duration_min: fm.get("duration_min").cloned(),
                    transcribed_by: fm.get("transcribed_by").cloned(),
                    model: fm.get("model").cloned(),
                },
            );
        }
    }

    let mut entries: Vec<AudioEntry> = Vec::new();
    let mut seen: Vec<String> = Vec::new();

    if let Ok(resolved) = guard::resolve(root, &audio_root) {
        if let Ok(read) = std::fs::read_dir(&resolved) {
            for item in read.flatten() {
                let name = item.file_name().to_string_lossy().to_string();
                if guard::is_hidden_from_listings(&name) {
                    continue;
                }
                if !is_audio(&name) {
                    continue;
                }
                let Ok(meta) = item.path().metadata() else {
                    continue;
                };
                if meta.is_dir() {
                    continue;
                }
                let rel = format!("{audio_root}/{name}");
                let matched_source = match_index.get(&normalize_for_match(&rel)).cloned();
                let transcript = matched_source.as_ref().and_then(|s| by_source.get(s)).cloned();
                if let Some(source) = &matched_source {
                    seen.push(source.clone());
                }
                entries.push(AudioEntry {
                    audio: rel,
                    bytes: Some(meta.len()),
                    state: if transcript.is_some() {
                        "transcribed".to_string()
                    } else {
                        "untranscribed".to_string()
                    },
                    transcript,
                });
            }
        }
    }

    // A transcript naming audio that never reached this machine. Not an error
    // and not noise: local/inbox is untracked and machine-local while the
    // transcript is committed, so on a second machine this is EVERY memo. The
    // state is the difference between "not yet transcribed" and "transcribed
    // elsewhere", which are invisible to each other without it.
    for (source, transcript) in by_source {
        if seen.contains(&source) {
            continue;
        }
        entries.push(AudioEntry {
            audio: source,
            bytes: None,
            state: "audio_absent".to_string(),
            transcript: Some(transcript),
        });
    }

    entries.sort_by_key(|a| a.audio.to_lowercase());

    Ok(VoicenoteIndex {
        audio_root,
        transcript_roots,
        entries,
    })
}

/// Derived on every request, never stored. An index file would be a second
/// source of truth, and it would drift the first time a transcript was written
/// by anything that did not know to update it — which is every hand-run of
/// transcribe.py. The `source:` field in the transcript's own frontmatter is
/// already the edge; this route only inverts it.
pub async fn voicenotes(
    State(state): State<SharedState>,
) -> Result<Json<VoicenoteIndex>, (StatusCode, String)> {
    let root = state.root.clone();
    let index = blocking::run(move || build(&root)).await?;
    Ok(Json(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Mirrors the real topology: audio that is untracked and machine-local,
    /// transcripts that are committed and travel — so the two can be present
    /// independently of each other.
    fn farm() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        fs::write(
            root.path().join("modules.toml"),
            r#"
[[protocols]]
name = "voicenotes"
source = "commons/practices/voicenotes/practice.md"
load = "on-demand"
audio = "local/inbox"
transcripts = ["commons/voicenotes"]
"#,
        )
        .unwrap();

        fs::create_dir_all(root.path().join("local/inbox")).unwrap();
        fs::write(root.path().join("local/inbox/done.m4a"), b"audio").unwrap();
        fs::write(root.path().join("local/inbox/pending.m4a"), b"audio").unwrap();

        fs::create_dir_all(root.path().join("commons/voicenotes")).unwrap();
        fs::write(
            root.path().join("commons/voicenotes/2026-07-22-done.md"),
            "---\ntitle: \"Done — raw transcript\"\nrecorded: 2026-07-22\nsource: \"local/inbox/done.m4a\"\nduration_min: 2.4\ntranscribed_by: \"@tycho\"\nmodel: \"faster-whisper large-v3\"\n---\n\nbody\n",
        )
        .unwrap();
        // A transcript whose audio never reached this machine — the third state.
        fs::write(
            root.path().join("commons/voicenotes/2026-07-23-elsewhere.md"),
            "---\ntitle: \"Elsewhere\"\nsource: \"local/inbox/elsewhere.m4a\"\n---\n\nbody\n",
        )
        .unwrap();
        // Malformed: no frontmatter at all. Must be skipped, not fatal.
        fs::write(root.path().join("commons/voicenotes/notes.md"), "just prose\n").unwrap();

        root
    }

    #[test]
    fn parses_the_frontmatter_block_only() {
        let fm = parse_frontmatter("---\ntitle: \"A\"\nsource: \"x.m4a\"\n---\nbody: not frontmatter\n");
        assert_eq!(fm.get("title").map(String::as_str), Some("A"));
        assert_eq!(fm.get("source").map(String::as_str), Some("x.m4a"));
        assert!(!fm.contains_key("body"));
    }

    #[test]
    fn a_file_without_frontmatter_yields_nothing_rather_than_failing() {
        assert!(parse_frontmatter("just prose\n").is_empty());
    }

    #[test]
    fn unterminated_frontmatter_yields_nothing_rather_than_absorbing_the_body() {
        // A markdown horizontal rule opens exactly like a frontmatter block.
        // Without a closing `---`, everything after it — including a body
        // line shaped like a field — must not be read as frontmatter.
        let fm = parse_frontmatter("---\ntitle: \"A\"\nSource: see below\nmore prose, never closed\n");
        assert!(fm.is_empty());
    }

    #[test]
    fn a_body_line_shaped_like_a_field_after_a_closed_block_is_not_absorbed() {
        let fm = parse_frontmatter("---\ntitle: \"A\"\n---\nsource: not real frontmatter\n");
        assert_eq!(fm.len(), 1);
        assert_eq!(fm.get("title").map(String::as_str), Some("A"));
    }

    #[test]
    fn derives_all_three_states() {
        let root = farm();
        let index = build(root.path()).unwrap();

        let state_of = |audio: &str| {
            index
                .entries
                .iter()
                .find(|e| e.audio.ends_with(audio))
                .unwrap_or_else(|| panic!("{audio} missing from index"))
                .state
                .clone()
        };

        assert_eq!(state_of("done.m4a"), "transcribed");
        assert_eq!(state_of("pending.m4a"), "untranscribed");
        assert_eq!(state_of("elsewhere.m4a"), "audio_absent");
    }

    #[test]
    fn one_malformed_transcript_does_not_empty_the_index() {
        let root = farm();
        assert_eq!(build(root.path()).unwrap().entries.len(), 3);
    }

    #[test]
    fn transcript_metadata_rides_along() {
        let root = farm();
        let index = build(root.path()).unwrap();
        let done = index.entries.iter().find(|e| e.audio.ends_with("done.m4a")).unwrap();
        let t = done.transcript.as_ref().unwrap();
        assert_eq!(t.path, "commons/voicenotes/2026-07-22-done.md");
        assert_eq!(t.transcribed_by.as_deref(), Some("@tycho"));
        assert_eq!(t.recorded.as_deref(), Some("2026-07-22"));
    }

    #[test]
    fn non_audio_noise_in_the_audio_directory_is_absent_not_untranscribed() {
        let root = tempfile::tempdir().unwrap();
        fs::write(
            root.path().join("modules.toml"),
            r#"
[[protocols]]
name = "voicenotes"
source = "commons/practices/voicenotes/practice.md"
load = "on-demand"
audio = "local/inbox"
transcripts = ["commons/voicenotes"]
"#,
        )
        .unwrap();
        fs::create_dir_all(root.path().join("local/inbox")).unwrap();
        fs::write(root.path().join("local/inbox/pending.m4a"), b"audio").unwrap();
        // The real case that surfaced this: .DS_Store is not a voice memo.
        fs::write(root.path().join("local/inbox/.DS_Store"), b"noise").unwrap();
        fs::create_dir_all(root.path().join("commons/voicenotes")).unwrap();

        let index = build(root.path()).unwrap();
        assert!(
            index.entries.iter().all(|e| !e.audio.ends_with(".DS_Store")),
            ".DS_Store must not appear in entries at all: {:?}",
            index.entries.iter().map(|e| &e.audio).collect::<Vec<_>>()
        );
        assert!(index.entries.iter().any(|e| e.audio.ends_with("pending.m4a")));
    }

    #[test]
    fn an_uppercase_audio_extension_is_still_included() {
        let root = tempfile::tempdir().unwrap();
        fs::write(
            root.path().join("modules.toml"),
            r#"
[[protocols]]
name = "voicenotes"
source = "commons/practices/voicenotes/practice.md"
load = "on-demand"
audio = "local/inbox"
transcripts = ["commons/voicenotes"]
"#,
        )
        .unwrap();
        fs::create_dir_all(root.path().join("local/inbox")).unwrap();
        fs::write(root.path().join("local/inbox/MEMO.M4A"), b"audio").unwrap();
        fs::create_dir_all(root.path().join("commons/voicenotes")).unwrap();

        let index = build(root.path()).unwrap();
        assert!(index.entries.iter().any(|e| e.audio.ends_with("MEMO.M4A")));
    }

    #[test]
    fn a_trailing_slash_on_a_declared_directory_still_matches() {
        let root = tempfile::tempdir().unwrap();
        fs::write(
            root.path().join("modules.toml"),
            r#"
[[protocols]]
name = "voicenotes"
source = "commons/practices/voicenotes/practice.md"
load = "on-demand"
audio = "local/inbox/"
transcripts = ["commons/voicenotes/"]
"#,
        )
        .unwrap();
        fs::create_dir_all(root.path().join("local/inbox")).unwrap();
        fs::write(root.path().join("local/inbox/done.m4a"), b"audio").unwrap();
        fs::create_dir_all(root.path().join("commons/voicenotes")).unwrap();
        fs::write(
            root.path().join("commons/voicenotes/done.md"),
            "---\nsource: \"local/inbox/done.m4a\"\n---\nbody\n",
        )
        .unwrap();

        let index = build(root.path()).unwrap();
        let entry = index.entries.iter().find(|e| e.audio.ends_with("done.m4a")).unwrap();
        assert_eq!(entry.state, "transcribed");
        assert!(!entry.audio.contains("//"), "audio path doubled a separator: {}", entry.audio);
    }

    #[test]
    fn a_source_differing_only_in_case_from_the_listed_filename_still_matches() {
        let root = tempfile::tempdir().unwrap();
        fs::write(
            root.path().join("modules.toml"),
            r#"
[[protocols]]
name = "voicenotes"
source = "commons/practices/voicenotes/practice.md"
load = "on-demand"
audio = "local/inbox"
transcripts = ["commons/voicenotes"]
"#,
        )
        .unwrap();
        fs::create_dir_all(root.path().join("local/inbox")).unwrap();
        fs::write(root.path().join("local/inbox/done.m4a"), b"audio").unwrap();
        fs::create_dir_all(root.path().join("commons/voicenotes")).unwrap();
        fs::write(
            root.path().join("commons/voicenotes/done.md"),
            "---\nsource: \"Local/Inbox/Done.m4a\"\n---\nbody\n",
        )
        .unwrap();

        let index = build(root.path()).unwrap();
        let entry = index.entries.iter().find(|e| e.audio.ends_with("done.m4a")).unwrap();
        assert_eq!(entry.state, "transcribed");
    }

    #[test]
    fn a_workspace_that_does_not_declare_voicenotes_has_no_index() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("modules.toml"), "").unwrap();
        // Presence-gated, like every other protocol: undeclared means the
        // feature is absent, which is a working state and not an error.
        assert!(build(root.path()).is_err());
    }
}
