//! Append-only, per-stream JSONL store — the on-disk half of I-1.
//!
//! One file per stream at `<data_dir>/streams/<stream>.jsonl`. `append`
//! enforces the monotonic-`seq` invariant (I-2 invariant iii) itself, at the
//! write boundary, rather than trusting callers: a wrong `seq` is rejected
//! before a single byte reaches disk, so the file on disk and the caller's
//! idea of the head never diverge (A-4's one-writer-per-stream discipline
//! only holds if the store itself refuses an out-of-order write). Transient
//! events (`seq == 0`) are rejected here too — I-4 says a hint is never
//! persisted, so the store is the last place that could accidentally do it
//! anyway, and it does not.

use crate::log::envelope::EventEnvelope;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Append-only JSONL log, one file per stream, all files under a single
/// `streams/` directory of a data dir.
pub struct LogStore {
    dir: PathBuf,
}

/// Stream names come from instance ids: no path separators, no leading dot.
/// Rejected here — not merely discouraged — so the store can never be
/// steered outside `dir` by a crafted stream name (the guard module's path
/// discipline, applied to this surface too).
fn validate_stream_name(stream: &str) -> io::Result<()> {
    if stream.is_empty() || stream.contains('/') || stream.contains('\\') || stream.starts_with('.')
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("bad_stream_name: {stream:?}"),
        ));
    }
    Ok(())
}

impl LogStore {
    /// Creates `<data_dir>/streams/` (recursively) and returns a store
    /// rooted there.
    pub fn open(data_dir: &Path) -> io::Result<Self> {
        let dir = data_dir.join("streams");
        fs::create_dir_all(&dir)?;
        Ok(LogStore { dir })
    }

    fn path_for(&self, stream: &str) -> PathBuf {
        self.dir.join(format!("{stream}.jsonl"))
    }

    /// Appends `ev` to `stream`'s file, enforcing I-2's monotonic-`seq`
    /// invariant and I-4's durable-only rule before a single byte reaches
    /// disk. Write + flush; any I/O error propagates untouched so it
    /// surfaces to the writer (I-5) rather than being swallowed here.
    pub fn append(&self, stream: &str, ev: &EventEnvelope) -> io::Result<()> {
        validate_stream_name(stream)?;

        if ev.seq == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "transient_not_durable: seq 0 events are never persisted (I-4)",
            ));
        }

        let head = self.head_seq(stream)?;
        if ev.seq != head + 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "wrong_seq: stream {stream:?} head is {head}, expected seq {}, got {}",
                    head + 1,
                    ev.seq
                ),
            ));
        }

        let mut line = serde_json::to_string(ev)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
        line.push('\n');

        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(self.path_for(stream))?;
        file.write_all(line.as_bytes())?;
        file.flush()?;
        Ok(())
    }

    /// Reads every event in `stream` with `seq >= from_seq`, in file order
    /// (which is append order, which is seq order — I-1). Absent stream
    /// reads as empty, not an error: a stream that has never been written
    /// to is not a corrupt one.
    pub fn read_from(&self, stream: &str, from_seq: u64) -> io::Result<Vec<EventEnvelope>> {
        validate_stream_name(stream)?;

        let path = self.path_for(stream);
        let file = match fs::File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };

        let mut events = Vec::new();
        for line in BufReader::new(file).lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let ev: EventEnvelope = serde_json::from_str(&line).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("corrupt log entry in stream {stream:?}: {e}"),
                )
            })?;
            if ev.seq >= from_seq {
                events.push(ev);
            }
        }
        Ok(events)
    }

    /// The last event's `seq`, or `0` when the stream is absent or empty.
    pub fn head_seq(&self, stream: &str) -> io::Result<u64> {
        Ok(self
            .read_from(stream, 0)?
            .last()
            .map(|e| e.seq)
            .unwrap_or(0))
    }

    /// The first event's `seq`, or `0` when the stream is absent or empty.
    pub fn earliest_seq(&self, stream: &str) -> io::Result<u64> {
        Ok(self
            .read_from(stream, 0)?
            .first()
            .map(|e| e.seq)
            .unwrap_or(0))
    }

    /// Stream names (file stems) with a durable log on disk.
    pub fn streams(&self) -> io::Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    names.push(stem.to_string());
                }
            }
        }
        Ok(names)
    }
}

#[cfg(test)]
mod tests {
    use crate::log::envelope::{Actor, Cost, EventEnvelope, Role};
    use crate::log::store::LogStore;
    use std::fs;
    use std::io;

    fn envelope(stream: &str, seq: u64) -> EventEnvelope {
        EventEnvelope {
            stream: stream.to_string(),
            id: format!("id-{seq}"),
            seq,
            ts: "2026-07-27T00:00:00Z".to_string(),
            actor: Actor {
                role: Role::User,
                instance: None,
            },
            event_type: "user_message".to_string(),
            thread: None,
            parent: None,
            version: 1,
            cost: None,
            data: serde_json::json!({"text": "hi"}),
        }
    }

    #[test]
    fn append_two_then_read_from_zero_returns_both_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        let e1 = envelope("s1", 1);
        let e2 = envelope("s1", 2);
        store.append("s1", &e1).unwrap();
        store.append("s1", &e2).unwrap();

        let read = store.read_from("s1", 0).unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].seq, 1);
        assert_eq!(read[1].seq, 2);
    }

    #[test]
    fn read_from_two_returns_only_the_second() {
        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        store.append("s1", &envelope("s1", 1)).unwrap();
        store.append("s1", &envelope("s1", 2)).unwrap();

        let read = store.read_from("s1", 2).unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].seq, 2);
    }

    #[test]
    fn head_and_earliest_seq_are_correct() {
        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        store.append("s1", &envelope("s1", 1)).unwrap();
        store.append("s1", &envelope("s1", 2)).unwrap();
        store.append("s1", &envelope("s1", 3)).unwrap();

        assert_eq!(store.head_seq("s1").unwrap(), 3);
        assert_eq!(store.earliest_seq("s1").unwrap(), 1);
    }

    #[test]
    fn head_and_earliest_seq_are_zero_for_unknown_stream() {
        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();

        assert_eq!(store.head_seq("nope").unwrap(), 0);
        assert_eq!(store.earliest_seq("nope").unwrap(), 0);
    }

    #[test]
    fn append_with_wrong_seq_errors_invalid_input() {
        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        store.append("s1", &envelope("s1", 1)).unwrap();

        // head is 1; next must be 2, not 3.
        let err = store.append("s1", &envelope("s1", 3)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("wrong_seq"));

        // The rejected write must not have reached disk: head is still 1.
        assert_eq!(store.head_seq("s1").unwrap(), 1);
    }

    #[test]
    fn append_rejects_transient_events() {
        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();

        let err = store.append("s1", &envelope("s1", 0)).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("transient_not_durable"));
        assert_eq!(store.head_seq("s1").unwrap(), 0);
    }

    #[test]
    fn append_rejects_stream_names_with_path_separators() {
        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();

        let err = store
            .append("../escape", &envelope("../escape", 1))
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("bad_stream_name"));
    }

    #[test]
    fn append_rejects_stream_names_starting_with_a_dot() {
        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();

        let err = store
            .append(".hidden", &envelope(".hidden", 1))
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("bad_stream_name"));
    }

    #[test]
    fn streams_lists_stem_names() {
        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        store.append("s1", &envelope("s1", 1)).unwrap();
        store.append("s2", &envelope("s2", 1)).unwrap();

        let mut streams = store.streams().unwrap();
        streams.sort();
        assert_eq!(streams, vec!["s1".to_string(), "s2".to_string()]);
    }

    #[test]
    fn read_from_a_corrupt_log_errors_rather_than_truncating() {
        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        store.append("s1", &envelope("s1", 1)).unwrap();

        let path = dir.path().join("streams").join("s1.jsonl");
        let mut contents = fs::read_to_string(&path).unwrap();
        contents.push_str("not json\n");
        fs::write(&path, contents).unwrap();

        let err = store.read_from("s1", 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("s1"));
    }

    #[cfg(unix)]
    #[test]
    fn append_to_an_unwritable_dir_returns_err_and_does_not_advance_head() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        store.append("s1", &envelope("s1", 1)).unwrap();

        let streams_dir = dir.path().join("streams");
        let mut perms = fs::metadata(&streams_dir).unwrap().permissions();
        perms.set_mode(0o444);
        fs::set_permissions(&streams_dir, perms.clone()).unwrap();

        let result = store.append("s1", &envelope("s1", 2));

        // Restore permissions so the tempdir can be cleaned up.
        perms.set_mode(0o755);
        fs::set_permissions(&streams_dir, perms).unwrap();

        assert!(result.is_err());
        // In-memory state (the on-disk head, which is what callers consult)
        // must not have advanced past what actually landed on disk (I-5).
        assert_eq!(store.head_seq("s1").unwrap(), 1);
    }

    #[test]
    fn envelope_round_trip_preserves_type_field_name_and_omits_absent_options() {
        let ev = EventEnvelope {
            stream: "s1".to_string(),
            id: "id-1".to_string(),
            seq: 1,
            ts: "2026-07-27T00:00:00Z".to_string(),
            actor: Actor {
                role: Role::Operator,
                instance: Some("tycho".to_string()),
            },
            event_type: "boot".to_string(),
            thread: None,
            parent: None,
            version: 1,
            cost: None,
            data: serde_json::json!({}),
        };

        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"type\":\"boot\""));
        assert!(!json.contains("event_type"));
        assert!(!json.contains("\"thread\""));
        assert!(!json.contains("\"parent\""));
        assert!(!json.contains("\"cost\""));

        let round_tripped: EventEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped.event_type, "boot");
        assert_eq!(round_tripped.actor.instance, Some("tycho".to_string()));
        assert!(round_tripped.thread.is_none());

        // Sanity-check Cost is part of the shape even though this test omits it.
        let _ = Cost {
            bytes: Some(1),
            tokens: Some(2),
        };
    }
}
