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
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Append-only JSONL log, one file per stream, all files under a single
/// `streams/` directory of a data dir.
pub struct LogStore {
    dir: PathBuf,
}

/// Before writing a new line, clean up a torn tail (see `LogStore::
/// read_from`'s doc) left on disk by a prior power-loss write, so the new
/// write lands on a properly newline-terminated file rather than
/// concatenating onto garbage.
///
/// `OpenOptions::append` positions at the literal end of the file
/// regardless of newlines — it has no concept of "lines" at all — so
/// without this step, appending straight onto a torn fragment would fuse
/// the fragment and the new event into a single line. That line WOULD end
/// in `\n` (this store always terminates what it writes), so it would no
/// longer be exempt as a torn tail on the next read: `read_from` would see
/// one fully-terminated, unparseable line and error forever, exactly the
/// brick this whole fix exists to prevent.
///
/// So: if the file's last byte isn't `\n`, find the last complete line (the
/// bytes after the last `\n`, or the whole file if it has none) and check
/// whether IT parses. If it doesn't, it's the torn fragment itself —
/// truncate it off entirely, discarding exactly the bytes `read_from`
/// already treats as absent, so nothing durable is lost (I-5: the write
/// that produced it already surfaced its own failure to its writer; this
/// is cleanup, not a second chance for unacknowledged data). If it DOES
/// parse, it's a complete event that simply never got its trailing
/// newline — not this fix's concern — so this only appends the missing
/// newline rather than discarding real content.
fn repair_torn_tail_before_append(path: &Path, stream: &str) -> io::Result<()> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    if bytes.is_empty() || bytes.last() == Some(&b'\n') {
        return Ok(());
    }

    let last_newline = bytes.iter().rposition(|&b| b == b'\n');
    let tail_start = last_newline.map(|i| i + 1).unwrap_or(0);
    let tail = &bytes[tail_start..];
    let tail_parses = std::str::from_utf8(tail)
        .ok()
        .and_then(|s| serde_json::from_str::<EventEnvelope>(s).ok())
        .is_some();

    if tail_parses {
        // A complete event, just missing its own trailing newline.
        let mut file = OpenOptions::new().append(true).open(path)?;
        file.write_all(b"\n")?;
        file.flush()?;
    } else {
        eprintln!(
            "warning: stream {stream:?} has a torn tail on disk ({} garbage byte(s) after the \
             last complete line) — truncating it before this append; the write that produced \
             it already surfaced its own error to its writer at write time (I-5), so nothing \
             here silently diverges.",
            tail.len()
        );
        let file = OpenOptions::new().write(true).open(path)?;
        file.set_len(tail_start as u64)?;
    }
    Ok(())
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

        let path = self.path_for(stream);
        // `head` above was computed via `read_from`, which already treats a
        // torn tail (see its doc) as absent — but the garbage bytes
        // themselves are still sitting on disk. Clean them up before this
        // write lands, so a plain `OpenOptions::append` below can't
        // concatenate onto them (see `repair_torn_tail_before_append`'s doc
        // for why that matters).
        repair_torn_tail_before_append(&path, stream)?;

        let mut file = OpenOptions::new().append(true).create(true).open(&path)?;
        file.write_all(line.as_bytes())?;
        file.flush()?;
        Ok(())
    }

    /// Reads every event in `stream` with `seq >= from_seq`, in file order
    /// (which is append order, which is seq order — I-1). Absent stream
    /// reads as empty, not an error: a stream that has never been written
    /// to is not a corrupt one.
    ///
    /// ## The torn-tail seam
    ///
    /// `append` is write + flush, not write + flush + fsync, so a power loss
    /// mid-write can leave the LAST line on disk truncated mid-byte — no
    /// trailing newline, garbage JSON. Before this function accounted for
    /// that, any unparseable line failed the entire read, forever: `read_from`
    /// errored, `head_seq` (which calls it) errored, and every future
    /// `append` (which calls `head_seq` first) errored too — a torn tail
    /// permanently bricked the stream.
    ///
    /// So: a line that fails to parse is treated as a genuine corruption
    /// (`InvalidData`, as before) UNLESS it is both (a) the last line in the
    /// file and (b) the file does not end in a trailing newline — the exact
    /// shape a torn write leaves. In that one case it is a torn tail, not
    /// corruption: the write that produced it already surfaced an error to
    /// ITS writer at write time (I-5 — in-memory and on-disk state cannot
    /// have silently diverged, because nobody ever observed that write as
    /// having succeeded), so dropping the fragment here doesn't lose an
    /// acknowledged fact, it just declines to resurrect an unacknowledged
    /// one. The line is logged and skipped — `read_from`, and everything
    /// built on it (`head_seq`, `earliest_seq`, `append`'s own next-`seq`
    /// check), behaves as if the torn line were never there.
    ///
    /// A corrupt line anywhere else in the file — mid-file, or a final line
    /// that DOES end with a newline (so it was fully flushed, and whatever
    /// is wrong with it isn't a torn write) — still errors exactly as before.
    pub fn read_from(&self, stream: &str, from_seq: u64) -> io::Result<Vec<EventEnvelope>> {
        validate_stream_name(stream)?;

        let path = self.path_for(stream);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let ends_with_newline = bytes.last() == Some(&b'\n');
        let content = String::from_utf8(bytes).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("non-utf8 bytes in stream {stream:?}: {e}"),
            )
        })?;

        // `str::lines()` splits on '\n' and does not report whether the
        // final line had a trailing terminator — hence `ends_with_newline`,
        // read from the raw bytes above, as the separate signal for that.
        let lines: Vec<&str> = content.lines().collect();
        let last_index = lines.len().checked_sub(1);

        let mut events = Vec::new();
        for (i, line) in lines.into_iter().enumerate() {
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<EventEnvelope>(line) {
                Ok(ev) => {
                    if ev.seq >= from_seq {
                        events.push(ev);
                    }
                }
                Err(e) => {
                    let is_torn_tail = Some(i) == last_index && !ends_with_newline;
                    if is_torn_tail {
                        eprintln!(
                            "warning: stream {stream:?} has a torn tail (unparseable final \
                             line, no trailing newline — a power-loss write, not corruption): \
                             {e}. Ignoring it; the write that produced it already surfaced its \
                             own error to its writer at write time (I-5), so nothing here \
                             silently diverges."
                        );
                        continue;
                    }
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("corrupt log entry in stream {stream:?}: {e}"),
                    ));
                }
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
    use std::fs::{self, OpenOptions};
    use std::io::{self, Write};

    fn envelope(stream: &str, seq: u64) -> EventEnvelope {
        EventEnvelope {
            stream: stream.to_string(),
            id: format!("id-{seq}"),
            seq,
            ts: "2026-07-27T00:00:00Z".to_string(),
            actor: Actor {
                role: Role::User,
                instance: None,
                principal: None,
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

    #[test]
    fn a_final_complete_but_corrupt_line_still_errors() {
        // Same fixture as the test above, but the point being pinned here is
        // narrower and specific to the torn-tail fix: a fully-flushed final
        // line (it HAS a trailing newline, so it was not torn by a
        // power-loss write) that is nonetheless bad JSON must still be
        // treated as real corruption, not silently forgiven the way an
        // actual torn tail now is.
        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        store.append("s1", &envelope("s1", 1)).unwrap();

        let path = dir.path().join("streams").join("s1.jsonl");
        let mut contents = fs::read_to_string(&path).unwrap();
        contents.push_str("not json but flushed completely\n");
        fs::write(&path, contents).unwrap();

        let err = store.read_from("s1", 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn a_corrupt_line_in_the_middle_of_the_file_still_errors() {
        // Only the LAST line, with no trailing newline, is forgiven as a
        // torn tail — a bad line anywhere else in the file is exactly as
        // fatal as before the fix, torn-tail or not.
        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        store.append("s1", &envelope("s1", 1)).unwrap();
        store.append("s1", &envelope("s1", 2)).unwrap();

        let path = dir.path().join("streams").join("s1.jsonl");
        let mut contents = fs::read_to_string(&path).unwrap();
        // Splice garbage between the two good lines, rather than after them.
        let mid = contents.find("\n{").map(|i| i + 1).unwrap_or(contents.len());
        contents.insert_str(mid, "not json\n");
        fs::write(&path, contents).unwrap();

        let err = store.read_from("s1", 0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn a_torn_tail_with_no_trailing_newline_is_ignored_and_append_still_works() {
        // The finding this pins: a power-loss write leaves the final line
        // truncated mid-byte, no trailing newline. Before the fix this
        // bricked the stream forever — read_from errored, so head_seq
        // (which calls it) errored, so every future append (which calls
        // head_seq first) errored too.
        let dir = tempfile::tempdir().unwrap();
        let store = LogStore::open(dir.path()).unwrap();
        store.append("s1", &envelope("s1", 1)).unwrap();

        let path = dir.path().join("streams").join("s1.jsonl");
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        // No trailing newline: exactly the shape write+flush-without-fsync
        // leaves behind on power loss mid-write.
        file.write_all(b"{\"stream\":\"s1\",\"seq\":2,\"id\":\"tor").unwrap();
        drop(file);

        let read = store.read_from("s1", 0).unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].seq, 1);

        assert_eq!(store.head_seq("s1").unwrap(), 1);

        // Appending the next event (seq 2, since the torn fragment is
        // treated as absent) must succeed rather than perpetually 409ing on
        // a head the store can never reach past.
        store.append("s1", &envelope("s1", 2)).unwrap();
        let read = store.read_from("s1", 0).unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[1].seq, 2);
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
                principal: None,
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
