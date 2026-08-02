//! The tool surface — what a session may do, and the switch that does it.
//!
//! Three things live here and nowhere else: the JSON definitions sent to the
//! provider, the name→function dispatch, and the mapping from a failure to the
//! machine code a `tool_result` event records.
//!
//! **This module is the join that had no owner.** The design that preceded it
//! specified three tool *bodies* and a path *gate* and never the switch between
//! them — the same shape as a boot event nobody wrote and a front door that
//! belonged to no task. It is written first, before the loop that calls it, so
//! that gap cannot reopen.
//!
//! **Verbs are engine-owned.** Which tools exist is not a composition question:
//! C-1 governs modules and protocols, and reading a file is a primitive, not a
//! composed module. What a workspace declares shapes what those verbs *reach*,
//! not which verbs exist. That is a recorded debt, not an oversight — a second
//! engine could ship a different surface and both would pass conformance today.

use axum::http::StatusCode;
use serde::Serialize;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

/// A tool call that did not succeed.
///
/// `status` is the machine code, and it is the sovereign half of S-1: the
/// engine reads it, the log records it typed, and loop control keys on it. The
/// model never sees this struct — it sees `is_error` plus whatever the
/// projection renders, which is all the provider channel has room for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolError {
    /// A stable code, never prose. Guard codes pass through verbatim so the
    /// transcript and the log say the same word.
    pub status: &'static str,
    /// Detail for the log and for rendering. May be empty.
    pub detail: String,
}

impl ToolError {
    pub(crate) fn new(status: &'static str, detail: impl Into<String>) -> Self {
        ToolError {
            status,
            detail: detail.into(),
        }
    }

    /// The HTTP status the same failure carries when it reaches a route.
    ///
    /// The routes and the tools now share their bodies, so they must agree
    /// about what each failure *is*. This is the one place that mapping lives,
    /// and it reproduces exactly what `/tree` and `/file` returned before the
    /// share — the Explorer is a shipped consumer and none of these codes is
    /// ours to change on the way past.
    pub fn http_status(&self) -> StatusCode {
        match self.status {
            "malformed_path" => StatusCode::BAD_REQUEST,
            "outside_workspace" | "denied_credential" | "denied_noise" => StatusCode::FORBIDDEN,
            "not_found" => StatusCode::NOT_FOUND,
            "is_a_directory" | "not_a_directory" | "invalid_input" | "invalid_pattern" => {
                StatusCode::BAD_REQUEST
            }
            "not_openable" | "not_text" => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "too_large" => StatusCode::PAYLOAD_TOO_LARGE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl From<crate::guard::GuardError> for ToolError {
    /// The guard's machine code passes through verbatim. D6′ turns on this:
    /// the transcript, the log, and the HTTP response all say the same word,
    /// so a denial read on a phone can be grepped in the log.
    fn from(e: crate::guard::GuardError) -> Self {
        ToolError::new(e.code(), String::new())
    }
}

/// The tool definitions sent with every request.
///
/// Order is fixed and deliberate. Tool definitions render at the front of the
/// prompt, ahead of the system prompt and messages, so a set that reorders
/// between requests invalidates the whole cached prefix. It costs nothing to
/// fix that now with one tool and is awkward to retrofit later.
pub fn definitions() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "name": "get_composition",
            "description": "Return what this workspace is composed of: the operators, \
                            the commons, the repos, and the protocols declared in its \
                            manifests, along with which protocol sources are actually \
                            present on disk. Call this to find out what exists before \
                            reasoning about it — the answer is derived from the \
                            manifest files, not from memory.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }),
        serde_json::json!({
            "name": "list_directory",
            "description": "List the entries of one directory in the workspace. \
                            Directories are marked with a trailing slash. Paths are \
                            relative to the workspace root; pass an empty string for \
                            the root itself. Credentials and build output are never \
                            listed. Use this to find out what is there before reading \
                            a file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the workspace root; \
                                        \"\" for the root.",
                    },
                },
                "required": ["path"],
            },
        }),
        serde_json::json!({
            "name": "read_file",
            "description": "Read a page of one text file in the workspace, with line \
                            numbers so you can cite what you read. Paths are relative \
                            to the workspace root. Reads are paginated: if the page \
                            ends before the file does, the result says so and gives \
                            the offset to continue from. No file is too large to \
                            read — it is only too large to read at once.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based line number to start at. Defaults to 1.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": format!(
                            "How many lines to read. Defaults to {DEFAULT_LINES}, \
                             capped at {MAX_LINES}."
                        ),
                    },
                },
                "required": ["path"],
            },
        }),
        serde_json::json!({
            "name": "find_files",
            "description": "Find files by a glob pattern over their path, e.g. \
                            \"**/2026-07-30-*\" or \"operators/**/*.md\". Searches \
                            the modules this workspace declares; content that is \
                            not composed (reference clones, git worktrees) is not \
                            searched unless you name it with `path`. Use this when \
                            you know roughly what a file is called and not where it is.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "A glob matched against the workspace-relative path.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional. Restrict the search to this directory, \
                                        which may be outside the composed modules.",
                    },
                },
                "required": ["pattern"],
            },
        }),
        serde_json::json!({
            "name": "search",
            "description": "Search the contents of this workspace's files for a \
                            regular expression, returning each matching line with \
                            its path and line number. A plain string is a valid \
                            regex. Searches the modules this workspace declares; \
                            content that is not composed (reference clones, git \
                            worktrees) is not searched unless you name it with \
                            `path`. Long lines are windowed around the match. Use \
                            this to find out where something is said before reading \
                            the file that says it.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A regular expression. A literal string works.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional. Restrict the search to this directory, \
                                        which may be outside the composed modules.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Optional. Defaults to false.",
                    },
                },
                "required": ["query"],
            },
        }),
    ]
}

/// Execute one tool call.
///
/// **An unknown name is an error result, never nothing.** Both surveyed
/// harnesses that hit this case guarantee a result rather than letting the pair
/// dangle, and they are right for a reason measured here: a `tool_use` with no
/// `tool_result` is a 400 that kills every later turn. A stale or misspelled
/// name must come back as a refusal the model can read.
pub fn dispatch(name: &str, input: &serde_json::Value, root: &Path) -> Result<String, ToolError> {
    match name {
        "get_composition" => get_composition(root),
        "list_directory" => list_directory(root, required_str(input, "path", name)?),
        "read_file" => read_page(
            root,
            required_str(input, "path", name)?,
            optional_usize(input, "offset", 1)?,
            optional_usize(input, "limit", DEFAULT_LINES)?,
        ),
        "find_files" => crate::search::find_files(
            root,
            required_str(input, "pattern", name)?,
            optional_str(input, "path"),
        ),
        "search" => crate::search::search(
            root,
            required_str(input, "query", name)?,
            optional_str(input, "path"),
            input.get("case_sensitive").and_then(|v| v.as_bool()).unwrap_or(false),
        ),
        other => Err(ToolError::new(
            "unknown_tool",
            format!("no tool named {other}"),
        )),
    }
}

/// The files a workspace's composition is byte-derived from (C-3).
///
/// **One list, three readers:** the payload builder hashes them, DD-1's event
/// records those hashes, and the projection re-checks them for drift. A second
/// copy would mean a manifest the engine parses but never watches.
pub const MANIFEST_FILES: [&str; 2] = ["modules.toml", "modules.local.toml"];

/// The composition as a durable event's `data` — **DD-1**.
///
/// Same builder as everything else here, reshaped into two halves that are
/// consumed by different code:
///
/// - **`manifests`** keeps its sha256 digests, because that is what the
///   projection re-checks every turn. A workspace's manifests are exactly the
///   thing that changes mid-session, and without this the session goes on
///   describing a workspace that no longer exists.
/// - **`composition`** is what the model reads, with the protocol-source
///   digests stripped. They are **not** re-checked and must not appear: a
///   digest nobody verifies is a promise the projection does not keep, and a
///   protocol body (a board, an athanor) changes constantly without the
///   composition changing at all. `present` survives, because presence is the
///   C-4 question that actually matters.
pub fn composition_event_data(root: &Path) -> Result<serde_json::Value, ToolError> {
    let mut body = composition_payload(root)?;
    let obj = body
        .as_object_mut()
        .ok_or_else(|| ToolError::new("composition_unreadable", "composition is not an object"))?;

    let manifests = obj
        .remove("manifests")
        .unwrap_or_else(|| serde_json::json!([]));

    if let Some(sources) = obj.get_mut("protocol_sources").and_then(|v| v.as_array_mut()) {
        for entry in sources {
            entry.as_object_mut().map(|o| o.remove("sha256"));
        }
    }

    Ok(serde_json::json!({ "manifests": manifests, "composition": body }))
}

/// The full composition payload — the single implementation, shared by the
/// `/composition` route and by the tool below.
///
/// C-3 as running code: byte-derived from the manifests by a TOML parser, never
/// re-derived by a model. The rule exists because a local model once read
/// commented-out examples as a live composition, and a parser makes that
/// structurally impossible — comments are not data.
///
/// Carries the sha256 of every manifest and every resolved protocol source.
/// Nothing consumes them yet; they cost one hash over bytes already in memory
/// and mean "which bytes were in that window" stays answerable.
pub fn composition_payload(root: &Path) -> Result<serde_json::Value, ToolError> {
    let unreadable = |e: String| ToolError::new("composition_unreadable", e);

    let composition = armillary_composition::parse_workspace(root)
        .map_err(|e| unreadable(e.to_string()))?;

    let mut manifests = Vec::new();
    for name in MANIFEST_FILES {
        if let Ok(bytes) = std::fs::read(root.join(name)) {
            manifests.push(serde_json::json!({
                "path": name,
                "sha256": crate::hash::sha256_hex(&bytes),
            }));
        }
    }

    // C-4: a protocol whose source is not present is reported absent, not an
    // error. Through the guard, not `root.join` — an absolute `source` would
    // otherwise be read verbatim, since `Path::join` with an absolute argument
    // discards the base.
    let protocol_sources: Vec<serde_json::Value> = composition
        .protocols
        .iter()
        .map(|p| {
            match crate::guard::resolve(root, &p.source)
                .and_then(|path| std::fs::read(&path).map_err(|_| crate::guard::GuardError::NotFound))
            {
                Ok(bytes) => serde_json::json!({
                    "name": p.name, "path": p.source, "present": true,
                    "sha256": crate::hash::sha256_hex(&bytes),
                }),
                Err(_) => serde_json::json!({
                    "name": p.name, "path": p.source, "present": false,
                }),
            }
        })
        .collect();

    let mut body = serde_json::to_value(&composition).map_err(|e| unreadable(e.to_string()))?;
    body["manifests"] = serde_json::json!(manifests);
    body["protocol_sources"] = serde_json::json!(protocol_sources);
    Ok(body)
}

/// The model-facing composition: `composition_payload` with the digests removed.
///
/// **The sha256 values are stripped deliberately, and the strip is real** — this
/// calls the same builder the `/composition` route does, so the two cannot drift
/// into disagreeing about what is composed. Roughly a quarter of that payload is
/// hex a model can neither verify nor act on; it exists for drift detection,
/// which is an engine concern.
///
/// `present` survives, because presence is the C-4 question a model actually
/// needs answered: a protocol whose source is missing is skipped, not an error.
fn get_composition(root: &Path) -> Result<String, ToolError> {
    let mut body = composition_payload(root)?;

    if let Some(manifests) = body.get_mut("manifests").and_then(|m| m.as_array_mut()) {
        for entry in manifests {
            entry.as_object_mut().map(|o| o.remove("sha256"));
        }
    }
    if let Some(sources) = body
        .get_mut("protocol_sources")
        .and_then(|m| m.as_array_mut())
    {
        for entry in sources {
            entry.as_object_mut().map(|o| o.remove("sha256"));
        }
    }

    serde_json::to_string_pretty(&body)
        .map_err(|e| ToolError::new("composition_unreadable", e.to_string()))
}

/// A directory listing is a thing a phone renders and a thing a model pays
/// for. Unbounded, one response can carry every entry of a build-index store —
/// 1,147 in this workspace's largest.
pub const MAX_ENTRIES: usize = 500;

/// How many lines `read_file` returns when the caller does not say.
pub const DEFAULT_LINES: usize = 500;

/// The most lines one page may carry, however large a `limit` asks for.
pub const MAX_LINES: usize = 2000;

#[derive(Serialize)]
pub struct Entry {
    pub name: String,
    pub dir: bool,
}

/// One directory listing, shared by `/tree` and by `list_directory`.
///
/// Synchronous and self-contained so it can be handed to a thread that is
/// allowed to block — `metadata()` follows symlinks, so one entry pointing at
/// a disconnected volume blocks for the mount timeout.
///
/// Returns the capped entries and the total the directory actually holds. The
/// caller decides how to say "this is a prefix"; both callers must say it.
pub fn list_entries(root: &Path, path: &str) -> Result<(Vec<Entry>, usize), ToolError> {
    let resolved = crate::guard::resolve(root, path)?;

    let read = std::fs::read_dir(&resolved)
        .map_err(|_| ToolError::new("not_a_directory", format!("{path} is not a directory")))?;

    let mut entries: Vec<Entry> = Vec::new();
    for item in read.flatten() {
        let name = item.file_name().to_string_lossy().to_string();
        if crate::guard::is_hidden_from_listings(&name) {
            continue;
        }
        // `file_type` does not follow symlinks; `metadata` does. A symlinked
        // directory should browse as a directory — this workspace routes real
        // content through symlinks (`models -> operators`, CLAUDE.local.md into
        // the commons). A dangling link resolves to nothing and is simply not
        // an entry, which is the presence-gated reading.
        let Ok(meta) = item.path().metadata() else {
            continue;
        };
        entries.push(Entry {
            name,
            dir: meta.is_dir(),
        });
    }

    entries.sort_by(|a, b| {
        b.dir
            .cmp(&a.dir)
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
            .then_with(|| a.name.cmp(&b.name))
    });

    // Sorted before truncating, so the prefix is stable and meaningful rather
    // than whatever the filesystem happened to return first.
    let total = entries.len();
    entries.truncate(MAX_ENTRIES);
    Ok((entries, total))
}

/// The model-facing listing.
///
/// **D7 — `total` and the prefix warning are rendered, not dropped.** `tree.rs`
/// already argues why the fields exist for the Explorer: a silently short list
/// reads exactly like a complete one. A model has strictly less recourse than
/// a human scrolling, so the same rule applies with more force, and the
/// warning names the recovery (narrow the path) rather than just the fact.
fn list_directory(root: &Path, path: &str) -> Result<String, ToolError> {
    let (entries, total) = list_entries(root, path)?;

    let shown = entries.len();
    let where_ = if path.is_empty() { "." } else { path };
    let mut out = format!("{where_} — {total} entries\n");
    for e in entries {
        out.push_str(&e.name);
        if e.dir {
            out.push('/');
        }
        out.push('\n');
    }
    if shown < total {
        out.push_str(&format!(
            "[showing the first {shown} of {total}; list a subdirectory to narrow]\n"
        ));
    }
    Ok(out)
}

/// 1 MiB. The whole-file ceiling, and **the route's alone**: `/file` is
/// all-or-nothing because the Explorer has nowhere to put a second page, so
/// over the ceiling it still refuses. The tool pages instead and never meets
/// this constant — which is the point of D15. No file is too large for a model
/// to read; it is only too large to read at once.
const MAX_BYTES: u64 = 1024 * 1024;

/// The most bytes one line may contribute before it is cut.
///
/// A minified bundle or a base64 blob is one line of several megabytes. Without
/// this, a single line defeats every other cap on this page.
const MAX_LINE_BYTES: usize = 4000;

/// The most bytes one page may carry, whatever the line budget allows.
///
/// ~16k tokens. This is the cap that usually binds, and it binds for a reason
/// the line count cannot see: a tool result is **durable** and re-projected
/// every round (D9), so its cost is paid once per round for the rest of the
/// turn, not once.
const MAX_PAGE_BYTES: usize = 64 * 1024;

/// The gate every file read passes, whichever caller asked.
///
/// Resolve through the guard (D2), refuse a directory, then refuse a type that
/// is not served as text. **Ordering is load-bearing and inherited from
/// `file.rs`:** the type check comes before any size check, so a 300 MB `.zip`
/// reads as "can't open this type" rather than "too large" — the type is the
/// true reason and the size would be a misleading one.
fn open_readable(root: &Path, path: &str) -> Result<(PathBuf, u64), ToolError> {
    let resolved = crate::guard::resolve(root, path)?;

    let meta = resolved
        .metadata()
        .map_err(|_| ToolError::new("not_found", format!("nothing at {path}")))?;
    if meta.is_dir() {
        return Err(ToolError::new(
            "is_a_directory",
            format!("{path} is a directory"),
        ));
    }

    let name = resolved
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    if !crate::guard::is_openable(&name) {
        return Err(ToolError::new(
            "not_openable",
            format!("{path} is not a type served as text"),
        ));
    }

    Ok((resolved, meta.len()))
}

/// One whole file, for the `/file` route. Unchanged behaviour, shared gate.
pub fn read_whole(root: &Path, path: &str) -> Result<(String, u64, String), ToolError> {
    let (resolved, bytes) = open_readable(root, path)?;

    // Checked from metadata, so an oversized file is never loaded in order to
    // be rejected.
    if bytes > MAX_BYTES {
        return Err(ToolError::new("too_large", format!("{bytes} bytes")));
    }

    let raw = std::fs::read(&resolved)
        .map_err(|_| ToolError::new("not_found", format!("nothing at {path}")))?;
    let sha256 = crate::hash::sha256_hex(&raw);

    // Binary gets a refusal rather than a guess. Inventing an encoding would be
    // less honest than saying no.
    let text = String::from_utf8(raw)
        .map_err(|_| ToolError::new("not_text", format!("{path} is not valid UTF-8")))?;

    Ok((sha256, bytes, text))
}

/// One line, read with a byte cap so a single line cannot defeat the page cap.
pub(crate) struct Line {
    pub(crate) text: String,
    pub(crate) truncated: bool,
}

/// Consume the remainder of an over-long line. Bounded per read so a
/// multi-megabyte line is skipped without ever being held in memory.
fn discard_to_newline(reader: &mut impl BufRead) -> std::io::Result<()> {
    let mut sink = Vec::new();
    loop {
        sink.clear();
        let n = reader.by_ref().take(64 * 1024).read_until(b'\n', &mut sink)?;
        if n == 0 || sink.ends_with(b"\n") {
            return Ok(());
        }
    }
}

/// The next line, or `None` at end of file.
///
/// Reads `MAX_LINE_BYTES + 1` so "at the cap" and "over the cap" are
/// distinguishable, and never holds more than that regardless of line length.
pub(crate) fn next_line(reader: &mut impl BufRead, path: &str) -> Result<Option<Line>, ToolError> {
    let unreadable = || ToolError::new("read_failed", format!("could not read {path}"));

    let mut buf = Vec::new();
    let n = reader
        .by_ref()
        .take(MAX_LINE_BYTES as u64 + 1)
        .read_until(b'\n', &mut buf)
        .map_err(|_| unreadable())?;
    if n == 0 {
        return Ok(None);
    }

    let mut truncated = false;
    if !buf.ends_with(b"\n") && n > MAX_LINE_BYTES {
        truncated = true;
        buf.truncate(MAX_LINE_BYTES);
        discard_to_newline(reader).map_err(|_| unreadable())?;
    }

    while matches!(buf.last(), Some(b'\n') | Some(b'\r')) {
        buf.pop();
    }

    let text = match std::str::from_utf8(&buf) {
        Ok(s) => s.to_string(),
        // A cut in the middle of a character is the one invalid sequence this
        // function creates itself, and it can only ever be in the last three
        // bytes. Anything earlier is the file's own, and gets the honest
        // refusal rather than a silent prefix of binary.
        Err(e) if truncated && e.valid_up_to() + 4 > buf.len() => {
            String::from_utf8_lossy(&buf[..e.valid_up_to()]).into_owned()
        }
        Err(_) => {
            return Err(ToolError::new(
                "not_text",
                format!("{path} is not valid UTF-8"),
            ))
        }
    };

    Ok(Some(Line { text, truncated }))
}

/// **D15 — one page of a file, with line numbers, and no terminal refusal.**
///
/// Before this, `/file` was all-or-nothing: over the ceiling `too_large` was
/// permanent and the model could never read that file, while under it a 900 KB
/// read was durable, re-projected every round, and removable only by evicting
/// the whole turn plus its batch. **One bad read bricked the stream.** Paging
/// removes that whole class of unrecoverable session death.
///
/// Three caps, each announced when it bites, each with the offset to continue
/// from — a cap the model cannot see is a cap it cannot work around.
fn read_page(root: &Path, path: &str, offset: usize, limit: usize) -> Result<String, ToolError> {
    let (resolved, _) = open_readable(root, path)?;
    let file = std::fs::File::open(&resolved)
        .map_err(|_| ToolError::new("not_found", format!("nothing at {path}")))?;
    let mut reader = BufReader::new(file);

    // Offsets are 1-based and a model will send 0. Refusing would be correct
    // and useless.
    let start = offset.max(1);
    let limit = limit.clamp(1, MAX_LINES);

    let mut seen = 0usize;
    let mut emitted = 0usize;
    let mut body = String::new();
    let mut more = false;

    while let Some(line) = next_line(&mut reader, path)? {
        seen += 1;
        if seen < start {
            continue;
        }
        if emitted == limit {
            more = true;
            break;
        }
        let mut rendered = format!("{seen:>6}\t{}", line.text);
        if line.truncated {
            rendered.push_str(" …[line truncated]");
        }
        rendered.push('\n');

        // `emitted > 0` so a single line larger than the whole page budget is
        // still served rather than producing an empty page forever.
        if emitted > 0 && body.len() + rendered.len() > MAX_PAGE_BYTES {
            more = true;
            break;
        }
        body.push_str(&rendered);
        emitted += 1;
    }

    // Every branch below returns non-empty content: an empty text block is a
    // 400, so "there is nothing here" must still be something.
    if seen == 0 {
        return Ok(format!("{path} — the file is empty\n"));
    }
    if emitted == 0 {
        return Ok(format!(
            "{path} — {seen} lines; offset {start} is past the end of the file\n"
        ));
    }

    let last = start + emitted - 1;
    let footer = if more {
        format!("[more lines follow; call read_file with offset={}]\n", last + 1)
    } else {
        "[end of file]\n".to_string()
    };
    Ok(format!("{path} lines {start}-{last}\n{body}{footer}"))
}

/// Pull a required string argument, or refuse in a way the model can act on.
///
/// A missing argument must never panic and must never reach the filesystem as
/// an empty string: `path: ""` is a *legal* request for the workspace root, so
/// defaulting would silently answer a question nobody asked.
fn required_str<'a>(
    input: &'a serde_json::Value,
    key: &str,
    tool: &str,
) -> Result<&'a str, ToolError> {
    input
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::new("invalid_input", format!("{tool} requires a `{key}` string")))
}

/// An optional whole-number argument.
///
/// A key that is present but not a number is a **refusal, not a default**. A
/// model that sends `offset: "10"` and silently gets page one would read the
/// wrong window and have no way to know it.
fn optional_usize(
    input: &serde_json::Value,
    key: &str,
    default: usize,
) -> Result<usize, ToolError> {
    match input.get(key) {
        None | Some(serde_json::Value::Null) => Ok(default),
        Some(v) => v.as_u64().map(|n| n as usize).ok_or_else(|| {
            ToolError::new(
                "invalid_input",
                format!("`{key}` must be a whole number, not {v}"),
            )
        }),
    }
}

/// An optional string argument. Absent and empty are the same thing here —
/// see `search::resolve_domain` for why `""` must not mean the workspace root.
fn optional_str<'a>(input: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    input.get(key).and_then(|v| v.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A workspace shaped like the real one: a public manifest, a private
    /// overlay, one protocol whose source exists and one whose source does not.
    fn workspace() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("modules.toml"),
            "# a commented-out example that MUST NOT read as a declaration\n\
             # [[repos]]\n\
             # name = \"ghost\"\n\
             # path = \"repos/ghost\"\n\
             [router]\n\
             contains = [\"CLAUDE.md\"]\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("modules.local.toml"),
            "[[operators]]\nname = \"tycho\"\npath = \"operators/tycho\"\n\n\
             [[repos]]\nname = \"kairos-engine\"\npath = \"repos/kairos-engine\"\n\n\
             [[protocols]]\nname = \"board\"\nsource = \"present.md\"\nload = \"boot\"\n\n\
             [[protocols]]\nname = \"athanor\"\nsource = \"absent.md\"\nload = \"on-demand\"\n",
        )
        .unwrap();
        fs::write(dir.path().join("present.md"), "# board").unwrap();
        dir
    }

    #[test]
    fn get_composition_reports_what_the_manifests_declare() {
        let dir = workspace();
        let out = dispatch("get_composition", &serde_json::json!({}), dir.path()).unwrap();

        assert!(out.contains("tycho"), "{out}");
        assert!(out.contains("kairos-engine"), "{out}");
        assert!(out.contains("board"), "{out}");
    }

    #[test]
    fn a_commented_out_entry_is_not_a_declaration() {
        // C-3, and the reason it exists: a local model once read commented-out
        // examples as a live composition. A TOML parser makes it impossible.
        let dir = workspace();
        let out = dispatch("get_composition", &serde_json::json!({}), dir.path()).unwrap();

        assert!(
            !out.contains("ghost"),
            "a commented-out repo reached the model: {out}"
        );
    }

    #[test]
    fn protocol_presence_is_reported_because_a_missing_source_is_skipped_not_an_error() {
        let dir = workspace();
        let out = dispatch("get_composition", &serde_json::json!({}), dir.path()).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();

        let by_name = |n: &str| -> bool {
            parsed["protocol_sources"]
                .as_array()
                .unwrap()
                .iter()
                .find(|p| p["name"] == n)
                .unwrap()["present"]
                .as_bool()
                .unwrap()
        };

        assert!(by_name("board"), "present.md exists: {out}");
        assert!(!by_name("athanor"), "absent.md does not exist: {out}");
    }

    #[test]
    fn the_sha256_digests_stay_out_of_the_model_facing_payload() {
        // A quarter of `/composition`'s bytes are hex a model cannot verify and
        // cannot act on. Drift detection is an engine concern.
        let dir = workspace();
        let out = dispatch("get_composition", &serde_json::json!({}), dir.path()).unwrap();

        assert!(!out.contains("sha256"), "{out}");
    }

    #[test]
    fn the_shared_builder_carries_the_digests_the_tool_strips() {
        // Without this, "no sha256 in the tool payload" is true by construction
        // rather than by design — a test that passes identically whether or not
        // the strip exists. Mutation-checked: removing the strip reddens the
        // pair below, not neither.
        let dir = workspace();
        let full = composition_payload(dir.path()).unwrap();

        assert!(
            full["manifests"][0]["sha256"].is_string(),
            "the route's payload must carry manifest digests: {full}"
        );
        assert!(
            full["protocol_sources"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["sha256"].is_string()),
            "a present protocol source must be hashed: {full}"
        );
    }

    #[test]
    fn a_bare_clone_composes_nothing_and_that_is_not_an_error() {
        // C-4: presence-gated throughout. A bare clone is a working host.
        let dir = tempfile::tempdir().unwrap();
        let out = dispatch("get_composition", &serde_json::json!({}), dir.path())
            .expect("a bare clone is a working host, not a failure");

        assert!(!out.contains("tycho"), "{out}");
    }

    #[test]
    fn an_unknown_tool_name_returns_an_error_result_rather_than_nothing() {
        // A tool_use with no tool_result is a 400 that kills every later turn,
        // so a stale or misspelled name must come back as something the model
        // can read.
        let dir = workspace();
        let err = dispatch("read_the_future", &serde_json::json!({}), dir.path()).unwrap_err();

        assert_eq!(err.status, "unknown_tool");
        assert!(err.detail.contains("read_the_future"));
    }

    #[test]
    fn the_definition_set_is_ordered_and_schema_shaped() {
        let defs = definitions();
        let names: Vec<&str> = defs.iter().map(|d| d["name"].as_str().unwrap()).collect();

        // Order is part of the cached prefix, so it is pinned rather than
        // incidental.
        assert_eq!(
            names,
            ["get_composition", "list_directory", "read_file", "find_files", "search"]
        );
        for d in &defs {
            assert_eq!(d["input_schema"]["type"], "object");
            assert!(
                d["description"].as_str().unwrap().len() > 40,
                "the description is what decides whether a model calls it: {d}"
            );
        }
    }

    // ---- list_directory ----

    /// A small tree with one subdirectory, one prose file, and the two shapes
    /// the guard refuses: a credential and a build directory.
    fn tree_fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("commons")).unwrap();
        fs::write(dir.path().join("README.md"), "# hello").unwrap();
        fs::write(dir.path().join(".env"), "TOKEN=hunter2").unwrap();
        fs::create_dir(dir.path().join("node_modules")).unwrap();
        dir
    }

    #[test]
    fn list_directory_names_entries_and_marks_which_are_directories() {
        let dir = tree_fixture();
        let out = dispatch(
            "list_directory",
            &serde_json::json!({ "path": "" }),
            dir.path(),
        )
        .unwrap();

        assert!(out.contains("commons/"), "a directory is marked: {out}");
        assert!(out.contains("README.md"), "{out}");
        assert!(
            !out.contains("README.md/"),
            "a file must not be marked as a directory: {out}"
        );
    }

    #[test]
    fn a_listing_reports_its_total_and_says_so_when_it_is_a_prefix() {
        // D7, and `tree.rs`'s own argument for why the fields exist: a silently
        // short list reads exactly like a complete one.
        let dir = tempfile::tempdir().unwrap();
        for n in 0..MAX_ENTRIES + 11 {
            fs::write(dir.path().join(format!("note-{n:04}.md")), "x").unwrap();
        }
        let out = dispatch("list_directory", &serde_json::json!({ "path": "" }), dir.path())
            .unwrap();

        assert!(
            out.contains(&(MAX_ENTRIES + 11).to_string()),
            "the total must survive into the text: {out}"
        );
        assert!(
            out.to_lowercase().contains("first"),
            "a truncated listing must say it is a prefix: {out}"
        );
    }

    #[test]
    fn a_complete_listing_reports_its_total_without_claiming_to_be_truncated() {
        // Mutation-found gap: with the total only asserted on the truncated
        // path, the truncation footer's own "of 511" satisfied it and the
        // header's count was tested by nothing.
        let dir = tree_fixture();
        let out = dispatch("list_directory", &serde_json::json!({ "path": "" }), dir.path())
            .unwrap();

        assert!(out.contains("2 entries"), "commons/ and README.md: {out}");
        assert!(!out.to_lowercase().contains("first"), "{out}");
    }

    #[test]
    fn credentials_and_build_output_never_reach_a_listing() {
        let dir = tree_fixture();
        let out = dispatch("list_directory", &serde_json::json!({ "path": "" }), dir.path())
            .unwrap();

        assert!(!out.contains(".env"), "{out}");
        assert!(!out.contains("node_modules"), "{out}");
    }

    #[test]
    fn an_empty_directory_still_produces_non_empty_content() {
        // An empty text block is a 400 (measured). "Nothing here" must still be
        // something.
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("empty")).unwrap();
        let out = dispatch(
            "list_directory",
            &serde_json::json!({ "path": "empty" }),
            dir.path(),
        )
        .unwrap();

        assert!(!out.trim().is_empty(), "an empty listing rendered to nothing");
    }

    #[test]
    fn list_directory_refuses_a_file_and_names_the_recovery() {
        let dir = tree_fixture();
        let err = dispatch(
            "list_directory",
            &serde_json::json!({ "path": "README.md" }),
            dir.path(),
        )
        .unwrap_err();

        assert_eq!(err.status, "not_a_directory");
    }

    #[test]
    fn list_directory_carries_the_guards_own_code_for_a_denied_path() {
        let dir = tree_fixture();
        let err = dispatch(
            "list_directory",
            &serde_json::json!({ "path": "node_modules" }),
            dir.path(),
        )
        .unwrap_err();

        assert_eq!(err.status, "denied_noise");
    }

    #[test]
    fn a_tool_call_missing_its_required_path_is_an_error_result_not_a_panic() {
        let dir = tree_fixture();
        for name in ["list_directory", "read_file"] {
            let err = dispatch(name, &serde_json::json!({}), dir.path()).unwrap_err();
            assert_eq!(err.status, "invalid_input", "{name}");
            assert!(err.detail.contains("path"), "{name}: {}", err.detail);
        }
    }

    // ---- read_file (D15) ----

    fn read(dir: &Path, input: serde_json::Value) -> String {
        dispatch("read_file", &input, dir).unwrap()
    }

    /// A file of `n` lines, each `line-0001`-shaped so a window is identifiable.
    fn lined_file(dir: &Path, name: &str, n: usize) {
        let body: String = (1..=n).map(|i| format!("line-{i:04}\n")).collect();
        fs::write(dir.join(name), body).unwrap();
    }

    #[test]
    fn read_file_numbers_its_lines_so_a_model_can_cite_one() {
        // ycc's affordance, and the reason it matters here: without numbers a
        // model can quote a file but cannot point at it.
        let dir = tempfile::tempdir().unwrap();
        lined_file(dir.path(), "notes.md", 3);
        let out = read(dir.path(), serde_json::json!({ "path": "notes.md" }));

        assert!(out.contains("1\tline-0001"), "{out}");
        assert!(out.contains("3\tline-0003"), "{out}");
    }

    #[test]
    fn read_file_returns_the_window_it_was_asked_for() {
        let dir = tempfile::tempdir().unwrap();
        lined_file(dir.path(), "notes.md", 50);
        let out = read(
            dir.path(),
            serde_json::json!({ "path": "notes.md", "offset": 10, "limit": 3 }),
        );

        assert!(out.contains("line-0010"), "{out}");
        assert!(out.contains("line-0012"), "{out}");
        assert!(!out.contains("line-0009"), "before the window: {out}");
        assert!(!out.contains("line-0013"), "after the window: {out}");
    }

    #[test]
    fn a_page_that_ends_short_of_the_file_gives_the_offset_to_continue_from() {
        // D6′: render the recovery action, not just the fact. A page that says
        // "there is more" and not "ask for line 13" leaves the model guessing.
        let dir = tempfile::tempdir().unwrap();
        lined_file(dir.path(), "notes.md", 50);
        let out = read(
            dir.path(),
            serde_json::json!({ "path": "notes.md", "offset": 10, "limit": 3 }),
        );

        assert!(out.contains("offset=13"), "{out}");
    }

    #[test]
    fn the_last_page_says_it_is_the_last_rather_than_inviting_another_call() {
        let dir = tempfile::tempdir().unwrap();
        lined_file(dir.path(), "notes.md", 3);
        let out = read(dir.path(), serde_json::json!({ "path": "notes.md" }));

        assert!(out.contains("end of file"), "{out}");
        assert!(!out.contains("offset="), "no next page exists: {out}");
    }

    #[test]
    fn a_file_over_the_byte_ceiling_returns_a_page_rather_than_a_terminal_refusal() {
        // The whole of D15. `/file` answers `too_large` and the model can never
        // read that file again — one bad read used to brick the stream.
        let dir = tempfile::tempdir().unwrap();
        let big: String = (1..=40_000)
            .map(|i| format!("line-{i:04} {}\n", "x".repeat(40)))
            .collect();
        fs::write(dir.path().join("huge.md"), &big).unwrap();
        assert!(big.len() as u64 > super::MAX_BYTES, "fixture must exceed the ceiling");

        let out = read(dir.path(), serde_json::json!({ "path": "huge.md" }));

        assert!(out.contains("line-0001"), "{out}");
        assert!(out.contains("offset="), "a page must invite the next one: {out}");
    }

    #[test]
    fn a_page_stops_at_the_byte_cap_before_it_spends_its_line_budget() {
        // A tool result is durable and re-projected every round (D9), so the
        // cap that binds first must be bytes, not lines.
        let dir = tempfile::tempdir().unwrap();
        let body: String = (1..=DEFAULT_LINES).map(|i| format!("{i:04} {}\n", "y".repeat(300))).collect();
        fs::write(dir.path().join("dense.md"), &body).unwrap();

        let out = read(dir.path(), serde_json::json!({ "path": "dense.md" }));

        assert!(out.len() < body.len(), "the whole file came through: {} bytes", out.len());
        assert!(out.contains("offset="), "a capped page must say how to continue: {out}");
    }

    #[test]
    fn an_over_long_line_is_truncated_and_the_line_admits_it() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("wide.md"),
            format!("{}\nshort\n", "z".repeat(MAX_LINE_BYTES * 2)),
        )
        .unwrap();

        let out = read(dir.path(), serde_json::json!({ "path": "wide.md" }));

        assert!(out.contains("line truncated"), "{out}");
        // The next line must still be reachable — a long line must not eat the
        // rest of the page.
        assert!(out.contains("short"), "{out}");
    }

    #[test]
    fn truncating_an_over_long_line_never_splits_a_character() {
        // The cap is measured in bytes and this workspace writes `.爻` files:
        // three bytes per character, so the cap lands mid-character by default.
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("wide.爻"),
            format!("{}\n", "爻".repeat(MAX_LINE_BYTES)),
        )
        .unwrap();

        let out = read(dir.path(), serde_json::json!({ "path": "wide.爻" }));

        assert!(
            !out.contains('\u{FFFD}'),
            "a character was split and lossily replaced: {out}"
        );
    }

    #[test]
    fn an_empty_file_says_so_rather_than_rendering_to_nothing() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("empty.md"), "").unwrap();

        let out = read(dir.path(), serde_json::json!({ "path": "empty.md" }));

        assert!(!out.trim().is_empty(), "an empty text block is a 400");
        assert!(out.contains("empty"), "{out}");
    }

    #[test]
    fn an_offset_past_the_end_reports_the_length_rather_than_an_empty_page() {
        let dir = tempfile::tempdir().unwrap();
        lined_file(dir.path(), "notes.md", 5);

        let out = read(
            dir.path(),
            serde_json::json!({ "path": "notes.md", "offset": 99 }),
        );

        assert!(out.contains('5'), "the real length must be reported: {out}");
        assert!(!out.trim().is_empty());
    }

    #[test]
    fn an_offset_of_zero_reads_from_the_first_line() {
        // Offsets are 1-based and a model will send 0. Refusing would be
        // correct and useless.
        let dir = tempfile::tempdir().unwrap();
        lined_file(dir.path(), "notes.md", 3);

        let out = read(
            dir.path(),
            serde_json::json!({ "path": "notes.md", "offset": 0 }),
        );

        assert!(out.contains("line-0001"), "{out}");
        assert!(
            out.contains("lines 1-3"),
            "the header must report the line it actually started at: {out}"
        );
    }

    #[test]
    fn a_limit_above_the_cap_is_clamped_rather_than_honoured() {
        let dir = tempfile::tempdir().unwrap();
        lined_file(dir.path(), "notes.md", MAX_LINES + 100);

        let out = read(
            dir.path(),
            serde_json::json!({ "path": "notes.md", "limit": 99_999 }),
        );
        let last = MAX_LINES + 1;
        assert!(!out.contains(&format!("line-{last:04}")), "the cap did not hold");
    }

    #[test]
    fn read_file_refuses_what_the_guard_and_the_type_gate_refuse() {
        let dir = tree_fixture();
        fs::write(dir.path().join("icon.png"), [0x89, 0x50, 0x4e, 0x47]).unwrap();
        fs::write(dir.path().join("bad.md"), [0xff, 0xfe, 0x00]).unwrap();

        for (path, status) in [
            ("commons", "is_a_directory"),
            (".env", "denied_credential"),
            ("icon.png", "not_openable"),
            ("bad.md", "not_text"),
            ("nope.md", "not_found"),
            ("../escape.md", "outside_workspace"),
        ] {
            let err = dispatch("read_file", &serde_json::json!({ "path": path }), dir.path())
                .unwrap_err();
            assert_eq!(err.status, status, "{path}");
        }
    }

    #[test]
    fn the_route_and_the_tool_share_one_gate() {
        // Not a style point. The gate is `guard::resolve` plus the openable
        // check, and two copies would eventually disagree about which of them
        // serves a credential.
        let dir = tree_fixture();
        let via_route = read_whole(dir.path(), ".env").unwrap_err();
        let via_tool = dispatch("read_file", &serde_json::json!({ "path": ".env" }), dir.path())
            .unwrap_err();

        assert_eq!(via_route.status, via_tool.status);
        assert_eq!(via_route.status, "denied_credential");
    }
}
