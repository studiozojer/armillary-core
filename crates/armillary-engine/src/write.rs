//! Changing things — the verbs that answer "make it so" rather than "where is it".
//!
//! **The gate is the read gate** (WD-1). The noise list already denies
//! `node_modules`, `target`, `.build`, `.worktrees`, `dist`, `.next`, `.expo`,
//! `deriveddata`, `.venv`, `.gradle`, `pods` and the engine's own `.armillary`,
//! plus credentials, plus anything `judge` refuses — and every one of those
//! denials transfers to writing unchanged. There is no write-specific denylist,
//! because there is nothing to put on it.
//!
//! What writing needs that reading never did is exactly two things: a way to
//! name a path that does not exist yet (`guard::resolve_for_create`), and one
//! authorization check the guard cannot hold, because the guard is a pure
//! function of `(root, path)` with no session state.
//!
//! Writes land on **real files on the engine's host**, in the real git
//! checkout, immediately. There is no sandbox and no working copy.

use crate::tools::{Effect, ToolCtx, ToolError, ToolOutcome};
use std::path::{Path, PathBuf};

/// **WD-15.** Writes serialize, process-wide.
///
/// Instances are the product: several live session windows share one engine
/// process and one workspace root. `edit_file`'s read → count → replace → write
/// is a read-modify-write with no lock, and `write_file`'s `before` hash can be
/// captured from bytes another instance has already replaced — putting a false
/// chain in the one record D-1 exists to make trustworthy. `begin_turn` bounds
/// one turn per *stream*, which is no protection across streams.
///
/// A single mutex held across resolve → hash → write is adequate at our write
/// volume and far simpler than a keyed map. **A `static` rather than a handle
/// on `ToolCtx`** (the design's suggestion): the hazard is stated as
/// process-wide, and a handle threaded through `AppState` would serialize
/// per-`AppState` — tests alone construct several over one root. Same
/// guarantee, taken literally.
///
/// `tokio::sync::Mutex`, not `std` (changed with the pull extension): the
/// pull route's check-then-merge holds this guard across two awaited git
/// subprocesses, and a `std` guard is `!Send` — the compiler refuses it in
/// an async fn. The write path acquires it blocking (below), which is legal
/// where that path runs: under `spawn_blocking`, never on an executor thread.
static WRITE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// The write path's acquire. Blocking is safe here — and ONLY here — because
/// every write body runs under `dispatch`'s `spawn_blocking` (or a sync
/// test); `blocking_lock` panics by design if an executor thread tries.
pub(crate) fn write_lock() -> tokio::sync::MutexGuard<'static, ()> {
    WRITE_LOCK.blocking_lock()
}

/// The async acquire, for the one async holder: pull's check-then-merge
/// (WD-15 extended, survey 2026-08-06 weakness #2). A `write_file` landing
/// between `is_dirty` and `git merge` would defeat the dirty refusal that
/// route's doc presents as the point — exactly the cross-session hazard this
/// lock was built for, previously unextended over it.
pub(crate) async fn write_lock_async() -> tokio::sync::MutexGuard<'static, ()> {
    WRITE_LOCK.lock().await
}

/// Where a verb intends to write, and what it will have to make first.
struct WriteTarget {
    path: PathBuf,
    existed: bool,
    dirs_to_create: Vec<String>,
}

/// Resolve a path a verb intends to write.
///
/// **WD-2, and this ordering is the spec's load-bearing structural choice.**
/// `resolve` runs first, so editing or overwriting an existing file takes
/// today's fully-audited path — canonicalization, symlink-escape check,
/// credential judging — with zero new surface. `resolve_for_create` is
/// reachable only once `resolve` has said `NotFound`, which is to say only for
/// a genuine create. The common case cannot regress, and the new code is
/// reachable only when the old code has already said "nothing there".
fn resolve_for_write(root: &Path, path: &str) -> Result<WriteTarget, ToolError> {
    match crate::guard::resolve(root, path) {
        Ok(p) => Ok(WriteTarget {
            path: p,
            existed: true,
            dirs_to_create: Vec::new(),
        }),
        Err(crate::guard::GuardError::NotFound) => {
            let target = crate::guard::resolve_for_create(root, path)?;
            Ok(WriteTarget {
                path: target.path,
                existed: false,
                dirs_to_create: target.dirs_to_create,
            })
        }
        Err(e) => Err(e.into()),
    }
}

/// The workspace-relative form of a resolved path — WD-13, and G-1: never
/// absolute.
///
/// `requested` is the fallback and is unreachable in practice: `guard::resolve`
/// or `resolve_for_create` already canonicalized `root` to get here, so the
/// `strip_prefix` cannot fail. It exists so a resolved ABSOLUTE path can never
/// reach a durable event.
pub(crate) fn workspace_relative(root: &Path, resolved: &Path, requested: &str) -> String {
    root.canonicalize()
        .ok()
        .and_then(|r| {
            resolved
                .strip_prefix(&r)
                .ok()
                .map(|p| p.to_string_lossy().to_string())
        })
        .unwrap_or_else(|| requested.to_string())
}

/// **WD-9/WD-11.** Refuse a write to either manifest unless this instance was
/// granted it. Returns whether the path IS a manifest, so a caller can apply
/// WD-12's parse check where the grant passed.
///
/// **Not in `guard`, deliberately.** `guard::resolve` is a pure function of
/// `(root, path)` with no session state, and giving it one would put session
/// policy inside the module every read route shares — which is how a read route
/// silently acquires a write rule. WD-1 still holds: the guard made every
/// *path* decision already; this is an *authorization* decision layered on the
/// path it returned.
///
/// **Two comparisons, and the second is not redundant.**
///
/// - *Canonical* catches every spelling that resolves to the same file. In this
///   workspace `modules.local.toml` is a symlink into the commons, so the file
///   has two names — but the commons' name is a studio fact, and an engine
///   comparing against a literal `zojercommons/setup/...` would be
///   workspace-specific in the one crate that must not be (G-1) *and* would
///   fail to protect a workspace whose manifest is an ordinary file.
/// - *Lexical* catches the spelling that does not resolve. A manifest that does
///   not exist yet canonicalizes to nothing, so canonical-only fails OPEN on
///   exactly the workspace where an ungranted instance could CREATE one — and
///   `parse_workspace` runs per request, so it would take effect on the next
///   tool call in the same turn.
///
/// The lexical half also locks any file anywhere named `modules.toml` /
/// `modules.local.toml`. Accepted: belt and braces, and refusing a nested
/// manifest in a composed repo is the safe direction.
pub(crate) fn refuse_composition_write(
    ctx: &ToolCtx,
    resolved: &Path,
) -> Result<bool, ToolError> {
    let lexical = resolved
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .map(|name| {
            crate::tools::MANIFEST_FILES
                .iter()
                .any(|m| m.eq_ignore_ascii_case(&name))
        })
        .unwrap_or(false);
    let canonical = crate::tools::MANIFEST_FILES.iter().any(|manifest| {
        ctx.root
            .join(manifest)
            .canonicalize()
            .map(|canon| canon == resolved)
            .unwrap_or(false)
    });
    let is_manifest = lexical || canonical;

    if is_manifest && !ctx.may_write_composition {
        return Err(ToolError::new(
            "composition_locked",
            "this file defines the workspace's composition; this instance was not \
             created with permission to write it",
        ));
    }
    Ok(is_manifest)
}

/// **WD-12.** A manifest write must parse before it lands.
///
/// `parse_workspace` runs per request, so one malformed TOML write degrades
/// `get_composition` and every default-domain `search`/`find_files` for the
/// rest of the turn — immediately, with no rebuild, and with the model given
/// only a `composition_unreadable` to diagnose.
///
/// Scoped to the two manifests only. These are the files whose corruption
/// degrades the engine mid-turn; an ordinary `.toml` elsewhere is prose as far
/// as this engine is concerned.
fn refuse_unparseable_manifest(is_manifest: bool, content: &str) -> Result<(), ToolError> {
    if !is_manifest {
        return Ok(());
    }
    content.parse::<toml::Value>().map(|_| ()).map_err(|e| {
        ToolError::new(
            "invalid_input",
            format!("this file is a workspace manifest and the content is not valid TOML: {e}"),
        )
    })
}

/// **WD-14.** Read output echoed back as file content is refused.
///
/// `read_file` truncates any line over `MAX_LINE_BYTES` and marks it. Writing
/// that page back makes the cut permanent and records it as a clean `modified`.
/// Checked against the same constant the renderer writes, so the two cannot
/// drift.
fn refuse_truncated_echo(content: &str) -> Result<(), ToolError> {
    if content.contains(crate::tools::TRUNCATION_MARKER) {
        return Err(ToolError::new(
            "invalid_input",
            format!(
                "the content carries `{}` — that is read_file's display text, not the \
                 file's contents, and writing it back would destroy the truncated bytes. \
                 Re-read the file and send its real text.",
                crate::tools::TRUNCATION_MARKER
            ),
        ));
    }
    Ok(())
}

/// Create a file, or replace one entirely.
///
/// The result states **which of the two it did**, plus the byte count — so "I
/// created a file I meant to edit" is visible rather than silent.
pub(crate) fn write_file(
    ctx: &ToolCtx,
    path: &str,
    content: &str,
) -> Result<ToolOutcome, ToolError> {
    // Argument defects first, before any filesystem work: reporting a path
    // error here would send the model off fixing the path and re-sending the
    // same doomed body.
    refuse_truncated_echo(content)?;

    // Held across resolve → hash → write (WD-15).
    let _guard = write_lock();

    let target = resolve_for_write(&ctx.root, path)?;
    let is_manifest = refuse_composition_write(ctx, &target.path)?;
    refuse_unparseable_manifest(is_manifest, content)?;

    let is_dir =
        target.existed && target.path.metadata().map(|m| m.is_dir()).unwrap_or(false);
    crate::tools::gate_openable(&target.path, path, is_dir)?;

    // WD-4: only now. Directories are created after the path has survived every
    // check, so a refused write never leaves a tree behind.
    //
    // One `if let` rather than a nested pair — `clippy::collapsible_if` fires
    // on the nested form under `-D warnings`.
    if let Some(parent) = target
        .path
        .parent()
        .filter(|_| !target.dirs_to_create.is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| ToolError::new("write_failed", e.to_string()))?;
    }

    // Read BEFORE writing. Hashing afterwards would make `before` equal `after`
    // on every modify, which looks right and carries nothing.
    let before = if target.existed {
        std::fs::read(&target.path)
            .ok()
            .map(|b| crate::hash::sha256_hex(&b))
    } else {
        None
    };

    std::fs::write(&target.path, content)
        .map_err(|e| ToolError::new("write_failed", e.to_string()))?;

    let op = if target.existed { "modified" } else { "created" };
    let rel = workspace_relative(&ctx.root, &target.path, path);
    let mut text = format!("{op} {rel} ({} bytes)", content.len());
    if !target.dirs_to_create.is_empty() {
        text.push_str(&format!(
            "; {} directories created: {}",
            target.dirs_to_create.len(),
            target.dirs_to_create.join(", ")
        ));
    }
    text.push('\n');

    Ok(ToolOutcome {
        text,
        effects: vec![Effect::FileChanged {
            path: rel,
            op,
            before,
            after: crate::hash::sha256_hex(content.as_bytes()),
        }],
    })
}

/// **WD-16.** Does `old_string` look like `read_file` output rather than file
/// content?
///
/// ycc's `editdiag.go` opens by naming its evidence: transcript analysis of
/// real sessions shows the dominant Edit failure is "old_string not found",
/// and their number-one diagnosed cause is the model pasting `old_string` with
/// `read_file`'s line-number prefixes still attached. Our `read_page` renders
/// `format!("{seen:>6}\t{}", line.text)` — byte-for-byte the format that regex
/// was written against, so we have the exact hazard.
///
/// Their full diagnostic, with a whitespace-normalized scorer and a capped
/// snippet echo, is 138 lines. This is the cheap check; it can grow if it
/// proves insufficient. The prefix chars are ASCII digits, so `digits` is a
/// byte count and the slice below is safe.
fn looks_like_read_output(old_string: &str) -> bool {
    old_string.lines().any(|line| {
        let trimmed = line.trim_start_matches(' ');
        let digits = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();
        digits > 0 && trimmed[digits..].starts_with('\t')
    })
}

/// Replace one unique occurrence of `old_string` with `new_string`.
///
/// **Exact matching only** (WD-6). No fuzzy matching, no regex, no line
/// numbers, no patch format. Every harness that added a fuzzy chain then needed
/// a guard against its own matcher — disproportionate spans, reindented
/// replacements, escape drift, unicode mangling — and each of those corruption
/// classes was *introduced* by the fuzzy layer. With exact matching they all
/// degrade to `no_match`: a wasted round trip, not a damaged file. The right
/// trade for a prose workspace under git with no in-session diff.
///
/// An edit never creates: it goes through `guard::resolve` alone, so a missing
/// file is `not_found` rather than falling through to the create path.
///
/// There is deliberately no `replace_all`. The consequence is that
/// `ambiguous_match` must not say only "include more surrounding text", because
/// for a rename the honest answer is "call me N times".
pub(crate) fn edit_file(
    ctx: &ToolCtx,
    path: &str,
    old_string: &str,
    new_string: &str,
) -> Result<ToolOutcome, ToolError> {
    if old_string.is_empty() {
        return Err(ToolError::new(
            "invalid_input",
            "old_string is empty — use write_file to create or fully replace a file",
        ));
    }
    if old_string == new_string {
        return Err(ToolError::new(
            "invalid_input",
            "new_string is identical to old_string — this edit would change nothing",
        ));
    }
    // WD-14 extended to this verb: inserting read_file's display text through
    // an edit is the same corruption by the same route as writing it whole.
    refuse_truncated_echo(new_string)?;

    let _guard = write_lock();

    let resolved = crate::guard::resolve(&ctx.root, path)?;
    let is_manifest = refuse_composition_write(ctx, &resolved)?;

    let meta = resolved
        .metadata()
        .map_err(|_| ToolError::new("not_found", format!("nothing at {path}")))?;
    crate::tools::gate_openable(&resolved, path, meta.is_dir())?;
    // Checked from metadata, so an oversized file is never loaded in order to
    // be rejected — the same guard `read_whole` states, on a path that reaches
    // bundles and lockfiles.
    if meta.len() > crate::tools::MAX_BYTES {
        return Err(ToolError::new("too_large", format!("{} bytes", meta.len())));
    }

    let raw =
        std::fs::read(&resolved).map_err(|e| ToolError::new("read_failed", e.to_string()))?;
    // Binary gets a refusal rather than a guess, and the SAME refusal
    // `read_page` gives — the same condition answering with two different
    // statuses is the drift the status table exists to prevent.
    let original = String::from_utf8(raw)
        .map_err(|_| ToolError::new("not_text", format!("{path} is not valid UTF-8")))?;

    // Count first, mutate second. Deciding and acting in one pass is how a
    // check ends up announcing a refusal it did not actually perform.
    let occurrences = original.matches(old_string).count();
    match occurrences {
        0 => {
            let hint = if looks_like_read_output(old_string) {
                " — old_string carries read_file's line-number prefixes (the \
                 \"   12\\t\" at the start of each line is display formatting, not \
                 file content); send the file's own text without them"
            } else {
                " — re-read the file and copy the text exactly, whitespace included"
            };
            return Err(ToolError::new(
                "no_match",
                format!("old_string does not appear in {path}{hint}"),
            ));
        }
        1 => {}
        n => {
            return Err(ToolError::new(
                "ambiguous_match",
                format!(
                    "old_string appears {n} times in {path} — include more surrounding \
                     text so exactly one match remains, or call edit_file once per \
                     occurrence (there is no replace-all)"
                ),
            ))
        }
    }

    let updated = original.replace(old_string, new_string);
    refuse_unparseable_manifest(is_manifest, &updated)?;

    std::fs::write(&resolved, &updated)
        .map_err(|e| ToolError::new("write_failed", e.to_string()))?;

    let rel = workspace_relative(&ctx.root, &resolved, path);
    Ok(ToolOutcome {
        text: format!("modified {rel} (1 replacement, {} bytes)\n", updated.len()),
        effects: vec![Effect::FileChanged {
            path: rel,
            op: "modified",
            before: Some(crate::hash::sha256_hex(original.as_bytes())),
            after: crate::hash::sha256_hex(updated.as_bytes()),
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A workspace shaped like the real one: a public manifest, a `notes/`
    /// directory with something already in it.
    fn ws() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("modules.toml"),
            "[router]\ncontains = [\"CLAUDE.md\"]\n",
        )
        .unwrap();
        fs::create_dir_all(dir.path().join("notes")).unwrap();
        fs::write(dir.path().join("notes/existing.md"), "original\n").unwrap();
        dir
    }

    fn ctx(dir: &tempfile::TempDir, may_write_composition: bool) -> ToolCtx {
        ToolCtx {
            root: dir.path().to_path_buf(),
            may_write_composition,
            turn: crate::tools::TurnIdentity::default(),
            instance_events: None,
        }
    }

    #[test]
    fn write_file_creates_a_new_file_and_says_it_created() {
        let dir = ws();
        let out = write_file(&ctx(&dir, false), "notes/fresh.md", "hello\n").unwrap();

        assert_eq!(
            fs::read_to_string(dir.path().join("notes/fresh.md")).unwrap(),
            "hello\n"
        );
        assert!(out.text.contains("created"), "{}", out.text);
        assert!(out.text.contains('6'), "the byte count: {}", out.text);
        assert!(!out.text.trim().is_empty(), "an empty text block is a 400");
    }

    #[test]
    fn write_file_replaces_an_existing_file_and_says_it_modified() {
        // "I created a file I meant to edit" must be visible rather than
        // silent, which is why the result states which of the two it did.
        let dir = ws();
        let out = write_file(&ctx(&dir, false), "notes/existing.md", "replaced\n").unwrap();

        assert_eq!(
            fs::read_to_string(dir.path().join("notes/existing.md")).unwrap(),
            "replaced\n"
        );
        assert!(out.text.contains("modified"), "{}", out.text);
    }

    #[test]
    fn write_file_records_the_effect_with_the_prior_content_hash() {
        // MUTATION-CHECKED. Hashing after the write yields before == after on
        // every modify — plausible-looking and silently useless, because the
        // one thing the record exists to carry is what the content WAS.
        let dir = ws();
        let out = write_file(&ctx(&dir, false), "notes/existing.md", "replaced\n").unwrap();

        assert_eq!(out.effects.len(), 1);
        let Effect::FileChanged { path, op, before, after } = &out.effects[0] else {
            panic!("a write records a FileChanged: {:?}", out.effects);
        };
        assert_eq!(path, "notes/existing.md");
        assert_eq!(*op, "modified");
        assert_eq!(
            before.as_deref(),
            Some(crate::hash::sha256_hex(b"original\n").as_str())
        );
        assert_eq!(after, &crate::hash::sha256_hex(b"replaced\n"));
    }

    #[test]
    fn a_created_file_has_no_before_hash() {
        let dir = ws();
        let out = write_file(&ctx(&dir, false), "notes/fresh.md", "hi\n").unwrap();

        let Effect::FileChanged { op, before, .. } = &out.effects[0] else {
            panic!("a write records a FileChanged: {:?}", out.effects);
        };
        assert_eq!(*op, "created");
        assert!(before.is_none(), "there was no prior content to hash");
    }

    #[test]
    fn a_no_op_write_is_recorded_rather_than_suppressed() {
        // D-1. "The model wrote and nothing moved" is precisely the
        // wasted-round signal the tool-call-usefulness work wants log-derived
        // rather than self-reported, so it must reach the log.
        let dir = ws();
        let out = write_file(&ctx(&dir, false), "notes/existing.md", "original\n").unwrap();

        let Effect::FileChanged { before, after, .. } = &out.effects[0] else {
            panic!("a write records a FileChanged: {:?}", out.effects);
        };
        assert_eq!(before.as_deref(), Some(after.as_str()));
    }

    #[test]
    fn missing_parents_are_created_and_every_one_is_named() {
        // WD-4, reversed. The original refusal prescribed a recovery the
        // session cannot perform — there is no `mkdir` verb and no bash — so it
        // left the model to give up or silently flatten the path it meant.
        // Creating them and REPORTING them keeps the whole of the original
        // reasoning (a conjured tree must not be invisible) and removes the
        // dead end.
        let dir = ws();
        let out = write_file(&ctx(&dir, false), "notes/2026/08/entry.md", "x\n").unwrap();

        assert!(dir.path().join("notes/2026/08/entry.md").exists());
        assert!(out.text.contains("2 directories created"), "{}", out.text);
        assert!(out.text.contains("notes/2026"), "{}", out.text);
        assert!(out.text.contains("notes/2026/08"), "{}", out.text);
    }

    #[test]
    fn a_refused_write_leaves_no_directory_tree_behind() {
        // WD-4: directories are created ONLY after the path survives every
        // check. A refusal that has already conjured three directories is the
        // invisible failure the reversal was argued against.
        let dir = ws();
        assert!(write_file(&ctx(&dir, false), "a/b/c/icon.png", "x").is_err());
        assert!(!dir.path().join("a").exists(), "a refused write built a tree");
    }

    #[test]
    fn a_credential_cannot_be_written() {
        // `secrets.json` is the discriminating case: a credential the EXTENSION
        // gate lets through, where the guard's judge is the only thing standing
        // between it and being written. Asserting the status, not merely
        // `is_err`, is what keeps that true.
        let dir = ws();
        let err = write_file(&ctx(&dir, false), "notes/secrets.json", "{\"k\":1}").unwrap_err();
        assert_eq!(err.status, "denied_credential");

        for p in ["notes/.env", ".env", "notes/Secrets.xcconfig"] {
            assert!(write_file(&ctx(&dir, false), p, "TOKEN=x").is_err(), "{p}");
        }
    }

    #[test]
    fn a_write_into_the_engines_own_data_dir_is_refused() {
        // I-1's append-only guarantee lives in `LogStore::append`; a write tool
        // reaching `.armillary/*.jsonl` as an ordinary file goes around it, and
        // correction-is-a-new-event stops meaning anything if the record can be
        // edited from outside the writer. No new rule was needed — `is_noise`
        // already denies it — but the absence of a new rule is exactly what
        // this pins.
        let dir = ws();
        fs::create_dir_all(dir.path().join(".armillary/streams")).unwrap();
        let err = write_file(&ctx(&dir, false), ".armillary/streams/s.jsonl", "{}").unwrap_err();
        assert_eq!(err.status, "denied_noise");
    }

    #[test]
    fn an_unservable_extension_cannot_be_written() {
        // WD-5. `write_file` takes a string, so writing a .png is meaningless;
        // reusing the allowlist means the set you can write is the set you can
        // read, with no second rule to keep in sync.
        let dir = ws();
        let err = write_file(&ctx(&dir, false), "notes/icon.png", "not a png").unwrap_err();
        assert_eq!(err.status, "not_openable");
    }

    #[test]
    fn writing_to_a_directory_says_it_is_a_directory() {
        // The misleading-reason failure `tools.rs` documents having fixed once
        // for reads: `not_openable` here would name the wrong cause.
        let dir = ws();
        let err = write_file(&ctx(&dir, false), "notes", "x").unwrap_err();
        assert_eq!(err.status, "is_a_directory");
    }

    #[test]
    fn a_write_through_a_dangling_symlink_never_lands_outside_the_workspace() {
        // The end-to-end form of the confirmed escape. The guard test proves
        // the resolution refuses; this proves nothing was planted.
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("planted.md");
        let dir = ws();
        std::os::unix::fs::symlink(&target, dir.path().join("notes/dangling.md")).unwrap();

        let err = write_file(&ctx(&dir, false), "notes/dangling.md", "owned\n").unwrap_err();

        assert_eq!(err.status, "outside_workspace");
        assert!(!target.exists(), "a write escaped the workspace");
    }

    #[test]
    fn write_file_refuses_content_carrying_the_truncation_marker() {
        // MUTATION-CHECKED. WD-14. That is `read_file`'s display text being
        // echoed back as a file — a read→write round trip that makes the cut
        // permanent and records it as a clean `modified`. The reader is not
        // faulty; what changed is that a writer exists.
        let dir = ws();
        let echoed = format!("some prose {} \nmore\n", crate::tools::TRUNCATION_MARKER);

        let err = write_file(&ctx(&dir, false), "notes/echo.md", &echoed).unwrap_err();

        assert_eq!(err.status, "invalid_input");
        assert!(err.detail.contains("read_file"), "{}", err.detail);
        assert!(!dir.path().join("notes/echo.md").exists());
    }

    #[test]
    fn the_manifest_is_locked_unless_the_instance_was_granted_it() {
        // MUTATION-CHECKED, and BOTH halves are the test. Asserting only the
        // refusal would pass identically if `write_file` were broken for every
        // path — declaring the grant and watching the SAME write succeed is
        // what proves the refusal is a consequence of the flag.
        let dir = ws();
        let body = "[router]\ncontains = [\"CLAUDE.md\"]\n# touched\n";

        let err = write_file(&ctx(&dir, false), "modules.toml", body).unwrap_err();
        assert_eq!(err.status, "composition_locked");
        assert!(!fs::read_to_string(dir.path().join("modules.toml"))
            .unwrap()
            .contains("touched"));

        write_file(&ctx(&dir, true), "modules.toml", body).unwrap();
        assert!(fs::read_to_string(dir.path().join("modules.toml"))
            .unwrap()
            .contains("touched"));
    }

    #[test]
    fn the_manifest_is_locked_through_every_spelling_that_resolves_to_it() {
        // MUTATION-CHECKED. In the real workspace `modules.local.toml` is a
        // symlink into the commons, so the same file is reachable by two names.
        // Comparing the RAW request path would gate one spelling and wave the
        // other through.
        //
        // The link's TARGET is deliberately named something else. The lexical
        // half of the lock matches on the resolved path's final component, so a
        // link whose target happens to share its name would be caught by the
        // lexical check alone and would prove nothing about the canonical one.
        // Neither side is hardcoded to a studio path — the commons' name is not
        // the engine's business (G-1).
        let dir = ws();
        fs::create_dir_all(dir.path().join("commons/setup")).unwrap();
        fs::write(dir.path().join("commons/setup/machine.toml"), "# real\n").unwrap();
        std::os::unix::fs::symlink(
            dir.path().join("commons/setup/machine.toml"),
            dir.path().join("modules.local.toml"),
        )
        .unwrap();

        for spelling in ["modules.local.toml", "commons/setup/machine.toml"] {
            let err = write_file(&ctx(&dir, false), spelling, "# nope\n").unwrap_err();
            assert_eq!(err.status, "composition_locked", "{spelling}");
        }
    }

    #[test]
    fn creating_a_manifest_that_does_not_exist_yet_is_still_locked() {
        // MUTATION-CHECKED — and this is the case WD-11's lexical half actually
        // closes. The approved check reasoned that "a manifest that does not
        // exist canonicalizes to nothing and simply does not match —
        // presence-gated like everything else", which fails OPEN on exactly the
        // workspace that has no `modules.local.toml` yet: an ungranted instance
        // CREATES one, and `parse_workspace` runs per request, so it composes
        // the workspace on the next tool call in the same turn.
        //
        // (The design's own row for WD-11 used a DANGLING manifest symlink.
        // With the guard's symlink_metadata step in place that path is refused
        // as `outside_workspace` before the lock is consulted, so it can no
        // longer discriminate. See the test below.)
        let dir = ws();
        assert!(!dir.path().join("modules.local.toml").exists());

        let err = write_file(&ctx(&dir, false), "modules.local.toml", "# mine\n").unwrap_err();

        assert_eq!(err.status, "composition_locked");
        assert!(!dir.path().join("modules.local.toml").exists());

        // The other half: granted, the same create succeeds.
        write_file(&ctx(&dir, true), "modules.local.toml", "# mine\n").unwrap();
        assert!(dir.path().join("modules.local.toml").exists());
    }

    #[test]
    fn a_dangling_manifest_symlink_is_refused_rather_than_followed() {
        // WD-11's original case, asserted with the status it ACTUALLY returns.
        // A dangling `modules.local.toml` is what a machine has before the
        // commons is cloned; the old check skipped its `if let Ok(canon)` body
        // entirely and the write then landed on the symlink's target. It is now
        // refused one step earlier, by the guard — which is a stronger refusal,
        // not a weaker one, because it also protects the target.
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("modules.local.toml");
        let dir = ws();
        std::os::unix::fs::symlink(&target, dir.path().join("modules.local.toml")).unwrap();

        let err = write_file(&ctx(&dir, false), "modules.local.toml", "# owned\n").unwrap_err();

        assert_eq!(err.status, "outside_workspace");
        assert!(!target.exists(), "a manifest write escaped the workspace");
    }

    #[test]
    fn a_granted_manifest_write_must_parse_as_toml_first() {
        // WD-12. `parse_workspace` runs PER REQUEST, so one malformed TOML
        // write degrades `get_composition` and every default-domain
        // `search`/`find_files` for the rest of the turn — immediately, with no
        // rebuild, and with the model given only a `composition_unreadable` to
        // diagnose.
        let dir = ws();
        let err = write_file(&ctx(&dir, true), "modules.toml", "[router\nbroken = \n").unwrap_err();

        assert_eq!(err.status, "invalid_input");
        assert!(
            fs::read_to_string(dir.path().join("modules.toml"))
                .unwrap()
                .contains("CLAUDE.md"),
            "the old manifest must survive a refused write"
        );

        // Scoped to the manifests only: an ordinary `.toml` elsewhere is prose
        // as far as this engine is concerned.
        write_file(&ctx(&dir, false), "notes/broken.toml", "[not\nvalid = \n").unwrap();
    }

    // ---- edit_file ----

    #[test]
    fn edit_file_replaces_a_unique_match() {
        let dir = ws();
        fs::write(dir.path().join("notes/e.md"), "alpha\nbeta\ngamma\n").unwrap();

        let out = edit_file(&ctx(&dir, false), "notes/e.md", "beta", "BETA").unwrap();

        assert_eq!(
            fs::read_to_string(dir.path().join("notes/e.md")).unwrap(),
            "alpha\nBETA\ngamma\n"
        );
        let Effect::FileChanged { op, before, after, .. } = &out.effects[0] else {
            panic!("an edit records a FileChanged: {:?}", out.effects);
        };
        assert_eq!(*op, "modified", "an edit never creates");
        assert_eq!(
            before.as_deref(),
            Some(crate::hash::sha256_hex(b"alpha\nbeta\ngamma\n").as_str())
        );
        assert_eq!(after, &crate::hash::sha256_hex(b"alpha\nBETA\ngamma\n"));
    }

    #[test]
    fn edit_file_refuses_an_ambiguous_match_and_changes_nothing() {
        // MUTATION-CHECKED, and the DISK assertion is the half that matters. A
        // check that announces ambiguity while still performing the first
        // replacement passes a text-only test and silently corrupts a file —
        // the "cap announced itself but did not bind" defect, ported.
        let dir = ws();
        let body = "same\nsame\n";
        fs::write(dir.path().join("notes/e.md"), body).unwrap();

        let err = edit_file(&ctx(&dir, false), "notes/e.md", "same", "changed").unwrap_err();

        assert_eq!(err.status, "ambiguous_match");
        assert!(err.detail.contains('2'), "the count must be named: {}", err.detail);
        assert_eq!(
            fs::read_to_string(dir.path().join("notes/e.md")).unwrap(),
            body,
            "a refused edit must leave the file byte-identical"
        );
    }

    #[test]
    fn a_missing_match_diagnoses_the_line_number_prefix() {
        // WD-16. ycc's evidence: transcript analysis shows the dominant Edit
        // failure is "old_string not found", and their number-one diagnosed
        // cause is the model pasting `old_string` with read_file's line-number
        // prefixes still attached. Our `read_page` renders exactly the format
        // that regex was written against, so we have the exact hazard.
        let dir = ws();
        let pasted = "     1\toriginal";

        let err = edit_file(&ctx(&dir, false), "notes/existing.md", pasted, "x").unwrap_err();

        assert_eq!(err.status, "no_match");
        assert!(
            err.detail.contains("line-number"),
            "the dominant cause must be named: {}",
            err.detail
        );

        // The other half: an ordinary miss must NOT get the hint, or the hint
        // is noise on every failure and diagnoses nothing.
        let plain = edit_file(&ctx(&dir, false), "notes/existing.md", "absent", "x").unwrap_err();
        assert_eq!(plain.status, "no_match");
        assert!(!plain.detail.contains("line-number"), "{}", plain.detail);
    }

    #[test]
    fn edit_file_refuses_an_empty_old_string_and_points_at_write_file() {
        let dir = ws();
        let err = edit_file(&ctx(&dir, false), "notes/existing.md", "", "x").unwrap_err();

        assert_eq!(err.status, "invalid_input");
        assert!(err.detail.contains("write_file"), "{}", err.detail);
    }

    #[test]
    fn edit_file_refuses_a_replacement_that_changes_nothing() {
        // A mistake, not a no-op worth performing.
        let dir = ws();
        let err =
            edit_file(&ctx(&dir, false), "notes/existing.md", "original", "original").unwrap_err();
        assert_eq!(err.status, "invalid_input");
    }

    #[test]
    fn edit_file_refuses_a_file_that_does_not_exist() {
        // An edit never creates — WD-2's fallback is `write_file`'s alone.
        let dir = ws();
        let err = edit_file(&ctx(&dir, false), "notes/absent.md", "a", "b").unwrap_err();
        assert_eq!(err.status, "not_found");
    }

    #[test]
    fn edit_file_passes_the_same_gate_write_file_does() {
        // WD-5, restored. The superseded plan violated this invariant in its
        // second verb: `edit_file` checked neither `is_openable` nor
        // is-a-directory, so it could modify files `read_file` refuses to open
        // — `.gitattributes`, `.csv`, `.xml`, `.lock`, anything outside the
        // twenty-extension allowlist.
        let dir = ws();
        fs::write(dir.path().join("notes/data.csv"), "a,b\n").unwrap();

        let err = edit_file(&ctx(&dir, false), "notes/data.csv", "a", "z").unwrap_err();
        assert_eq!(err.status, "not_openable");

        let dir_err = edit_file(&ctx(&dir, false), "notes", "a", "z").unwrap_err();
        assert_eq!(dir_err.status, "is_a_directory");
    }

    #[test]
    fn edit_file_answers_not_text_on_a_non_utf8_file() {
        // The same condition returning two different statuses — one of them a
        // 500 for a legible user error — is the drift the status table exists
        // to prevent. `read_page` says `not_text`; so does this.
        let dir = ws();
        fs::write(dir.path().join("notes/bad.md"), [0xff, 0xfe, 0x00]).unwrap();

        let err = edit_file(&ctx(&dir, false), "notes/bad.md", "a", "b").unwrap_err();
        assert_eq!(err.status, "not_text");
    }

    #[test]
    fn edit_file_respects_the_composition_lock() {
        let dir = ws();
        let err = edit_file(&ctx(&dir, false), "modules.toml", "[router]", "[ROUTER]").unwrap_err();
        assert_eq!(err.status, "composition_locked");
    }

    #[test]
    fn a_granted_manifest_edit_must_still_parse() {
        // WD-12 applies to the RESULT, not to the argument — an edit can break
        // a manifest just as thoroughly as a rewrite.
        let dir = ws();
        let err = edit_file(&ctx(&dir, true), "modules.toml", "[router]", "[router").unwrap_err();

        assert_eq!(err.status, "invalid_input");
        assert!(
            fs::read_to_string(dir.path().join("modules.toml"))
                .unwrap()
                .contains("[router]"),
            "the old manifest must survive a refused edit"
        );
    }

    #[test]
    fn edit_file_refuses_a_new_string_carrying_the_truncation_marker() {
        // WD-14, extended to this verb: inserting `…[line truncated]` through
        // an edit is the same corruption by the same route.
        let dir = ws();
        let poisoned = format!("original {}", crate::tools::TRUNCATION_MARKER);

        let err =
            edit_file(&ctx(&dir, false), "notes/existing.md", "original", &poisoned).unwrap_err();

        assert_eq!(err.status, "invalid_input");
        assert_eq!(
            fs::read_to_string(dir.path().join("notes/existing.md")).unwrap(),
            "original\n"
        );
    }

    #[test]
    fn a_multibyte_edit_does_not_corrupt_the_file() {
        // `.爻` is three bytes per character and this workspace writes node
        // files in it. Rust's `str::replace` is character-safe, so this is a
        // regression pin rather than a fix — it exists so a future byte-indexed
        // "optimization" cannot land quietly.
        let dir = ws();
        fs::write(dir.path().join("notes/n.爻"), "爻爻 keep 爻爻\n").unwrap();

        edit_file(&ctx(&dir, false), "notes/n.爻", "keep", "kept").unwrap();

        let after = fs::read_to_string(dir.path().join("notes/n.爻")).unwrap();
        assert_eq!(after, "爻爻 kept 爻爻\n");
        assert!(!after.contains('\u{FFFD}'));
    }

    #[test]
    fn the_recorded_path_is_the_resolved_one_not_the_request_spelling() {
        // WD-13. Two writes to one inode through two spellings would otherwise
        // produce two events a consumer cannot join, and the before/after hash
        // chain silently breaks across them. Workspace-relative, never absolute
        // (G-1).
        let dir = ws();
        fs::create_dir_all(dir.path().join("commons")).unwrap();
        fs::write(dir.path().join("commons/board.md"), "# board\n").unwrap();
        std::os::unix::fs::symlink(
            dir.path().join("commons/board.md"),
            dir.path().join("BOARD.md"),
        )
        .unwrap();

        let out = write_file(&ctx(&dir, false), "BOARD.md", "# board, edited\n").unwrap();

        let Effect::FileChanged { path, .. } = &out.effects[0] else {
            panic!("a write records a FileChanged: {:?}", out.effects);
        };
        assert_eq!(path, "commons/board.md", "the request spelling was recorded");
        assert!(!path.starts_with('/'), "never absolute: {path}");
    }
}
