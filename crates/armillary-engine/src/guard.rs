//! Path safety — the engine's only security-critical surface.
//!
//! # The ordering that matters
//!
//! The first version of this module validated **the string the client sent**
//! and then opened **whatever that resolved to**. On a case-insensitive
//! filesystem, in a workspace that deliberately routes real content through
//! symlinks, "a different thing" is the normal case rather than the edge case.
//! That single inversion produced two independent one-request bypasses,
//! closed by this ordering: `.ENV` reached `.env`, and a symlink named
//! `config.local` reached `.env`.
//!
//! So: cheap rejections on the request first (absolute paths, `..`), then
//! resolve, then **judge the canonical path** — because the canonical path is
//! the thing that will actually be opened.
//!
//! What this ordering does **not** close: a hard link. `canonicalize`
//! resolves symlinks by following their target, but a hard link has no
//! target to resolve — it is a second directory entry pointing at the same
//! inode, indistinguishable from an ordinary file once opened. A hard link
//! named `notes.md` pointing at `.env` still canonicalizes to a path judged
//! on the name `notes.md`, and serves `.env`'s bytes. The extension allowlist
//! added since (`is_openable`) narrows this — a link named `config.local`,
//! with no allowlisted extension, no longer opens — but a link named with an
//! allowlisted extension still serves whatever it points at. Closing that
//! would mean comparing inode numbers against known-sensitive files, or
//! refusing to open any file with a link count above one; neither is done
//! here.
//!
//! Matching is case-insensitive everywhere. That is a property of the
//! filesystem to be assumed, not detected: a service that is safe only on a
//! case-sensitive volume is not safe.

use axum::http::StatusCode;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, PartialEq)]
pub enum GuardError {
    /// Not usable as a path at all.
    Malformed,
    /// Resolves outside the root, or tries to.
    Escaped,
    /// Legal, but nothing is there.
    NotFound,
    /// Present, and never served: credentials and secret material.
    DeniedCredential,
    /// Present, and not served because it is noise rather than content.
    DeniedNoise,
}

impl GuardError {
    pub fn status(&self) -> StatusCode {
        match self {
            GuardError::Malformed => StatusCode::BAD_REQUEST,
            GuardError::NotFound => StatusCode::NOT_FOUND,
            GuardError::Escaped | GuardError::DeniedCredential | GuardError::DeniedNoise => {
                StatusCode::FORBIDDEN
            }
        }
    }

    /// Stable machine-readable code for the response body. Never the `Debug`
    /// rendering of this enum — a client would start matching on it, and then
    /// the variant names become public API.
    pub fn code(&self) -> &'static str {
        match self {
            GuardError::Malformed => "malformed_path",
            GuardError::Escaped => "outside_workspace",
            GuardError::NotFound => "not_found",
            GuardError::DeniedCredential => "denied_credential",
            GuardError::DeniedNoise => "denied_noise",
        }
    }
}

/// Names that are never served and never listed, at any depth.
///
/// The engine serves the **disk**, not git, so `.gitignore` filters nothing.
///
/// This list was originally one entry — `.env` — arrived at by looking at the
/// workspace and finding one. `Secrets.xcconfig`, holding a live API key, was
/// sitting in the same tree the whole time. A denylist built by inspection is
/// exactly as complete as the last inspection, which is the standing argument
/// for the allowlist posture noted at the bottom of this file.
fn is_credential(name_lower: &str) -> bool {
    // Committed templates exist to be read, and are the reason `.env` cannot
    // simply be a prefix rule.
    const TEMPLATE_SUFFIXES: [&str; 4] = [".example", ".sample", ".template", ".dist"];
    if name_lower.starts_with(".env") && TEMPLATE_SUFFIXES.iter().any(|s| name_lower.ends_with(s)) {
        return false;
    }

    name_lower.starts_with(".env")
        // `.git` is a security boundary wearing a usability costume: history
        // holds every secret ever committed and later removed, and .git/config
        // can carry a tokenised remote.
        || matches!(
            name_lower,
            ".git" | ".ssh" | ".gnupg" | ".aws" | ".netrc" | ".npmrc" | ".pypirc"
                | "id_rsa" | "id_dsa" | "id_ecdsa" | "id_ed25519" | ".htpasswd"
        )
        // Catches Secrets.xcconfig, secrets.json, .secrets, credentials.*
        || name_lower.contains("secret")
        || name_lower.contains("credential")
        || [".pem", ".key", ".p8", ".p12", ".pfx", ".keystore", ".jks", ".mobileprovision"]
            .iter()
            .any(|ext| name_lower.ends_with(ext))
}

/// Build output and dependency trees: not secret, but not content either.
///
/// Refused rather than merely unlisted. Hiding `node_modules` from its parent
/// while `/tree?path=node_modules` still enumerates it is a rule that only
/// looks like one — and enumerating tens of thousands of entries is the exact
/// hazard the hiding was for.
fn is_noise(name_lower: &str) -> bool {
    matches!(
        name_lower,
        "node_modules"
            | "target"
            | "build"
            // Swift Package Manager's is `.build`, with the dot — and it is
            // where this workspace's thousand-entry directories actually live
            // (index-store records under daoUI, KairosCore, Mercurial). Listing
            // `build` and not `.build` is the same failure as a credential
            // denylist assembled by remembering names.
            | ".build"
            // A git worktree is a second checkout of a repo that is already
            // composed here — derived duplicate state, the same category as
            // `target/` and `.build`. Seven of them exist in this workspace
            // and each holds a full copy of its repo, so a walk that entered
            // them would return the same file from two branches with nothing
            // in the result to say which was which. Every byte in a worktree
            // is readable at its real path.
            | ".worktrees"
            | "dist"
            | ".next"
            | ".expo"
            | "deriveddata"
            | ".venv"
            | ".gradle"
            | "pods"
            // The engine's own guarded data dir (session logs, `sessions.rs`
            // + `log/store.rs`). Not secret in the credential sense, but it
            // must never be readable through the Explorer either: every
            // session's full event log sits under here, and this service
            // serves the disk — without this entry, `/tree` and `/file`
            // would happily hand it out. Applied here, not just at the
            // default `--data-dir` value, so a caller cannot regain access by
            // repointing the flag inside the served root.
            | ".armillary"
    )
}

/// Extensions whose contents may be served.
///
/// The inverse of `is_credential`, and the reason it exists: a denylist is
/// exactly as complete as the last inspection of the tree, and the first one
/// missed a live API key sitting in `Secrets.xcconfig`. The set of things this
/// service *should* serve is small and enumerable. The set it must not serve is
/// neither. So: enumerate the small one, and let forgetting mean "a file does
/// not open" rather than "a secret is served".
const OPENABLE_EXTENSIONS: [&str; 20] = [
    // prose
    "md", "爻", "txt",
    // config
    "toml", "json", "yaml", "yml", "ini",
    // source
    "rs", "ts", "tsx", "js", "jsx", "swift", "py", "sh", "sql", "css", "html", "mjs",
];

/// Files whose extension carries no information, allowlisted by exact name.
/// Lowercased before comparison, like everything else in this module.
const OPENABLE_NAMES: [&str; 7] = [
    "license",
    "readme",
    "makefile",
    "dockerfile",
    ".gitignore",
    ".editorconfig",
    "cargo.lock",
];

/// True when a file of this name may have its contents served.
///
/// Governs **opening only**. Listings are unaffected: a `.png` still appears in
/// its directory and simply refuses to open, because hiding what cannot be
/// opened would reintroduce, one level down, exactly the projection this whole
/// change removes.
pub fn is_openable(name: &str) -> bool {
    let lower = name.to_lowercase();
    if OPENABLE_NAMES.contains(&lower.as_str()) {
        return true;
    }
    // `rsplit_once` on a dotfile like `.gitignore` yields an EMPTY stem, which
    // is why the name list is consulted first and why the stem is checked here:
    // otherwise `.md` (a file literally named that) would open on the strength
    // of being a dotfile whose "extension" happens to be allowlisted.
    match lower.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => OPENABLE_EXTENSIONS.contains(&ext),
        _ => false,
    }
}

/// Judge one path component. Used both for listings and for resolution, so a
/// route cannot consult one rule and forget the other.
fn judge(name: &str) -> Option<GuardError> {
    let lower = name.to_lowercase();
    if is_credential(&lower) {
        Some(GuardError::DeniedCredential)
    } else if is_noise(&lower) {
        Some(GuardError::DeniedNoise)
    } else {
        None
    }
}

/// True for entries a directory listing must omit.
pub fn is_hidden_from_listings(name: &str) -> bool {
    judge(name).is_some()
}

/// Resolve a user-supplied path against the workspace root.
///
/// Order is load-bearing:
///   1. reject absolute paths **before** joining — `Path::join` with an
///      absolute argument discards the root entirely;
///   2. reject `..` so traversal reports `Escaped` rather than masquerading as
///      `NotFound`;
///   3. canonicalize, and re-check the prefix — the only step that catches a
///      symlink pointing out of the tree;
///   4. **judge the canonical path**, so the thing being judged is the thing
///      that will be opened.
pub fn resolve(root: &Path, user_path: &str) -> Result<PathBuf, GuardError> {
    if user_path.contains('\0') {
        return Err(GuardError::Malformed);
    }

    let rel = Path::new(user_path);
    if rel.is_absolute() {
        return Err(GuardError::Escaped);
    }
    for component in rel.components() {
        match component {
            Component::ParentDir => return Err(GuardError::Escaped),
            // Unreachable on Unix after the is_absolute check above; kept as
            // belt-and-braces rather than because it carries weight today.
            Component::Prefix(_) | Component::RootDir => return Err(GuardError::Escaped),
            _ => {}
        }
    }

    let root_canonical = root.canonicalize().map_err(|_| GuardError::NotFound)?;
    let canonical = root_canonical
        .join(rel)
        .canonicalize()
        .map_err(|_| GuardError::NotFound)?;

    let inside = canonical
        .strip_prefix(&root_canonical)
        .map_err(|_| GuardError::Escaped)?;

    // Judge what will actually be opened, not what was asked for. A symlink
    // or a differently-cased spelling both resolve to the same canonical path
    // as their target, so this judges the target's real name either way. A
    // hard link does not: it has no target to resolve to, so this judges the
    // link's own name — see the module doc for what that leaves open.
    for component in inside.components() {
        if let Component::Normal(name) = component {
            if let Some(refusal) = judge(&name.to_string_lossy()) {
                return Err(refusal);
            }
        }
    }

    Ok(canonical)
}

/// Where a file that does not exist yet may be created, and what has to be
/// made first.
#[derive(Debug, PartialEq)]
pub struct CreateTarget {
    /// The absolute path the new file may occupy. Nothing exists there.
    pub path: PathBuf,
    /// Workspace-relative directories that do not exist yet, **outermost
    /// first**. Empty when the parent already exists. WD-4 requires the success
    /// text to name every directory a write created, and this is the only place
    /// that knows which were missing.
    pub dirs_to_create: Vec<String>,
}

/// Resolve where a file that does not exist yet may be created.
///
/// `resolve` canonicalizes the whole path, so it answers `NotFound` for
/// anything absent — right for reads, useless for creates. This is the
/// narrowest possible sibling: it canonicalizes the deepest **existing**
/// ancestor of the parent, judges it, judges every component that would be
/// brought into existence, judges the name being created, and refuses anything
/// already there.
///
/// **Callers must try `resolve` first and fall back here only on `NotFound`**
/// (WD-2). That keeps every existing-file write on the already-audited path and
/// makes this code reachable only when the other has said "nothing there".
///
/// Three steps are load-bearing and each is easy to omit:
///
/// 1. **Canonicalizing the parent and re-checking the prefix.** Without it a
///    parent symlinked out of the tree lets a create write outside the
///    workspace — the exact escape `resolve`'s ordering exists to close.
/// 2. **Judging the final component's own name.** The parent can be perfectly
///    innocent while the file being created is `secrets.json`. Nothing else
///    will ever look at that name, because the path does not exist for `judge`
///    to reach through a canonical form.
/// 3. **Refusing anything `symlink_metadata` can still see.** A DANGLING
///    SYMLINK is what `canonicalize` fails `NotFound` on, which is exactly what
///    routes a path here — and `fs::write` and `create_dir_all` both follow
///    one, straight out of the workspace. Applied to every component this
///    create would bring into existence, not only the last: the same reasoning
///    holds for each, and parent creation means there can now be several.
///
/// ycc handles the dangling case by resolving the longest existing prefix and
/// re-appending the tail, but explicitly disclaims being a security boundary
/// because its agents also have bash. We have no bash. Ours is the boundary.
pub fn resolve_for_create(root: &Path, user_path: &str) -> Result<CreateTarget, GuardError> {
    // Cheap rejections on the request, exactly as `resolve` does them and for
    // the same reason: `Path::join` with an absolute argument discards the
    // base, and `..` must report `Escaped` rather than masquerade as
    // `NotFound`.
    if user_path.contains('\0') {
        return Err(GuardError::Malformed);
    }
    let rel = Path::new(user_path);
    if rel.is_absolute() {
        return Err(GuardError::Escaped);
    }
    for component in rel.components() {
        match component {
            Component::ParentDir => return Err(GuardError::Escaped),
            Component::Prefix(_) | Component::RootDir => return Err(GuardError::Escaped),
            _ => {}
        }
    }

    // The name being created, and the directory it lands in.
    let Some(file_name) = rel.file_name().map(|n| n.to_string_lossy().to_string()) else {
        return Err(GuardError::Malformed);
    };
    let parent_rel = rel.parent().unwrap_or_else(|| Path::new(""));

    let root_canonical = root.canonicalize().map_err(|_| GuardError::NotFound)?;

    // Canonicalize the parent. When it does not exist, walk up to the deepest
    // ancestor that does — everything below it is what a create would have to
    // conjure. `canonicalize` succeeding is the existence test AND the
    // resolution, in one call.
    let mut existing_rel = parent_rel;
    let parent_canonical = loop {
        match root_canonical.join(existing_rel).canonicalize() {
            Ok(p) => break p,
            Err(_) => match existing_rel.parent() {
                Some(up) => existing_rel = up,
                // `parent_rel` was already empty and the root itself did not
                // canonicalize. Nothing can be created under a root that is
                // not there.
                None => return Err(GuardError::NotFound),
            },
        }
    };

    // The only step that catches a parent symlinked out of the tree.
    let inside = parent_canonical
        .strip_prefix(&root_canonical)
        .map_err(|_| GuardError::Escaped)?;

    // Every component of the parent that already exists, judged on its
    // canonical name — the thing that will actually be written into.
    for component in inside.components() {
        if let Component::Normal(name) = component {
            if let Some(refusal) = judge(&name.to_string_lossy()) {
                return Err(refusal);
            }
        }
    }

    // Every component that does NOT exist yet. Counted by depth rather than
    // `strip_prefix` so an empty existing prefix needs no special case.
    let existing_depth = existing_rel
        .components()
        .filter(|c| matches!(c, Component::Normal(_)))
        .count();
    let missing: Vec<String> = parent_rel
        .components()
        .filter_map(|c| match c {
            Component::Normal(n) => Some(n.to_string_lossy().to_string()),
            _ => None,
        })
        .skip(existing_depth)
        .collect();
    for name in &missing {
        // These are about to become real directories. `target/debug/x.md` in a
        // workspace with no `target/` must refuse for the same reason it
        // refuses when `target/` is already there.
        if let Some(refusal) = judge(name) {
            return Err(refusal);
        }
    }

    // The name that does not exist yet.
    if let Some(refusal) = judge(&file_name) {
        return Err(refusal);
    }

    // Nothing `symlink_metadata` can see may be brought into existence here.
    // `canonicalize` already failed for each of these, so anything still
    // visible is a dangling symlink — and both `create_dir_all` and `fs::write`
    // follow one. An ordinary existing file lands here too, which is correct:
    // it means the caller skipped `resolve`.
    let mut walk = parent_canonical.clone();
    let mut rel_so_far = inside.to_path_buf();
    let mut dirs_to_create = Vec::with_capacity(missing.len());
    for name in &missing {
        walk.push(name);
        if std::fs::symlink_metadata(&walk).is_ok() {
            return Err(GuardError::Escaped);
        }
        rel_so_far.push(name);
        dirs_to_create.push(rel_so_far.to_string_lossy().to_string());
    }
    walk.push(&file_name);
    if std::fs::symlink_metadata(&walk).is_ok() {
        return Err(GuardError::Escaped);
    }

    Ok(CreateTarget {
        path: walk,
        dirs_to_create,
    })
}

// The posture noted here previously — an **extension allowlist** (`.md`,
// `.爻`, `.toml`, `.json`, source files) rather than a denylist — has now been
// taken; see `is_openable` above. It governs opening, not listing: the
// denylist above still decides what a directory refuses to enumerate at all,
// and is consulted first, so a credential-shaped name reads as "never served"
// rather than "unknown type" even when its extension would otherwise open.

// TRIPWIRE, CLOSED 2026-08-02 by the write-tools branch. `projection::
// resolve_boot_path` and `loop_::rerecord_boot` read a `boot` event's
// `data.path` via plain `std::fs`, and that read used to carry a bare
// root-containment check, entirely bypassing this module's
// `judge`/`is_credential`/`is_noise` denial — a `boot` event could name
// `.env` or `.git/config` and this module would never be consulted. That was
// safe ONLY because nothing let a client choose or write a boot path.
//
// D-4 makes `modules.local.toml` operator-writable and `write_file`/`edit_file`
// can reach it, so an operator declaring `boot = ["repos/x/Secrets.xcconfig"]`
// became a real disclosure path — not self-exfiltration inside a turn, since it
// needs a new instance to be created, but real. `resolve_boot_path` now calls
// `resolve` above, exactly as this tripwire prescribed; `rerecord_boot` reads
// through the same function and is covered by the same change.
//
// The tripwire that REMAINS: this module is still a pure function of
// `(root, path)`. Any authorization that depends on SESSION state — the
// composition lock (WD-9) is the first — belongs in the caller, not here, or
// every read route silently acquires a write rule.

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Mirrors the real workspace's topology: symlinks resolving inside (which
    /// must pass), a symlink resolving outside (which must not), and the
    /// credential shapes actually present in the served tree.
    fn farm() -> (tempfile::TempDir, tempfile::TempDir) {
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret.md"), "not yours").unwrap();

        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("commons")).unwrap();
        fs::write(root.path().join("commons/board.md"), "# board").unwrap();
        fs::write(root.path().join("environment.md"), "# not a secret").unwrap();

        fs::create_dir_all(root.path().join("repos/app")).unwrap();
        fs::write(root.path().join("repos/app/.env"), "TOKEN=hunter2").unwrap();
        fs::write(root.path().join("repos/app/.env.example"), "TOKEN=").unwrap();
        fs::write(root.path().join("repos/app/Secrets.xcconfig"), "KEY=live").unwrap();

        fs::create_dir(root.path().join(".git")).unwrap();
        fs::write(root.path().join(".git/config"), "url = https://x:tok@h/r").unwrap();
        fs::create_dir(root.path().join("node_modules")).unwrap();

        std::os::unix::fs::symlink(root.path().join("commons"), root.path().join("inside-link"))
            .unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("outside-link")).unwrap();
        // An innocuously named link to a credential — the C2 bypass.
        std::os::unix::fs::symlink(
            root.path().join("repos/app/.env"),
            root.path().join("config.local"),
        )
        .unwrap();

        (root, outside)
    }

    #[test]
    fn resolves_content_inside_the_root() {
        let (root, _o) = farm();
        assert!(resolve(root.path(), "commons/board.md").is_ok());
        assert!(resolve(root.path(), "").is_ok());
        assert!(resolve(root.path(), "inside-link/board.md").is_ok());
    }

    #[test]
    fn refuses_escapes() {
        let (root, _o) = farm();
        for attempt in ["..", "../", "../../etc/passwd", "commons/../../escape"] {
            assert_eq!(resolve(root.path(), attempt), Err(GuardError::Escaped), "{attempt}");
        }
        assert_eq!(resolve(root.path(), "/etc/passwd"), Err(GuardError::Escaped));
        assert_eq!(
            resolve(root.path(), "outside-link/secret.md"),
            Err(GuardError::Escaped)
        );
    }

    #[test]
    fn refuses_embedded_nul() {
        let (root, _o) = farm();
        assert_eq!(resolve(root.path(), "commons/bo\0ard.md"), Err(GuardError::Malformed));
    }

    #[test]
    fn missing_file_is_not_found() {
        // 404 rather than 403 is deliberate, not incidental: with no
        // authentication, a distinct 403 would be an existence oracle for
        // paths outside the root. Leak less.
        let (root, _o) = farm();
        assert_eq!(resolve(root.path(), "commons/nope.md"), Err(GuardError::NotFound));
    }

    // ---- the three verified bypasses, each with a test that had no opinion before ----

    #[test]
    fn credential_denied_regardless_of_case() {
        // C1: macOS is case-insensitive, so `.ENV` opened the file `.env`.
        let (root, _o) = farm();
        for spelling in ["repos/app/.env", "repos/app/.ENV", "repos/app/.Env"] {
            assert_eq!(
                resolve(root.path(), spelling),
                Err(GuardError::DeniedCredential),
                "{spelling}"
            );
        }
    }

    #[test]
    fn innocently_named_symlink_to_a_credential_is_denied() {
        // C2: the guard judged the request string and opened the target.
        let (root, _o) = farm();
        assert_eq!(
            resolve(root.path(), "config.local"),
            Err(GuardError::DeniedCredential)
        );
    }

    #[test]
    fn git_directory_and_its_contents_are_denied() {
        // C3: .git was hidden from listings but fully readable.
        let (root, _o) = farm();
        assert_eq!(resolve(root.path(), ".git"), Err(GuardError::DeniedCredential));
        assert_eq!(
            resolve(root.path(), ".git/config"),
            Err(GuardError::DeniedCredential)
        );
    }

    #[test]
    fn credentials_that_are_not_named_dotenv_are_denied() {
        // C4: the denylist was one prefix found by inspecting one file.
        let (root, _o) = farm();
        assert_eq!(
            resolve(root.path(), "repos/app/Secrets.xcconfig"),
            Err(GuardError::DeniedCredential)
        );
    }

    #[test]
    fn noise_is_refused_when_requested_directly() {
        // I4: hiding node_modules from its parent while /tree?path=node_modules
        // still enumerated it was a rule that only looked like one.
        let (root, _o) = farm();
        assert_eq!(
            resolve(root.path(), "node_modules"),
            Err(GuardError::DeniedNoise)
        );
    }

    #[test]
    fn denial_does_not_over_reach() {
        let (root, _o) = farm();
        // Prefix rules must not eat ordinary prose...
        assert!(resolve(root.path(), "environment.md").is_ok());
        // ...and a committed template exists to be read.
        assert!(resolve(root.path(), "repos/app/.env.example").is_ok());
    }

    #[test]
    fn listings_hide_exactly_what_resolution_refuses() {
        for name in [
            ".env",
            ".ENV",
            ".git",
            "Secrets.xcconfig",
            "node_modules",
            "id_rsa",
            ".armillary",
        ] {
            assert!(is_hidden_from_listings(name), "{name} should be hidden");
        }
        for name in ["zojercommons", "environment.md", ".env.example", "README.md"] {
            assert!(!is_hidden_from_listings(name), "{name} should be visible");
        }
    }

    #[test]
    fn the_data_dir_is_denied_as_noise_at_any_depth() {
        // The engine serves the disk; without this, every session's full
        // event log becomes readable through /tree and /file the moment
        // `--data-dir` (defaulting to `<root>/.armillary`) resolves under
        // the served root.
        let (root, _o) = farm();
        fs::create_dir_all(root.path().join(".armillary/streams")).unwrap();
        fs::write(
            root.path().join(".armillary/streams/some-instance.jsonl"),
            "{}",
        )
        .unwrap();

        assert_eq!(
            resolve(root.path(), ".armillary"),
            Err(GuardError::DeniedNoise)
        );
        assert_eq!(
            resolve(root.path(), ".armillary/streams/some-instance.jsonl"),
            Err(GuardError::DeniedNoise)
        );
        assert!(is_hidden_from_listings(".armillary"));
    }

    #[test]
    fn openable_covers_prose_config_and_source() {
        for name in [
            "board.md", "standing-model.爻", "notes.txt",
            "modules.toml", "package.json", "app.yaml", "app.yml", "setup.ini",
            "guard.rs", "client.ts", "index.tsx", "app.js", "View.swift",
            "transcribe.py", "deploy.sh", "schema.sql", "global.css", "index.html",
        ] {
            assert!(is_openable(name), "{name} should open");
        }
    }

    #[test]
    fn openable_covers_extensionless_files_by_exact_name() {
        for name in [
            "LICENSE", "license", "README", "Makefile", "Dockerfile",
            ".gitignore", ".editorconfig", "Cargo.lock",
        ] {
            assert!(is_openable(name), "{name} should open");
        }
    }

    #[test]
    fn unknown_types_do_not_open() {
        for name in [
            "memo.m4a", "icon.png", "ephe.zip", "2026-06-26.kairosbackup",
            "sepl_54.se1", "mystery", "archive.tar.gz",
        ] {
            assert!(!is_openable(name), "{name} should not open");
        }
    }

    #[test]
    fn env_example_no_longer_opens() {
        // Accepted regression, pinned so it stays a decision rather than becoming a
        // surprise: `.env.example` is a committed template with no allowlisted
        // extension. Templates are read on the machine, not from a phone.
        assert!(!is_openable(".env.example"));
    }

    // ---- resolve_for_create (WD-3) ----

    #[test]
    fn a_create_resolves_a_path_that_does_not_exist_yet() {
        // `resolve` canonicalizes the full path and so returns NotFound for
        // anything absent — correct for reads, structurally useless for creates.
        let (root, _o) = farm();
        let target = resolve_for_create(root.path(), "commons/brand-new.md").unwrap();

        assert!(target.path.ends_with("brand-new.md"));
        assert!(target.path.starts_with(root.path().canonicalize().unwrap()));
        assert!(target.dirs_to_create.is_empty(), "commons/ already exists");
        assert!(!target.path.exists(), "resolution must not create anything");
    }

    #[test]
    fn a_create_under_a_symlinked_parent_pointing_out_is_refused() {
        // MUTATION-CHECKED. THE FIRST of two different escapes: a symlinked
        // PARENT. The naive implementation joins without canonicalizing the
        // parent, and a parent symlinked outside the root then lets a create
        // write outside the workspace entirely.
        //
        // This does NOT substitute for the dangling-symlink test below and that
        // one does not substitute for this: this covers a parent that resolves
        // out, that covers a final component that cannot be resolved at all —
        // and only the second is reachable BECAUSE full canonicalization
        // already failed.
        let (root, _o) = farm();
        assert_eq!(
            resolve_for_create(root.path(), "outside-link/planted.md"),
            Err(GuardError::Escaped)
        );
    }

    #[test]
    fn a_create_through_a_dangling_symlink_is_refused() {
        // MUTATION-CHECKED. THE SECOND escape — a confirmed one, reproduced by
        // running the superseded plan's code. `canonicalize` fails NotFound on
        // a dangling symlink, which is exactly what routes the path into this
        // function; the parent then canonicalizes inside the root, the filename
        // is innocent, containment passes, and `fs::write` FOLLOWS THE LINK and
        // writes outside the workspace.
        //
        // Not hypothetical here: `modules.local.toml` is a symlink into the
        // commons, so on a machine where the commons is not yet cloned it
        // dangles — the file the composition lock exists to protect is the
        // likeliest dangling symlink this workspace has.
        let (root, outside) = farm();
        let target = outside.path().join("planted.md");
        assert!(!target.exists(), "the fixture must be DANGLING");
        std::os::unix::fs::symlink(&target, root.path().join("dangling.md")).unwrap();

        // The precondition that routes it here at all.
        assert_eq!(resolve(root.path(), "dangling.md"), Err(GuardError::NotFound));

        assert_eq!(
            resolve_for_create(root.path(), "dangling.md"),
            Err(GuardError::Escaped)
        );
        assert!(!target.exists(), "nothing may have been planted outside the root");
    }

    #[test]
    fn a_create_whose_own_name_is_denied_is_refused() {
        // MUTATION-CHECKED. The step most easily forgotten: the parent is
        // innocent, so judging only the parent's components lets the name being
        // CREATED through. It does not exist yet, so nothing else will ever
        // look at it.
        //
        // **`secrets.json` is the discriminating case and `.env` is not.**
        // `.env` has no allowlisted extension, so it is refused by the openable
        // gate whether or not this judge runs — a test built on it stays green
        // under the mutation and lies. `secrets.json` is a credential the type
        // gate lets through, where this judge is the only thing standing
        // between it and being written. Asserting the exact variant, not merely
        // `is_err`, is the other half of the same discipline.
        let (root, _o) = farm();
        for name in ["commons/secrets.json", "secrets.json", "commons/credentials.toml"] {
            assert_eq!(
                resolve_for_create(root.path(), name),
                Err(GuardError::DeniedCredential),
                "{name}"
            );
        }
        assert!(resolve_for_create(root.path(), ".env").is_err(), ".env too");
    }

    #[test]
    fn a_create_under_a_denied_directory_is_refused() {
        let (root, _o) = farm();
        fs::create_dir_all(root.path().join("node_modules/pkg")).unwrap();
        assert_eq!(
            resolve_for_create(root.path(), "node_modules/pkg/x.md"),
            Err(GuardError::DeniedNoise)
        );
    }

    #[test]
    fn a_create_under_a_denied_directory_that_does_not_exist_yet_is_refused() {
        // MUTATION-CHECKED. New surface, created by parents-are-now-made: a
        // name that is about to become a real directory has to be judged before
        // it is conjured. `target/` does not exist in `farm()`, which is the
        // point.
        let (root, _o) = farm();
        assert_eq!(
            resolve_for_create(root.path(), "target/debug/notes.md"),
            Err(GuardError::DeniedNoise)
        );
        assert!(!root.path().join("target").exists(), "nothing may be created");
    }

    #[test]
    fn a_create_names_the_directories_it_would_have_to_make() {
        // WD-4, reversed: parents are created, and every directory created is
        // named in the result. The original objection stands and is not
        // abandoned — a typo'd path that silently conjures a directory tree is
        // a worse failure than one that refuses, because it is invisible
        // afterwards. Naming them answers it without prescribing a recovery the
        // session cannot perform (there is no `mkdir` verb and no bash).
        let (root, _o) = farm();
        let target = resolve_for_create(root.path(), "notes/2026/08/entry.md").unwrap();

        assert_eq!(
            target.dirs_to_create,
            vec!["notes", "notes/2026", "notes/2026/08"],
            "outermost first"
        );
        assert!(target.path.ends_with("entry.md"));
        assert!(!root.path().join("notes").exists(), "resolution creates nothing");
    }

    #[test]
    fn a_create_still_refuses_escapes_and_absolutes() {
        let (root, _o) = farm();
        for attempt in ["../escape.md", "/etc/passwd", "commons/../../x.md"] {
            assert_eq!(
                resolve_for_create(root.path(), attempt),
                Err(GuardError::Escaped),
                "{attempt}"
            );
        }
        assert_eq!(
            resolve_for_create(root.path(), "com\0mons/x.md"),
            Err(GuardError::Malformed)
        );
    }

    #[test]
    fn a_create_over_something_that_already_exists_is_refused_here() {
        // The contract WD-2 relies on: this function is for paths that do not
        // exist. A caller reaching it with an existing path skipped `resolve`,
        // and silently succeeding would hand it a path that never passed the
        // audited resolution. The symlink_metadata step catches this too — it
        // succeeds on an ordinary file just as it does on a dangling link.
        let (root, _o) = farm();
        assert_eq!(
            resolve_for_create(root.path(), "commons/board.md"),
            Err(GuardError::Escaped)
        );
    }

    #[test]
    fn worktrees_are_denied_as_derived_duplicate_state() {
        // SD-5. A worktree is a second checkout of a repo already composed here:
        // same category as `target/` and `.build`. Every file in one is readable
        // at its real path, and a search that walked both would return the same
        // file twice from two different branches.
        let (root, _o) = farm();
        fs::create_dir_all(root.path().join(".worktrees/feat-x/src")).unwrap();
        fs::write(root.path().join(".worktrees/feat-x/src/lib.rs"), "fn main() {}").unwrap();

        assert_eq!(resolve(root.path(), ".worktrees"), Err(GuardError::DeniedNoise));
        assert_eq!(
            resolve(root.path(), ".worktrees/feat-x/src/lib.rs"),
            Err(GuardError::DeniedNoise)
        );

        // The accepted cost, asserted rather than assumed: SD-5 knowingly removes
        // worktrees from /tree and /file, which the Explorer consumes. Pinned here
        // so it stays a decision rather than arriving later as a surprise report —
        // the `env_example_no_longer_opens` precedent.
        assert!(is_hidden_from_listings(".worktrees"));
    }
}
