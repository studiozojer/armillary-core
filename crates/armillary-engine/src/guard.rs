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
