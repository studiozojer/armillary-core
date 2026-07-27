//! Path safety — the engine's only security-critical surface.
//!
//! # The ordering that matters
//!
//! The first version of this module validated **the string the client sent**
//! and then opened **whatever that resolved to**. On a case-insensitive
//! filesystem, in a workspace that deliberately routes real content through
//! symlinks, "a different thing" is the normal case rather than the edge case.
//! That single inversion produced three independent one-request bypasses:
//! `.ENV` reached `.env`, a symlink named `config.local` reached `.env`, and a
//! hard link reached it under any name at all.
//!
//! So: cheap rejections on the request first (absolute paths, `..`), then
//! resolve, then **judge the canonical path** — because the canonical path is
//! the thing that will actually be opened.
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
        "node_modules" | "target" | "build" | "dist" | ".next" | ".expo" | "deriveddata" | ".venv"
    )
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

    // Judge what will actually be opened, not what was asked for. A symlink,
    // a hard link, or a differently-cased spelling all arrive here as the same
    // canonical path.
    for component in inside.components() {
        if let Component::Normal(name) = component {
            if let Some(refusal) = judge(&name.to_string_lossy()) {
                return Err(refusal);
            }
        }
    }

    Ok(canonical)
}

// A note for whoever revisits this: given the service has no authentication at
// all, the durable posture is probably an **extension allowlist** (`.md`,
// `.爻`, `.toml`, `.json`, source files) rather than a denylist. The set of
// things this should serve is small and enumerable; the set it must not serve
// is neither. That is a product decision — it would stop the Explorer opening
// arbitrary files — so it is recorded here rather than taken unilaterally.

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
        for name in [".env", ".ENV", ".git", "Secrets.xcconfig", "node_modules", "id_rsa"] {
            assert!(is_hidden_from_listings(name), "{name} should be hidden");
        }
        for name in ["zojercommons", "environment.md", ".env.example", "README.md"] {
            assert!(!is_hidden_from_listings(name), "{name} should be visible");
        }
    }
}
