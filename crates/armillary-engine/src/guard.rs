//! Path safety and the denylist — the engine's only security-critical surface.
//!
//! Both answer the same question ("may this path be reached?"), so they live
//! together. Splitting them would let a route consult one and forget the other.

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
    /// Legal and present, but never served.
    Denied,
}

impl GuardError {
    /// Kept beside the guard so every route maps these identically rather than
    /// each one deciding again.
    pub fn status(&self) -> StatusCode {
        match self {
            GuardError::Malformed => StatusCode::BAD_REQUEST,
            GuardError::Escaped => StatusCode::FORBIDDEN,
            GuardError::NotFound => StatusCode::NOT_FOUND,
            GuardError::Denied => StatusCode::FORBIDDEN,
        }
    }
}

/// Never listed and never served.
///
/// The engine serves the *disk*, not git, so `.gitignore` filters nothing:
/// `repos/kairos-engine/.env` is real, under the size cap, and valid UTF-8.
/// D7's "the tailnet edge is the privacy boundary" was decided about grove
/// content — credentials are a different category and were not in view.
const DENIED_PREFIXES: [&str; 1] = [".env"];

/// Never listed, but harmless if reached directly. Mostly a usability rule:
/// one `node_modules` in this workspace has 364 top-level entries and tens of
/// thousands nested, which would hang a phone on a useless list.
const HIDDEN_NAMES: [&str; 5] = [".git", "node_modules", "target", "build", ".next"];

/// True for names that must never be served, at any depth.
pub fn is_denied(name: &str) -> bool {
    DENIED_PREFIXES.iter().any(|prefix| name.starts_with(prefix))
}

/// True for names a directory listing should omit. Denied names are also hidden
/// — something never served should not be advertised.
pub fn is_hidden_from_listings(name: &str) -> bool {
    is_denied(name) || HIDDEN_NAMES.contains(&name)
}

/// Resolve a user-supplied path against the workspace root, refusing anything
/// that escapes it or that the denylist covers.
///
/// Four defenses, in order, because each catches what the others miss:
///   1. reject absolute paths *before* joining — `Path::join` with an absolute
///      argument discards the root entirely, the classic footgun;
///   2. reject `..` components explicitly, so traversal returns `Escaped`
///      rather than masquerading as `NotFound`;
///   3. reject denied names at any depth, so a guessed path fails even though
///      no listing ever revealed it;
///   4. canonicalize and re-check the prefix — the only one that catches a
///      symlink pointing out of the tree.
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
            Component::Prefix(_) | Component::RootDir => return Err(GuardError::Escaped),
            Component::Normal(name) => {
                if is_denied(&name.to_string_lossy()) {
                    return Err(GuardError::Denied);
                }
            }
            Component::CurDir => {}
        }
    }

    let root_canonical = root.canonicalize().map_err(|_| GuardError::NotFound)?;
    let candidate = root_canonical.join(rel);
    let canonical = candidate.canonicalize().map_err(|_| GuardError::NotFound)?;

    if !canonical.starts_with(&root_canonical) {
        return Err(GuardError::Escaped);
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A root containing both an inside-pointing and an outside-pointing
    /// symlink, because the real workspace has both: `models -> operators` and
    /// `CLAUDE.local.md` resolve inside and must pass; transcripts resolve to
    /// `/Volumes/cache` and must be refused.
    fn farm() -> (tempfile::TempDir, tempfile::TempDir) {
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret.md"), "not yours").unwrap();

        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("commons")).unwrap();
        fs::write(root.path().join("commons/board.md"), "# board").unwrap();
        fs::write(root.path().join(".env"), "TOKEN=hunter2").unwrap();
        fs::write(root.path().join("environment.md"), "# not a secret").unwrap();
        std::os::unix::fs::symlink(root.path().join("commons"), root.path().join("inside-link"))
            .unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("outside-link")).unwrap();

        (root, outside)
    }

    #[test]
    fn resolves_a_plain_path_inside_the_root() {
        let (root, _o) = farm();
        assert!(resolve(root.path(), "commons/board.md").is_ok());
    }

    #[test]
    fn empty_path_resolves_to_the_root() {
        let (root, _o) = farm();
        assert!(resolve(root.path(), "").is_ok());
    }

    #[test]
    fn follows_a_symlink_that_stays_inside() {
        let (root, _o) = farm();
        assert!(resolve(root.path(), "inside-link/board.md").is_ok());
    }

    #[test]
    fn refuses_a_symlink_that_leaves_the_root() {
        let (root, _o) = farm();
        assert_eq!(
            resolve(root.path(), "outside-link/secret.md"),
            Err(GuardError::Escaped)
        );
    }

    #[test]
    fn refuses_parent_traversal() {
        let (root, _o) = farm();
        for attempt in ["..", "../", "../../etc/passwd", "commons/../../escape"] {
            assert_eq!(
                resolve(root.path(), attempt),
                Err(GuardError::Escaped),
                "should have refused {attempt}"
            );
        }
    }

    #[test]
    fn refuses_absolute_paths() {
        let (root, _o) = farm();
        assert_eq!(resolve(root.path(), "/etc/passwd"), Err(GuardError::Escaped));
    }

    #[test]
    fn refuses_embedded_nul() {
        let (root, _o) = farm();
        assert_eq!(
            resolve(root.path(), "commons/bo\0ard.md"),
            Err(GuardError::Malformed)
        );
    }

    #[test]
    fn missing_file_is_not_found_rather_than_escaped() {
        let (root, _o) = farm();
        assert_eq!(
            resolve(root.path(), "commons/nope.md"),
            Err(GuardError::NotFound)
        );
    }

    #[test]
    fn refuses_dotenv_even_when_it_exists() {
        let (root, _o) = farm();
        assert_eq!(resolve(root.path(), ".env"), Err(GuardError::Denied));
    }

    #[test]
    fn refuses_dotenv_at_depth_so_guessing_does_not_work() {
        let (root, _o) = farm();
        fs::create_dir_all(root.path().join("repos/kairos-engine")).unwrap();
        fs::write(root.path().join("repos/kairos-engine/.env"), "SECRET=1").unwrap();
        assert_eq!(
            resolve(root.path(), "repos/kairos-engine/.env"),
            Err(GuardError::Denied)
        );
    }

    #[test]
    fn prefix_matching_does_not_over_reach() {
        // `.env` is a prefix rule, so a file merely *named* like it must still
        // resolve — otherwise the denylist quietly eats ordinary prose.
        let (root, _o) = farm();
        assert!(resolve(root.path(), "environment.md").is_ok());
    }

    #[test]
    fn listings_hide_noise_and_secrets_but_not_content() {
        assert!(is_hidden_from_listings("node_modules"));
        assert!(is_hidden_from_listings(".git"));
        assert!(is_hidden_from_listings(".env.local"));
        assert!(!is_hidden_from_listings("zojercommons"));
        assert!(!is_hidden_from_listings("environment.md"));
    }
}
