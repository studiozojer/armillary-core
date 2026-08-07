//! Who is asking, and what they are allowed to ask for.
//!
//! # Why this lives under `~/.config`, not in the workspace
//!
//! `modules.local.toml` — the file the router's own docs call "private,
//! per-machine bindings" — is a SYMLINK into `zojercommons/setup/`, the
//! canonical copy that syncs to every machine. An authority granted there
//! is granted everywhere, which is the defect found on 2026-08-05 by trying
//! to grant `push` on one host and discovering it was not expressible.
//!
//! Composition can be shared; authority cannot. A file under the user's
//! home config directory is *structurally* unable to reach the commons,
//! which is why per-host scoping falls out of the location rather than
//! needing machinery.
//!
//! # The registry is read per request
//!
//! Deliberately no cache. The manifest gates already have this property and
//! it is one David named as valuable at the grant site: "both keys are read
//! per request, so neither takes a restart, and deleting a line revokes it."
//! A cached registry would make `revoke` mean "revoked after a restart",
//! which is a different and worse promise.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// The two authorities a principal can hold, mirroring the manifest keys
/// that bound them.
///
/// **Exact, with no inheritance.** `Push` does not imply `Sync`. D7 split
/// the manifest keys for a reason — fetching is "make this host talk to a
/// remote", publishing is "spend this host's credential with no undo" — and
/// collapsing them here would quietly re-merge what that decision separated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Grant {
    Sync,
    Push,
}

impl Grant {
    pub fn as_str(&self) -> &'static str {
        match self {
            Grant::Sync => "sync",
            Grant::Push => "push",
        }
    }

    pub fn parse(s: &str) -> Option<Grant> {
        match s {
            "sync" => Some(Grant::Sync),
            "push" => Some(Grant::Push),
            _ => None,
        }
    }
}

/// A fresh 256-bit token, lowercase hex.
///
/// `OsRng` is the operating system's CSPRNG, not a userspace generator that
/// could be seeded reproducibly. That distinction is the whole security of
/// this scheme: everything else here assumes the token is unguessable.
pub fn mint_token() -> String {
    use rand::{rngs::OsRng, RngCore};
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `"sha256:<hex>"` — what the registry stores in place of a token.
///
/// **SHA-256 and deliberately not a password KDF.** argon2 and bcrypt exist
/// to make LOW-ENTROPY secrets expensive to guess; there is nothing to guess
/// in 256 bits of CSPRNG output, so a work factor here would buy latency and
/// no security. Reaching for one anyway is cargo-cult, and this comment is
/// here so the next reader does not "fix" it.
pub fn hash_token(token: &str) -> String {
    format!("sha256:{}", crate::hash::sha256_hex(token.as_bytes()))
}

/// One enrolled device (or the host itself).
///
/// `token_hash` is the SHA-256 of the token, prefixed `sha256:`. The token
/// itself is never stored, never recoverable, and never logged — a lost
/// token is re-enrolled, not looked up.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Principal {
    pub name: String,
    pub token_hash: String,
    pub grants: Vec<Grant>,
    pub minted: String,
}

impl Principal {
    pub fn holds(&self, g: Grant) -> bool {
        self.grants.contains(&g)
    }
}

/// Every principal on this host.
#[derive(Debug, Default)]
pub struct Registry {
    principals: Vec<Principal>,
}

impl Registry {
    /// Read every `*.toml` in `dir`.
    ///
    /// **A missing directory is an empty registry, and an unparseable file
    /// is skipped, not fatal.** Both postures fail closed downstream — an
    /// empty registry holds no grants — and the alternative is worse than it
    /// looks: returning `Err` for one malformed file takes every other
    /// device down with it, presenting as "all my devices stopped working"
    /// with a stray `.DS_Store` as the cause. The manifest's own warning
    /// posture (`declared_modules`) made exactly this trade, and the cost
    /// recorded there is that a silent degrade is indistinguishable from
    /// working — so this one is NOT silent: it warns per file, naming the
    /// path.
    pub fn load(dir: &Path) -> Registry {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Registry::default();
        };

        let mut principals = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            match std::fs::read_to_string(&path)
                .map_err(|e| e.to_string())
                .and_then(|text| toml::from_str::<Principal>(&text).map_err(|e| e.to_string()))
            {
                Ok(p) => principals.push(p),
                Err(e) => eprintln!(
                    "warning: skipping unreadable principal {} — {e}",
                    path.display()
                ),
            }
        }
        Registry { principals }
    }

    pub fn names(&self) -> Vec<&Principal> {
        self.principals.iter().collect()
    }

    /// The principal this token belongs to, or none.
    ///
    /// The presented token is HASHED and the hashes compared — the registry
    /// never holds anything that could be replayed, so a leaked registry
    /// file is not a set of credentials.
    ///
    /// **No constant-time comparison, on purpose.** A timing oracle on this
    /// comparison would leak how many leading hex digits of a *hash* match,
    /// and turning that into a valid token is a preimage attack on SHA-256,
    /// not an iteration. Constant-time comparison matters where an attacker
    /// can walk a secret byte by byte; here the walk terminates at a problem
    /// nobody has solved.
    pub fn authenticate(&self, token: &str) -> Option<&Principal> {
        if token.is_empty() {
            return None;
        }
        let presented = hash_token(token);
        self.principals.iter().find(|p| p.token_hash == presented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// One registry directory with three principals and one file that is not a
    /// principal at all — a stray `.DS_Store` is the realistic case, and a
    /// reader that errors on it is a reader that stops working when Finder
    /// visits the directory.
    fn farm() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("iphone.toml"),
            "name = \"iphone\"\ntoken_hash = \"sha256:aaaa\"\ngrants = [\"sync\", \"push\"]\nminted = \"2026-08-07T14:22:31Z\"\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("ipad.toml"),
            "name = \"ipad\"\ntoken_hash = \"sha256:bbbb\"\ngrants = [\"sync\"]\nminted = \"2026-08-07T14:23:00Z\"\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("laptop.toml"),
            "name = \"laptop\"\ntoken_hash = \"sha256:cccc\"\ngrants = [\"push\"]\nminted = \"2026-08-07T14:24:00Z\"\n",
        )
        .unwrap();
        fs::write(dir.path().join(".DS_Store"), "junk").unwrap();
        dir
    }

    #[test]
    fn loads_every_principal_in_the_directory() {
        let dir = farm();
        let reg = Registry::load(dir.path());
        let mut names: Vec<&str> = reg.names().iter().map(|p| p.name.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["ipad", "iphone", "laptop"]);
    }

    #[test]
    fn grants_are_exact_with_no_inheritance() {
        // `push` does NOT imply `sync`, and absence is denial. Stated as a
        // test because "push is the bigger authority" invites an implementer
        // to make it a superset, and D7 deliberately kept them independent.
        let dir = farm();
        let reg = Registry::load(dir.path());
        let ipad = reg.names().into_iter().find(|p| p.name == "ipad").unwrap();
        assert!(ipad.holds(Grant::Sync));
        assert!(!ipad.holds(Grant::Push));
    }

    #[test]
    fn push_does_not_imply_sync() {
        // Regression guard for the specific direction that was unobservable
        // in the original fixture: a principal with push-only must not hold sync.
        let dir = farm();
        let reg = Registry::load(dir.path());
        let laptop = reg.names().into_iter().find(|p| p.name == "laptop").unwrap();
        assert!(laptop.holds(Grant::Push));
        assert!(!laptop.holds(Grant::Sync));
    }

    #[test]
    fn an_unparseable_file_is_skipped_not_fatal() {
        // A reader that returns Err on one bad file takes every OTHER
        // principal down with it — and the failure mode would be "all my
        // devices stopped working", traced to a stray file. Skip and warn.
        let dir = farm();
        fs::write(dir.path().join("broken.toml"), "this is not toml = = =").unwrap();
        let reg = Registry::load(dir.path());
        assert_eq!(reg.names().len(), 3, "the three good ones still load");
    }

    #[test]
    fn a_missing_directory_is_an_empty_registry() {
        // First run, before any minting. Empty is the honest answer and it
        // fails closed downstream: no principal holds any grant.
        let reg = Registry::load(Path::new("/nonexistent/definitely/not/here"));
        assert!(reg.names().is_empty());
    }

    #[test]
    fn a_minted_token_is_256_bits_of_hex() {
        let t = mint_token();
        assert_eq!(t.len(), 64, "32 bytes, lowercase hex");
        assert!(t.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn two_minted_tokens_differ() {
        // Guards the one mistake that would silently destroy the whole
        // scheme: a seeded or constant RNG. Not a probabilistic test — a
        // collision here means the source is not the OS CSPRNG.
        assert_ne!(mint_token(), mint_token());
    }

    #[test]
    fn a_token_authenticates_as_its_own_principal() {
        let dir = tempfile::tempdir().unwrap();
        let token = mint_token();
        std::fs::write(
            dir.path().join("iphone.toml"),
            format!(
                "name = \"iphone\"\ntoken_hash = \"{}\"\ngrants = [\"sync\"]\nminted = \"2026-08-07T00:00:00Z\"\n",
                hash_token(&token)
            ),
        )
        .unwrap();

        let reg = Registry::load(dir.path());
        let who = reg.authenticate(&token).expect("the minted token must authenticate");
        assert_eq!(who.name, "iphone");
    }

    #[test]
    fn a_wrong_token_authenticates_as_nobody() {
        let dir = farm();
        let reg = Registry::load(dir.path());
        assert!(reg.authenticate(&mint_token()).is_none());
        assert!(reg.authenticate("").is_none());
        // The stored hash spelled back at us is NOT a token. Guards an
        // implementation that compares the presented string to `token_hash`
        // directly instead of hashing it first — which would make the
        // registry file itself the credential.
        assert!(reg.authenticate("sha256:aaaa").is_none());
    }
}
