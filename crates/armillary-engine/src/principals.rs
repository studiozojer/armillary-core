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

/// `$HOME/.config/armillary/devices` — beside `anthropic-key` and `zen-key`.
///
/// Built from `HOME` directly, matching `default_key_file`'s posture in
/// `main.rs`: no path-lookup crate, and the same directory the studio
/// already uses for machine-local secrets.
pub fn default_registry_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
        .join(".config/armillary/devices")
}

/// `$HOME/.config/armillary/host-token`.
pub fn default_host_token_path() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
        .join(".config/armillary/host-token")
}

/// Write one principal to `<dir>/<name>.toml`, creating `dir` if needed.
pub fn write_principal(dir: &Path, p: &Principal) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let text = toml::to_string(p)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    std::fs::write(dir.join(format!("{}.toml", p.name)), text)
}

/// Mint the `host` principal if none exists.
///
/// Returns the token when it minted one, `None` when a `host` principal was
/// already there — a re-mint on every start would invalidate the token a
/// local caller already holds, and this engine is a launchd service, so
/// restarts are routine rather than notable.
///
/// **`full ∧ manifest ≡ manifest`.** The host is minted with BOTH grants, so
/// the effective authority of a loopback caller is exactly what the manifest
/// already said — which is why this change is a no-op for host behavior on
/// the day it lands. `a_full_host_grant_changes_nothing_the_manifest_allowed`
/// in `routes/repos.rs` pins that.
pub fn ensure_host(dir: &Path, host_token_path: &Path) -> std::io::Result<Option<String>> {
    let host_exists = Registry::load(dir).names().iter().any(|p| p.name == "host");
    if host_exists && host_token_path.exists() {
        return Ok(None);
    }

    // A `host` principal whose token file is missing is not a completed mint,
    // it is a broken one, and re-minting repairs it. This makes deleting
    // `host-token` a supported way to force a fresh host token.

    let token = mint_token();
    write_principal(
        dir,
        &Principal {
            name: "host".to_string(),
            token_hash: hash_token(&token),
            grants: vec![Grant::Sync, Grant::Push],
            minted: humantime::format_rfc3339_millis(std::time::SystemTime::now()).to_string(),
        },
    )?;

    if let Some(parent) = host_token_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    use std::os::unix::fs::OpenOptionsExt;
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .mode(0o600)
        .open(host_token_path)?;
    f.write_all(token.as_bytes())?;

    // `.mode()` applies only when the file is created; if a wider-permissioned
    // file was already there, `.mode()` is ignored and this `set_permissions`
    // is the thing that narrows it. Both are needed and neither is redundant,
    // per the ruling in `run_git`'s stdin guard (David 2026-07-31: "Rather
    // than ship a green light that means nothing, the guarantee is stated
    // here and left unasserted").
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(host_token_path, std::fs::Permissions::from_mode(0o600))?;

    Ok(Some(token))
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
        // Guards a fresh-per-call seeded generator mistake (e.g., rng = StdRng::seed_from_u64(x)
        // reseeded identically each call). OsRng is confirmed by reading the code, not by this test.
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

    #[test]
    fn two_principals_each_token_resolves_to_its_own_name() {
        // Closes Critical 1 and Critical 2: a_token_authenticates_as_its_own_principal
        // is a tautology over whatever hash_token does, and all positive assertions
        // run against a one-principal registry. This test builds two principals with
        // two different minted tokens and asserts each token authenticates as *that*
        // principal by name.
        let dir = tempfile::tempdir().unwrap();
        let token1 = mint_token();
        let token2 = mint_token();

        std::fs::write(
            dir.path().join("iphone.toml"),
            format!(
                "name = \"iphone\"\ntoken_hash = \"{}\"\ngrants = [\"sync\"]\nminted = \"2026-08-07T00:00:00Z\"\n",
                hash_token(&token1)
            ),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("ipad.toml"),
            format!(
                "name = \"ipad\"\ntoken_hash = \"{}\"\ngrants = [\"sync\"]\nminted = \"2026-08-07T00:00:00Z\"\n",
                hash_token(&token2)
            ),
        )
        .unwrap();

        let reg = Registry::load(dir.path());
        let iphone = reg.authenticate(&token1).expect("token1 must authenticate");
        assert_eq!(iphone.name, "iphone", "token1 resolves to iphone");

        let ipad = reg.authenticate(&token2).expect("token2 must authenticate");
        assert_eq!(ipad.name, "ipad", "token2 resolves to ipad");
    }

    #[test]
    fn hash_token_contract_is_enforced() {
        // Closes Important 3: the sha256: prefix is unpinned. Asserts the contract
        // directly, not through authenticate. The key assertion is that different
        // inputs produce different hashes — breaking the tautology of the original test.
        let h1 = hash_token("token1");
        let h2 = hash_token("token2");

        assert!(h1.starts_with("sha256:"), "hash must start with sha256:");
        assert_eq!(h1.len(), 71, "7 chars (sha256:) + 64 chars hex = 71");
        assert_ne!(h1, h2, "different inputs produce different hashes");
    }

    #[test]
    fn empty_token_with_stored_empty_hash_rejects_empty_credential() {
        // Closes Important 4: the empty-token guard has no test that can see it.
        // A fixture storing hash_token("") represents a plausible enrollment bug.
        // Without the guard, authenticate("") would authenticate as that principal.
        let dir = tempfile::tempdir().unwrap();
        let empty_hash = hash_token("");
        std::fs::write(
            dir.path().join("broken.toml"),
            format!(
                "name = \"broken\"\ntoken_hash = \"{}\"\ngrants = [\"sync\"]\nminted = \"2026-08-07T00:00:00Z\"\n",
                empty_hash
            ),
        )
        .unwrap();

        let reg = Registry::load(dir.path());
        assert!(reg.authenticate("").is_none(), "empty credential must not authenticate, even against hash of empty token");
    }

    #[test]
    fn first_run_mints_a_host_principal_with_full_grants() {
        let dir = tempfile::tempdir().unwrap();
        let reg_dir = dir.path().join("devices");
        let token_path = dir.path().join("host-token");

        let minted = ensure_host(&reg_dir, &token_path).unwrap();
        assert!(minted.is_some(), "first run must mint");

        let reg = Registry::load(&reg_dir);
        let host = reg.names().into_iter().find(|p| p.name == "host").unwrap();
        assert!(host.holds(Grant::Sync));
        assert!(host.holds(Grant::Push));

        // The token is on disk where a local caller can read it, and it is
        // the one that authenticates.
        let on_disk = fs::read_to_string(&token_path).unwrap().trim().to_string();
        assert_eq!(on_disk, minted.unwrap());
        assert_eq!(reg.authenticate(&on_disk).unwrap().name, "host");
    }

    #[test]
    fn the_host_token_file_is_not_group_or_world_readable() {
        // The whole argument for a token on disk is that its protection is
        // the SAME filesystem boundary already guarding the SSH key this
        // grant ultimately spends. Mode 0600 is that argument; asserted
        // rather than assumed, because a default-permissioned write would
        // quietly make it weaker than the credential it guards.
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let token_path = dir.path().join("host-token");
        ensure_host(&dir.path().join("devices"), &token_path).unwrap();

        let mode = fs::metadata(&token_path).unwrap().permissions().mode();
        assert_eq!(mode & 0o077, 0, "no group or world bits");
    }

    #[test]
    fn a_second_run_does_not_re_mint() {
        // Re-minting on every start would invalidate the token a local
        // caller is already holding, on every restart — and the engine is a
        // launchd service now, so restarts are routine.
        let dir = tempfile::tempdir().unwrap();
        let reg_dir = dir.path().join("devices");
        let token_path = dir.path().join("host-token");

        let first = ensure_host(&reg_dir, &token_path).unwrap().unwrap();
        let second = ensure_host(&reg_dir, &token_path).unwrap();
        assert!(second.is_none(), "an existing host principal is left alone");
        // Observe what is on disk, not just the in-memory return value.
        let on_disk = fs::read_to_string(&token_path).unwrap().trim().to_string();
        assert_eq!(on_disk, first, "the token on disk is the first minted one");
        assert_eq!(Registry::load(&reg_dir).authenticate(&first).unwrap().name, "host");
    }

    #[test]
    fn a_broken_mint_with_missing_token_file_is_repaired() {
        // A `host` principal whose token file is missing is not a completed
        // mint; it is a broken one, and re-minting repairs it. This makes
        // deleting `host-token` a supported way to force a fresh host token.
        let dir = tempfile::tempdir().unwrap();
        let reg_dir = dir.path().join("devices");
        let token_path = dir.path().join("host-token");

        // Write a host principal but no token file, simulating a broken mint.
        write_principal(
            &reg_dir,
            &Principal {
                name: "host".to_string(),
                token_hash: hash_token("old_token"),
                grants: vec![Grant::Sync, Grant::Push],
                minted: "2026-08-07T00:00:00Z".to_string(),
            },
        )
        .unwrap();
        assert!(!token_path.exists(), "token file should not exist yet");

        // ensure_host should detect the broken mint and repair it.
        let repaired = ensure_host(&reg_dir, &token_path).unwrap();
        assert!(repaired.is_some(), "broken mint is repaired with a fresh token");

        // The new token file exists and authenticates.
        assert!(token_path.exists(), "token file now exists");
        let on_disk = fs::read_to_string(&token_path).unwrap().trim().to_string();
        assert_eq!(on_disk, repaired.unwrap(), "token on disk matches returned token");
        assert_eq!(
            Registry::load(&reg_dir).authenticate(&on_disk).unwrap().name,
            "host"
        );
    }
}
