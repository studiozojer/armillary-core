use armillary_engine::{
    app,
    log::store::LogStore,
    provider::{AnthropicProvider, KeylessProvider, ModelProvider},
    sessions::Sessions,
    state::{AppState, ModelConfig},
};
use clap::Parser;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    name = "armillary-engine",
    about = "files service + chat loop (v0, single-provider) for a composed armillary workspace"
)]
struct Args {
    /// Workspace root — the directory holding modules.toml.
    #[arg(long)]
    root: PathBuf,

    /// Interface to bind. Defaults to loopback; pass the tailnet address to
    /// serve the tailnet.
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::LOCALHOST))]
    bind: IpAddr,

    /// Port. 7778 sits beside the Python inbox endpoint on 7777.
    #[arg(long, default_value_t = 7778)]
    port: u16,

    /// Where the event log lives. Defaults to `<root>/.armillary` — a name
    /// `guard.rs` denies from every Explorer surface (`/tree`, `/file`) no
    /// matter where it resolves, so session logs are never readable through
    /// this same service.
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Which model pilots sessions. No provider call exists yet (Task 10) —
    /// this just rides in `AppState` until one does. The credential is never
    /// a flag (`ANTHROPIC_API_KEY` only, below) so it never lands in shell
    /// history or `ps`.
    #[arg(long, default_value = "claude-sonnet-5")]
    model: String,
}

/// True when `addr` falls inside Tailscale's CGNAT IPv4 range
/// (`100.64.0.0/10`) or its ULA IPv6 range (`fd7a:115c:a1e0::/48`) — the two
/// ranges Tailscale actually assigns tailnet addresses from.
fn in_tailscale_range(addr: IpAddr) -> bool {
    match addr {
        IpAddr::V4(v4) => {
            let octets = v4.octets();
            // /10: the first octet fixed at 100, the second octet's top two
            // bits fixed at 01 (i.e. 64..=127) — together 100.64.0.0 through
            // 100.127.255.255.
            octets[0] == 100 && (octets[1] & 0b1100_0000) == 0b0100_0000
        }
        IpAddr::V6(v6) => {
            let seg = v6.segments();
            // /48: the first three 16-bit segments fixed.
            seg[0] == 0xfd7a && seg[1] == 0x115c && seg[2] == 0xa1e0
        }
    }
}

/// constitution/instances.md A-5: a server must refuse to serve without
/// authentication on any interface that is not loopback or a device-
/// authenticating overlay, and where the overlay exception is claimed it
/// must bind ONE specific overlay address rather than a wildcard, so the
/// exception lives in the bind where it is checkable rather than in a
/// comment.
///
/// A-5's text names *any* device-authenticating overlay in general ("a
/// WireGuard-style mesh with per-device keys and an enforced access
/// policy"); it does not name Tailscale specifically. Recognizing exactly
/// Tailscale's two address ranges (`in_tailscale_range` above) — and
/// refusing every other non-loopback address, including an ordinary LAN
/// address like `192.168.1.7` — is an ENGINE-LOCAL DECISION, stricter than
/// what A-5's text requires, not a restatement of A-5 itself. A different
/// engine could recognize a different overlay's ranges and be equally
/// A-5-conformant.
///
/// (An earlier version of this function's home cited "D7", a decision-sheet
/// number from the sprint-1 design doc, as though it were normative. It
/// resolved nowhere in this repo, and A-5 — which does exist — was being
/// violated while the phantom rule was cited as justification: the refusal
/// caught only wildcards, so an ordinary LAN address like `192.168.1.7`
/// bound and served unauthenticated. This function is the fix.)
///
/// A pure, testable function rather than inline in `main()` — `main()`
/// itself isn't test-runnable, so the check that actually matters lives
/// here, where `#[cfg(test)]` can reach it.
fn bind_permitted(addr: IpAddr) -> Result<(), String> {
    if addr.is_loopback() || in_tailscale_range(addr) {
        return Ok(());
    }
    Err(format!(
        "refusing to bind {addr} — this serves unauthenticated reads of the whole \
         workspace, and constitution/instances.md A-5 permits that only on loopback \
         or on ONE specific address of a device-authenticating overlay, never a \
         wildcard. Failing open is not a mode; a LAN or public interface is not a \
         device-authenticating overlay. This engine recognizes Tailscale's ranges \
         (100.64.0.0/10 IPv4, fd7a:115c:a1e0::/48 IPv6) — an engine-local decision \
         stricter than A-5's text, see `bind_permitted`'s doc. Find yours with: \
         tailscale ip -4"
    ))
}

/// Resolves the Anthropic API key, in priority order: (1) `env_val` — the
/// `ANTHROPIC_API_KEY` environment variable's value, if the caller found one
/// set and it is non-empty; (2) `key_file` — read, trimmed of surrounding
/// whitespace (including a trailing newline), used if non-empty. Neither
/// present → `None`, i.e. keyless, exactly as before this fallback existed.
///
/// This resolution order is an ENGINE-LOCAL ERGONOMICS DECISION — a
/// convenience for running the engine day to day without exporting an env
/// var in every shell — not a constitution rule; nothing in `constitution/`
/// requires or forbids a key-file fallback, and a different engine need not
/// have one. The file's home, `~/.config/armillary/anthropic-key`, is
/// deliberately outside any directory this engine ever serves reads from
/// (contrast a workspace-local `.env`, which `guard.rs` refuses to serve —
/// see `dotenv_is_refused_even_when_guessed_directly` — precisely because a
/// secret sitting inside the served/composed tree is one `/file` request
/// away from leaking). A per-machine dotfile under the user's home config
/// directory instead matches this studio's standing convention for
/// machine-local secrets (`modules.local.toml`, `CLAUDE.local.md`):
/// configuration that varies by machine, is never committed, and lives
/// outside any tree this process composes or serves.
///
/// Takes `key_file` as a parameter — rather than resolving
/// `~/.config/armillary/anthropic-key` inline — so tests can point it at a
/// tempdir; `main()` below constructs the real path.
///
/// Never panics and never crashes boot: an unreadable file (e.g.
/// permissions) is treated the same as an absent one, aside from a stderr
/// warning naming the *path* (never the key, since none could be read) —
/// the Explorer must serve regardless of a key file's state.
fn resolve_api_key(env_val: Option<String>, key_file: &Path) -> Option<String> {
    if let Some(v) = env_val {
        if !v.is_empty() {
            return Some(v);
        }
    }

    match std::fs::read_to_string(key_file) {
        Ok(contents) => {
            let trimmed = contents.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => {
            eprintln!(
                "warning: could not read {} — continuing keyless: {e}",
                key_file.display()
            );
            None
        }
    }
}

/// The per-machine key file's real path: `$HOME/.config/armillary/anthropic-key`.
/// Built from `HOME` directly (`std::env::var_os`, no path-lookup crate) —
/// see `resolve_api_key`'s doc for why this file's home was chosen.
fn default_key_file() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
        .join(".config/armillary/anthropic-key")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if let Err(msg) = bind_permitted(args.bind) {
        return Err(msg.into());
    }

    let root = args
        .root
        .canonicalize()
        .map_err(|e| format!("--root {} is not readable: {e}", args.root.display()))?;

    // C-4: a workspace that composes nothing is a working host, not an error.
    // Said out loud rather than silently, because an empty Explorer and a
    // misaimed --root look identical from a phone.
    if !root.join("modules.toml").exists() {
        eprintln!(
            "warning: {} has no modules.toml — serving a workspace that composes nothing",
            root.display()
        );
    }

    // Which file boots a session is a manifest fact, not a flag — the same
    // C-3 reasoning as /composition: byte-derived from the manifest, never
    // re-derived by a model. A parse failure is a warning, not a fatal: the
    // Explorer must keep serving even when the manifest is malformed.
    let boot = match armillary_composition::parse_workspace(&root) {
        Ok(composition) => composition.router.boot,
        Err(e) => {
            eprintln!("warning: could not parse the manifest for [router] boot ({e}); sessions will start with no system prompt");
            None
        }
    };
    match &boot {
        Some(path) => eprintln!("boot: [router] boot = {path:?}"),
        None => eprintln!("boot: no [router] boot declared — sessions start with no system prompt"),
    }

    let data_dir = args
        .data_dir
        .clone()
        .unwrap_or_else(|| root.join(".armillary"));
    let store = LogStore::open(&data_dir).map_err(|e| {
        format!(
            "failed to open data dir {} — sessions cannot be logged: {e}",
            data_dir.display()
        )
    })?;
    let sessions = Arc::new(Sessions::new(store));

    // Never a flag, never logged — see the struct doc on `ModelConfig` and
    // `resolve_api_key`'s doc for the env-then-file priority.
    let env_val = std::env::var("ANTHROPIC_API_KEY").ok();
    let key_file = default_key_file();
    let env_present = env_val.as_deref().is_some_and(|v| !v.is_empty());
    let api_key = resolve_api_key(env_val, &key_file);

    let model = ModelConfig {
        model: args.model.clone(),
        api_key,
    };

    // The engine boots and serves the Explorer regardless of whether a model
    // is wired in: `KeylessProvider` fails every turn with a named error
    // (`no_api_key`) rather than refusing to start. Which provider is active,
    // and where its key came from, is announced — without ever printing the
    // key itself — see `ModelConfig`'s and `AnthropicProvider`'s redacting
    // `Debug` impls, and `resolve_api_key`'s doc for why only the source
    // (env vs. file), never the value, is nameable here.
    let provider: Arc<dyn ModelProvider> = match &model.api_key {
        Some(api_key) => {
            let source = if env_present {
                "key from env".to_string()
            } else {
                format!("key from {}", key_file.display())
            };
            eprintln!("provider: anthropic (model {}, {source})", model.model);
            Arc::new(AnthropicProvider {
                model: model.model.clone(),
                api_key: api_key.clone(),
            })
        }
        None => {
            eprintln!(
                "provider: keyless — no ANTHROPIC_API_KEY set and no usable key at {}; \
                 the Explorer works, but every send will fail with no_api_key",
                key_file.display()
            );
            Arc::new(KeylessProvider)
        }
    };

    let addr = SocketAddr::new(args.bind, args.port);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!(
        "armillary-engine serving {} on http://{}",
        root.display(),
        addr
    );

    axum::serve(
        listener,
        app(AppState {
            root,
            sessions,
            model,
            provider,
            boot,
        }),
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv6Addr;
    use std::str::FromStr;

    #[test]
    fn loopback_is_permitted() {
        assert!(bind_permitted(IpAddr::V4(Ipv4Addr::LOCALHOST)).is_ok());
        assert!(bind_permitted(IpAddr::V6(Ipv6Addr::LOCALHOST)).is_ok());
    }

    #[test]
    fn tailscale_ipv4_cgnat_range_is_permitted() {
        for addr in ["100.64.0.1", "100.100.1.1", "100.127.255.255"] {
            assert!(
                bind_permitted(IpAddr::from_str(addr).unwrap()).is_ok(),
                "{addr} should be permitted"
            );
        }
    }

    #[test]
    fn tailscale_ipv6_range_is_permitted() {
        assert!(bind_permitted(IpAddr::from_str("fd7a:115c:a1e0::1").unwrap()).is_ok());
    }

    #[test]
    fn lan_address_is_refused() {
        let err = bind_permitted(IpAddr::from_str("192.168.1.7").unwrap()).unwrap_err();
        assert!(err.contains("A-5"), "error should cite A-5: {err}");
    }

    #[test]
    fn public_address_is_refused() {
        let err = bind_permitted(IpAddr::from_str("8.8.8.8").unwrap()).unwrap_err();
        assert!(err.contains("A-5"), "error should cite A-5: {err}");
    }

    #[test]
    fn addresses_just_outside_the_tailscale_cgnat_range_are_refused() {
        // 100.63.x.x and 100.128.x.x sit just outside the /10 — the mask
        // math above must not accidentally widen the range.
        for addr in ["100.63.255.255", "100.128.0.0"] {
            assert!(
                bind_permitted(IpAddr::from_str(addr).unwrap()).is_err(),
                "{addr} should be refused"
            );
        }
    }

    // resolve_api_key: env-vs-file priority, trimming, and the never-crash
    // posture on an unreadable file.

    fn write_key_file(dir: &std::path::Path, contents: &str) -> PathBuf {
        let path = dir.join("anthropic-key");
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn env_wins_over_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_key_file(dir.path(), "file-key\n");
        let got = resolve_api_key(Some("env-key".to_string()), &path);
        assert_eq!(got, Some("env-key".to_string()));
    }

    #[test]
    fn file_used_when_env_absent() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_key_file(dir.path(), "file-key");
        let got = resolve_api_key(None, &path);
        assert_eq!(got, Some("file-key".to_string()));
    }

    #[test]
    fn file_contents_are_trimmed_of_a_trailing_newline() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_key_file(dir.path(), "file-key\n");
        let got = resolve_api_key(None, &path);
        assert_eq!(got, Some("file-key".to_string()));
    }

    #[test]
    fn an_empty_file_resolves_to_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_key_file(dir.path(), "   \n");
        let got = resolve_api_key(None, &path);
        assert_eq!(got, None);
    }

    #[test]
    #[cfg(unix)]
    fn an_unreadable_file_resolves_to_none_not_a_crash() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = write_key_file(dir.path(), "file-key");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();

        let got = resolve_api_key(None, &path);

        // Restore permissions so the tempdir can clean itself up.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

        assert_eq!(got, None);
    }

    #[test]
    fn no_sources_resolves_to_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("anthropic-key"); // never written
        let got = resolve_api_key(None, &path);
        assert_eq!(got, None);
    }

    #[test]
    fn an_empty_env_value_falls_through_to_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_key_file(dir.path(), "file-key");
        let got = resolve_api_key(Some(String::new()), &path);
        assert_eq!(got, Some("file-key".to_string()));
    }
}
