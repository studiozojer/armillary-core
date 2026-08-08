use armillary_engine::{
    app,
    log::store::LogStore,
    models,
    provider::{KeyedProviders, ProviderFor},
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
    about = "files service + chat loop for a composed armillary workspace, piloted per-instance across Anthropic and OpenCode Zen"
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

    /// The process-wide default model — the fallback for any instance whose
    /// own log names none; an instance's own recorded model wins over this
    /// (`loop_::run_turn`). A `zen/<slug>` value routes to the OpenAI-compat
    /// provider against OpenCode Zen (key: `OPENCODE_ZEN_API_KEY` or
    /// `~/.config/armillary/zen-key`); anything else is an Anthropic model
    /// name. The credential is never a flag (env or key file only, below) so
    /// it never lands in shell history or `ps`.
    ///
    /// Absent, the default comes from `models.toml`, then `claude-sonnet-5`.
    #[arg(long)]
    model: Option<String>,
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

/// Reads `[router] boot` from the workspace at `root` — the declaration that
/// gives a session its system prompt.
///
/// Which file boots a session is a manifest fact, not a flag — the same C-3
/// reasoning as `/composition`: byte-derived from the manifest, never re-derived
/// by a model. `parse_workspace` is what applies the C-6 overlay, so a private
/// `modules.local.toml` can name a different boot file than the public
/// `modules.toml` and win.
///
/// A parse failure is a warning, not a fatal: the Explorer must keep serving even
/// when the manifest is malformed. The cost of the warning posture is that the
/// session starts with no system prompt, which is why it says so.
///
/// A pure, testable function rather than inline in `main()` — like
/// `bind_permitted` and `resolve_api_key` above. Reading the *right* field of the
/// *merged* composition is the entire join between "the manifest declares a boot
/// file" and "the instance records one"; inline in `main()` it would be
/// unreachable from any test, and reading `router.contains.first()`, or skipping
/// the overlay, would present as "the phone still has no system prompt" rather
/// than as a failing test.
fn declared_boot(root: &Path) -> Option<String> {
    match armillary_composition::parse_workspace(root) {
        Ok(composition) => composition.router.boot,
        Err(e) => {
            eprintln!("warning: could not parse the manifest for [router] boot ({e}); sessions will start with no system prompt");
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

/// The zen key's home, beside `anthropic-key` — per-provider key files, one
/// ritual (the board's ratified shape, 2026-08-07).
fn default_zen_key_file() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
        .join(".config/armillary/zen-key")
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

    // See `declared_boot`'s doc: a manifest fact, overlay-merged, warn-not-fatal.
    let boot = declared_boot(&root);
    match &boot {
        Some(path) => eprintln!("boot: [router] boot = {path:?}"),
        None => eprintln!("boot: no [router] boot declared — sessions start with no system prompt"),
    }

    // Announced for the same reason `boot:` is: each gate lives in
    // `Router.extra`, which C-5 forbids validating, so a misspelled `snyc` or
    // `psuh` disables the grant with no error anywhere. Both are named, not
    // just `sync` — `push` is a second, independently misspellable key (D7:
    // it lets the host publish under its own credential, a strictly bigger
    // authority than fetch/fast-forward), and a typo there would be exactly
    // as silent if it went unannounced.
    let banner_comp =
        armillary_composition::parse_workspace(&root).unwrap_or_default();
    let sync_on = armillary_engine::repos::gate_enabled(&banner_comp);
    let push_on = armillary_engine::repos::push_enabled(&banner_comp);
    if sync_on {
        eprintln!(
            "sync: enabled by [router] sync — /repos/{{name}}/fetch and /repos/{{name}}/pull will act"
        );
    } else {
        eprintln!(
            "sync: not declared — /repos/{{name}}/fetch and /repos/{{name}}/pull will refuse (add `sync = true` under [router] in modules.local.toml); GET /repos still reads status"
        );
    }
    if push_on {
        eprintln!("push: enabled by [router] push — /repos/{{name}}/push will act");
    } else {
        eprintln!(
            "push: not declared — /repos/{{name}}/push will refuse (add `push = true` under [router] in modules.local.toml)"
        );
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

    // `--model` wins if passed; otherwise `models.toml`'s own `default`
    // (Task 4's `models::declared_default`); otherwise this literal, exactly
    // what clap's `default_value` used to produce. Unchanged by Task 3, which
    // gives the INSTANCE's own recorded model precedence over this default
    // per turn (`loop_::run_turn`, not here) — this is only the process-wide
    // fallback for an instance that names none.
    let model_str = args
        .model
        .clone()
        .or_else(models::declared_default)
        .unwrap_or_else(|| "claude-sonnet-5".to_string());

    // Never a flag, never logged — see `resolve_api_key`'s doc for the
    // env-then-file priority. BOTH keys, not just the configured model's —
    // an Anthropic instance and a Zen instance now coexist in one engine
    // (design decision 1), so a key resolved lazily would be a key resolved
    // never. Each line below names its env var and, when there is no usable
    // key, the key file's PATH — an operator needs to know WHERE to drop a
    // key on this host, which is exactly what the pre-this-task keyless
    // message said. Sources (env vs. file, or the file's path) are named;
    // key VALUES never are.
    let anthropic_key_file = default_key_file();
    let anthropic_env = std::env::var("ANTHROPIC_API_KEY").ok();
    let anthropic_env_present = anthropic_env.as_deref().is_some_and(|v| !v.is_empty());
    let anthropic_key = resolve_api_key(anthropic_env, &anthropic_key_file);

    let zen_key_file = default_zen_key_file();
    let zen_env = std::env::var("OPENCODE_ZEN_API_KEY").ok();
    let zen_env_present = zen_env.as_deref().is_some_and(|v| !v.is_empty());
    let zen_key = resolve_api_key(zen_env, &zen_key_file);

    let anthropic_key_present = anthropic_key.is_some();
    let zen_key_present = zen_key.is_some();

    eprintln!(
        "provider: anthropic {}",
        match (anthropic_key_present, anthropic_env_present) {
            (true, true) => "ready (key from env)".to_string(),
            (true, false) => format!("ready (key from {})", anthropic_key_file.display()),
            (false, _) => format!(
                "no ANTHROPIC_API_KEY set and no usable key at {} — those instances fail with no_api_key",
                anthropic_key_file.display()
            ),
        }
    );
    eprintln!(
        "provider: opencode-zen {}",
        match (zen_key_present, zen_env_present) {
            (true, true) => "ready (key from env)".to_string(),
            (true, false) => format!("ready (key from {})", zen_key_file.display()),
            (false, _) => format!(
                "no OPENCODE_ZEN_API_KEY set and no usable key at {} — those instances fail with no_api_key",
                zen_key_file.display()
            ),
        }
    );

    // `model` keeps the full spelling (`zen/<slug>` included) — it is what
    // the log's `model` field records, and the prefix is honest provenance.
    let model = ModelConfig { model: model_str };

    // The engine boots and serves the Explorer regardless of whether a
    // model is wired in: a model whose provider has no key resolves to
    // `KeylessProvider` per turn (`KeyedProviders::provider_for`), which
    // fails with the named `no_api_key` error rather than refusing to
    // start.
    let providers: Arc<dyn ProviderFor> = Arc::new(KeyedProviders { anthropic_key, zen_key });

    let models_path = models::default_path();

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
            providers,
            models_path,
            anthropic_key_present,
            zen_key_present,
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

    // declared_boot: the join between a manifest on disk and `AppState.boot`.
    // Everything else in the suite either passes `boot: None` or hand-writes a
    // `Some(..)`, so without these the engine could read the wrong field, or read
    // only `modules.toml`, and stay green.

    #[test]
    fn declared_boot_reads_the_router_boot_key() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("modules.toml"),
            "[router]\ncontains = [\"CLAUDE.md\"]\nboot = \"getting-started.md\"\n",
        )
        .unwrap();
        assert_eq!(
            declared_boot(dir.path()),
            Some("getting-started.md".to_string())
        );
    }

    #[test]
    fn declared_boot_is_none_when_the_router_declares_only_contains() {
        // The failure mode this pins: reading `router.contains.first()` instead of
        // `router.boot` would return "CLAUDE.md" here and look plausible.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("modules.toml"),
            "[router]\ncontains = [\"CLAUDE.md\"]\n",
        )
        .unwrap();
        assert_eq!(declared_boot(dir.path()), None);
    }

    #[test]
    fn declared_boot_is_none_when_nothing_is_composed() {
        let dir = tempfile::tempdir().unwrap();
        // C-4: no modules.toml at all is a working host, not an error.
        assert_eq!(declared_boot(dir.path()), None);
    }

    #[test]
    fn declared_boot_lets_the_private_overlay_win() {
        // C-6 at the engine level, not only as a composition-crate fixture: the
        // whole point of a per-machine overlay is that it can name a different
        // boot file, so reading `modules.toml` alone would be silently wrong.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("modules.toml"),
            "[router]\ncontains = [\"CLAUDE.md\"]\nboot = \"public-boot.md\"\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("modules.local.toml"),
            "[router]\nboot = \"getting-started.md\"\n",
        )
        .unwrap();
        assert_eq!(
            declared_boot(dir.path()),
            Some("getting-started.md".to_string())
        );
    }

    #[test]
    fn declared_boot_takes_the_base_when_the_overlay_is_silent_about_boot() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("modules.toml"),
            "[router]\nboot = \"public-boot.md\"\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("modules.local.toml"),
            "[[repos]]\nname = \"kairos\"\npath = \"repos/kairos\"\n",
        )
        .unwrap();
        assert_eq!(declared_boot(dir.path()), Some("public-boot.md".to_string()));
    }

    #[test]
    fn declared_boot_warns_rather_than_aborting_on_a_malformed_manifest() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("modules.toml"), "[router\nboot = ").unwrap();
        // The Explorer must keep serving a workspace whose manifest is broken.
        assert_eq!(declared_boot(dir.path()), None);
    }
}
