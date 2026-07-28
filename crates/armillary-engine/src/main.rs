use armillary_engine::{
    app,
    log::store::LogStore,
    provider::{AnthropicProvider, KeylessProvider, ModelProvider},
    sessions::Sessions,
    state::{AppState, ModelConfig},
};
use clap::Parser;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    name = "armillary-engine",
    about = "read-only files service for a composed armillary workspace"
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
    let model = ModelConfig {
        model: args.model.clone(),
        // Never a flag, never logged — see the struct doc on `ModelConfig`.
        api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
    };

    // The engine boots and serves the Explorer regardless of whether a model
    // is wired in: `KeylessProvider` fails every turn with a named error
    // (`no_api_key`) rather than refusing to start. Which provider is active
    // is announced (without ever printing the key itself — see
    // `ModelConfig`'s and `AnthropicProvider`'s redacting `Debug` impls).
    let provider: Arc<dyn ModelProvider> = match &model.api_key {
        Some(api_key) => {
            eprintln!("provider: AnthropicProvider (model {})", model.model);
            Arc::new(AnthropicProvider {
                model: model.model.clone(),
                api_key: api_key.clone(),
            })
        }
        None => {
            eprintln!(
                "provider: KeylessProvider — no ANTHROPIC_API_KEY set; the Explorer works, \
                 but every send will fail with no_api_key"
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
}
