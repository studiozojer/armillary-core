use armillary_engine::{app, state::AppState};
use clap::Parser;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // constitution/instances.md A-5: a server must refuse to serve without
    // authentication on any interface that is not loopback or a device-
    // authenticating overlay — and where the overlay exception is claimed, it
    // must bind ONE specific overlay address rather than a wildcard, so the
    // exception lives in the bind where it is checkable rather than in a
    // comment. This refusal is that clause.
    //
    // (An earlier version of this comment cited "D7", a decision-sheet number
    // from the sprint-1 design doc, as though it were normative. It resolved
    // nowhere in this repo, and A-5 — which does exist — was being violated
    // while the phantom rule was cited as justification.)
    if args.bind.is_unspecified() {
        return Err(format!(
            "refusing to bind {} — this serves unauthenticated reads of the whole \
             workspace, and constitution/instances.md A-5 permits that only on loopback \
             or on ONE specific address of a device-authenticating overlay, never a \
             wildcard. Find yours with: tailscale ip -4",
            args.bind
        )
        .into());
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

    let addr = SocketAddr::new(args.bind, args.port);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!(
        "armillary-engine serving {} on http://{}",
        root.display(),
        addr
    );

    axum::serve(listener, app(AppState { root })).await?;
    Ok(())
}
