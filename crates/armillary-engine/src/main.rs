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

    // D7: the tailnet edge is the privacy boundary, and binding is what makes
    // that true rather than assumed. Refusing here turns the decision into a
    // property of the program instead of something each deploy must remember.
    if args.bind.is_unspecified() {
        return Err(format!(
            "refusing to bind {} — the engine serves unauthenticated reads of the whole \
             workspace, so it must bind loopback or a specific tailnet address (D7). \
             Find yours with: tailscale ip -4",
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
