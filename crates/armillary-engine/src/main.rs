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
