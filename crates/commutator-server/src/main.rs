//! `commutator`: an open-source AI gateway for robots.

mod http;

use std::{net::SocketAddr, path::PathBuf, sync::Arc};

use clap::{Parser, Subcommand};
use commutator_router::{RobotRequest, Router, RouterConfig, config::DEFAULT_CONFIG_YAML};
use serde_json::Value;

#[derive(Parser)]
#[command(
    name = "commutator",
    version,
    about = "An open-source AI gateway for robots: route every request to the right model, on the right compute, inside a safety envelope."
)]
struct Cli {
    /// Router config (YAML or JSON). Defaults to the built-in config.
    #[arg(long, global = true, env = "COMMUTATOR_CONFIG")]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the HTTP API.
    Serve {
        /// Address to listen on. Defaults to the config's listener address, else 127.0.0.1.
        #[arg(long)]
        host: Option<String>,
        /// Port to listen on. Defaults to the config's listener port, else 8899.
        #[arg(long)]
        port: Option<u16>,
    },
    /// Route one request given as JSON and print the decision.
    Route {
        /// Robot request JSON, e.g. '{"instruction":"pick up the cup","embodiment":"so101"}'.
        request: String,
        /// Route the configured pipeline (perceive, reason, act) instead of a single request.
        #[arg(long)]
        task: bool,
    },
    /// Load and validate a config, then print a short summary.
    CheckConfig,
}

fn load_config(path: &Option<PathBuf>) -> Result<RouterConfig, String> {
    match path {
        Some(p) => RouterConfig::load(p).map_err(|e| e.to_string()),
        None => RouterConfig::from_yaml_str(DEFAULT_CONFIG_YAML).map_err(|e| e.to_string()),
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let cli = Cli::parse();
    if let Err(message) = run(cli).await {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<(), String> {
    let config = load_config(&cli.config)?;
    match cli.command {
        Command::CheckConfig => {
            println!(
                "config OK: {} models, {} decisions, {} providers, {} pipeline stages",
                config.models.len(),
                config.decisions.len(),
                config.providers.len(),
                config.pipeline.len()
            );
            Ok(())
        }
        Command::Route { request, task } => {
            let router = Router::new(config).map_err(|e| e.to_string())?;
            let req: RobotRequest =
                serde_json::from_str(&request).map_err(|e| format!("invalid request JSON: {e}"))?;
            let out = if task {
                router.route_task(&req).to_value()
            } else {
                serde_json::to_value(router.route(&req)).expect("serializes")
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&out).expect("serializes")
            );
            Ok(())
        }
        Command::Serve { host, port } => {
            let listener_cfg = config.listener.clone();
            let host = host
                .or_else(|| {
                    listener_cfg
                        .get("address")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .unwrap_or_else(|| "127.0.0.1".into());
            let port = port
                .or_else(|| {
                    listener_cfg
                        .get("port")
                        .and_then(Value::as_u64)
                        .and_then(|p| u16::try_from(p).ok())
                })
                .unwrap_or(8899);
            let router = Router::new(config).map_err(|e| e.to_string())?;
            // This release routes only; calling model backends is not implemented yet.
            let state = Arc::new(http::AppState {
                router,
                dry_run: true,
            });
            let addr: SocketAddr = format!("{host}:{port}")
                .parse()
                .map_err(|e| format!("invalid address {host}:{port}: {e}"))?;
            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .map_err(|e| format!("cannot listen on {addr}: {e}"))?;
            tracing::info!(
                "commutator {} listening on http://{addr} (routing only)",
                env!("CARGO_PKG_VERSION")
            );
            axum::serve(listener, http::app(state))
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await
                .map_err(|e| e.to_string())
        }
    }
}
