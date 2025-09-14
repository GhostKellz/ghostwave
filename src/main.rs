use anyhow::Result;
use clap::Parser;
use tracing::info;

mod audio;
mod autoload;
mod config;
mod noise_suppression;
mod phantomlink;
mod pipewire_module;
mod systemd;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "balanced")]
    profile: String,

    #[arg(short, long)]
    verbose: bool,

    #[arg(long)]
    pipewire_module: bool,

    #[arg(long)]
    phantomlink: bool,

    #[arg(long)]
    frames: Option<u32>,

    #[arg(long)]
    samplerate: Option<u32>,

    #[arg(long)]
    install_pipewire: bool,

    #[arg(long)]
    install_systemd: bool,

    #[arg(long)]
    uninstall: bool,

    #[arg(long)]
    service_start: bool,

    #[arg(long)]
    service_stop: bool,

    #[arg(long)]
    service_status: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("ghostwave={}", level))
        .init();

    // Handle installation and service management commands
    if args.install_pipewire {
        return autoload::setup_pipewire_autoload();
    }

    if args.install_systemd {
        return systemd::install_systemd_service();
    }

    if args.uninstall {
        autoload::remove_pipewire_autoload().ok();
        systemd::uninstall_systemd_service().ok();
        info!("✅ GhostWave completely uninstalled");
        return Ok(());
    }

    if args.service_start {
        return systemd::start_service();
    }

    if args.service_stop {
        return systemd::stop_service();
    }

    if args.service_status {
        return systemd::service_status();
    }

    info!("🎧 GhostWave starting - NVIDIA RTX Voice for Linux");
    info!("Profile: {}", args.profile);

    let config = config::Config::load(&args.profile)?
        .with_overrides(args.samplerate, args.frames);

    if args.phantomlink {
        info!("Starting in PhantomLink integration mode");
        phantomlink::run_phantomlink_mode(config).await?
    } else if args.pipewire_module {
        info!("Starting as native PipeWire module");
        pipewire_module::run(config).await?
    } else {
        info!("Starting standalone audio processor");
        audio::run_standalone(config).await?
    }

    Ok(())
}
