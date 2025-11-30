use anyhow::Result;
use clap::Parser;
use tracing::{info, warn};
use ghostwave_core::latency_optimizer::{LatencyOptimizer, LatencyConfig};

mod alsa_module;
mod audio;
mod autoload;
mod config;
mod device_detection;
mod ipc;
mod jack_module;
mod low_latency;
mod noise_suppression;
mod phantomlink;
mod pipewire_module;
mod rtx_acceleration;
mod systemd;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Processing profile to use
    #[arg(short, long, default_value = "balanced")]
    profile: String,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Enable quiet mode (errors only)
    #[arg(short, long)]
    quiet: bool,

    /// Buffer size in frames (32-4096, must be power of 2)
    #[arg(long)]
    frames: Option<u32>,

    /// Sample rate in Hz (44100, 48000, 96000, 192000)
    #[arg(long)]
    samplerate: Option<u32>,

    /// Input device name or "auto" for automatic selection
    #[arg(long)]
    input: Option<String>,

    /// Output device name or "auto" for automatic selection
    #[arg(long)]
    output: Option<String>,

    /// Number of audio channels (1=mono, 2=stereo)
    #[arg(long)]
    channels: Option<u8>,

    /// Configuration file path (overrides default)
    #[arg(long)]
    config: Option<String>,

    /// Run in PipeWire module mode
    #[arg(long)]
    pipewire_module: bool,

    /// Run in PhantomLink integration mode
    #[arg(long)]
    phantomlink: bool,

    /// Enable NVIDIA RTX acceleration
    #[arg(long)]
    nvidia_rtx: bool,

    /// Audio backend to use (auto, pipewire, alsa, jack, cpal)
    #[arg(long)]
    backend: Option<String>,

    /// Use ALSA backend directly
    #[arg(long)]
    alsa: bool,

    /// Use JACK backend
    #[arg(long)]
    jack: bool,

    /// Enable IPC server for remote control
    #[arg(long)]
    ipc_server: bool,

    /// IPC socket path
    #[arg(long)]
    ipc_socket: Option<String>,

    /// Run system diagnostics
    #[arg(long)]
    doctor: bool,

    /// Run performance benchmark
    #[arg(long)]
    bench: bool,

    /// Validate configuration and exit
    #[arg(long)]
    dry_run: bool,

    /// Install PipeWire autoload module
    #[arg(long)]
    install_pipewire: bool,

    /// Install systemd user service
    #[arg(long)]
    install_systemd: bool,

    /// Uninstall all components
    #[arg(long)]
    uninstall: bool,

    /// Start systemd service
    #[arg(long)]
    service_start: bool,

    /// Stop systemd service
    #[arg(long)]
    service_stop: bool,

    /// Show systemd service status
    #[arg(long)]
    service_status: bool,

    /// Enable real-time priority
    #[arg(long)]
    realtime: bool,

    /// CPU cores for affinity (comma-separated)
    #[arg(long)]
    cpu_affinity: Option<String>,

    /// Target latency in milliseconds
    #[arg(long)]
    latency: Option<f32>,

    /// Memory pool size for audio buffers
    #[arg(long)]
    memory_pool_size: Option<usize>,

    /// Disable all processing (passthrough mode)
    #[arg(long)]
    passthrough: bool,

    /// Show version information and exit
    #[arg(long)]
    version: bool,

    /// List available profiles
    #[arg(long)]
    list_profiles: bool,

    /// List available devices
    #[arg(long)]
    list_devices: bool,

    /// Generate shell completion script
    #[arg(long)]
    generate_completion: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Handle special commands first
    if args.version {
        println!("GhostWave {}", env!("CARGO_PKG_VERSION"));
        println!("Linux RTX Voice Alternative");
        return Ok(());
    }

    if args.list_profiles {
        println!("Available processing profiles:");
        println!("  balanced   - Balanced noise reduction for everyday use");
        println!("  streaming  - Aggressive noise reduction for streaming");
        println!("  studio     - Minimal processing for professional recording");
        return Ok(());
    }

    if args.list_devices {
        println!("Available audio devices:");
        let detector = device_detection::DeviceDetector::new();
        let devices = detector.detect_devices().await?;

        for device in devices {
            println!("  • {} {} ({})", device.vendor, device.model, device.name);
            if device.is_xlr_interface {
                println!("    ✅ XLR Interface - Profile: {}", device.recommended_profile);
            }
        }
        return Ok(());
    }

    if let Some(shell) = args.generate_completion {
        generate_completions(&shell)?;
        return Ok(());
    }

    // Set up logging based on verbosity
    let level = if args.quiet {
        "error"
    } else if args.verbose {
        "debug"
    } else {
        "info"
    };

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

    if args.doctor {
        info!("🩺 GhostWave System Diagnostics");
        rtx_acceleration::check_rtx_system_requirements()?;

        let detector = device_detection::DeviceDetector::new();
        let devices = detector.detect_devices().await?;

        info!("📱 Audio Devices Found: {}", devices.len());
        for device in &devices {
            info!("  • {} {} ({})", device.vendor, device.model, device.name);
            if device.is_xlr_interface {
                info!("    ✅ XLR Interface - Recommended profile: {}", device.recommended_profile);
            }
        }

        if let Some(scarlett) = detector.find_scarlett_solo_4th_gen().await? {
            info!("🎤 Scarlett Solo 4th Gen: FOUND");
            info!("   Optimal sample rates: {:?}", scarlett.sample_rates);
        } else {
            info!("🎤 Scarlett Solo 4th Gen: Not detected");
        }

        // Test RTX acceleration
        match rtx_acceleration::RtxAccelerator::new() {
            Ok(rtx) => {
                info!("🚀 RTX Acceleration: {}", rtx.get_processing_mode());
                if let Some(caps) = rtx.get_capabilities() {
                    info!("   Compute: {}.{}", caps.compute_capability.0, caps.compute_capability.1);
                    info!("   Memory: {:.1} GB", caps.memory_gb);
                    info!("   RTX Voice: {}", caps.supports_rtx_voice);
                }
            }
            Err(e) => info!("🚀 RTX Acceleration: Not available ({})", e),
        }

        info!("✅ Diagnostics complete");
        return Ok(());
    }

    if args.bench {
        info!("🏁 GhostWave Audio Performance Benchmark");
        info!("Testing real-time audio processing latency and throughput");

        // Auto-detect Scarlett Solo for optimal testing
        let detector = device_detection::DeviceDetector::new();
        let config = if let Some(scarlett) = detector.find_scarlett_solo_4th_gen().await? {
            info!("Using Scarlett Solo 4th Gen for benchmark");
            detector.get_optimal_config_for_device(&scarlett).await?
        } else {
            info!("Using default configuration for benchmark");
            config::Config::load(&args.profile)?
        };

        return run_audio_benchmark(config).await;
    }

    // Load configuration with CLI overrides
    let mut config = if let Some(config_path) = &args.config {
        config::Config::load_from_file(config_path)?
    } else {
        config::Config::load(&args.profile)?
    };

    // Apply CLI overrides
    if let Some(sample_rate) = args.samplerate {
        config.audio.sample_rate = sample_rate;
    }
    if let Some(frames) = args.frames {
        config.audio.buffer_size = frames;
    }
    if let Some(channels) = args.channels {
        config.audio.channels = channels;
    }
    if let Some(ref input) = args.input {
        config.audio.input_device = Some(input.clone());
    }
    if let Some(ref output) = args.output {
        config.audio.output_device = Some(output.clone());
    }
    if let Some(ref backend) = args.backend {
        config.audio.backend = backend.clone();
    }
    if let Some(latency) = args.latency {
        config.audio.target_latency_ms = latency;
    }

    // Dry-run validation
    if args.dry_run {
        info!("🧪 Dry-run validation mode");
        info!("Configuration: {}Hz, {} frames, {} channels",
              config.audio.sample_rate, config.audio.buffer_size, config.audio.channels);

        // Validate configuration
        if config.audio.buffer_size < 32 || config.audio.buffer_size > 4096 {
            return Err(anyhow::anyhow!("Buffer size must be between 32-4096 frames"));
        }

        if !config.audio.buffer_size.is_power_of_two() {
            return Err(anyhow::anyhow!("Buffer size must be power of 2"));
        }

        if ![44100, 48000, 96000, 192000].contains(&config.audio.sample_rate) {
            return Err(anyhow::anyhow!("Unsupported sample rate"));
        }

        info!("✅ Configuration validation passed");
        return Ok(());
    }

    // Backend selection
    if args.alsa {
        info!("🔊 Starting GhostWave with ALSA direct backend");
        return alsa_module::run_alsa_mode(config).await;
    }

    if args.jack {
        info!("🎶 Starting GhostWave with JACK professional backend");
        return jack_module::run_jack_mode(config).await;
    }

    info!("🎧 GhostWave starting - NVIDIA RTX Voice for Linux");
    info!("Profile: {}", args.profile);

    // Apply real-time optimizations if requested
    if args.realtime {
        let latency_config = LatencyConfig {
            target_latency_ms: config.audio.target_latency_ms,
            sample_rate: config.audio.sample_rate,
            buffer_size: config.audio.buffer_size as usize,
            buffer_count: 2,
            aggressive_mode: true,
            cpu_affinity: args.cpu_affinity.as_ref().map(|cores| {
                cores.split(',').filter_map(|s| s.trim().parse().ok()).collect()
            }),
        };

        let mut optimizer = LatencyOptimizer::new(latency_config);
        if let Err(e) = optimizer.optimize() {
            warn!("RT optimization failed: {}. Continuing with normal priority.", e);
        }
    }

    if args.ipc_server {
        info!("Starting IPC server for PhantomLink integration");
        ipc::run_ipc_server(config).await?
    } else if args.phantomlink {
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

fn generate_completions(shell: &str) -> Result<()> {
    use clap::CommandFactory;
    use clap_complete::{generate, Shell};
    use std::io;

    let mut app = Args::command();
    let shell_type = match shell.to_lowercase().as_str() {
        "bash" => Shell::Bash,
        "zsh" => Shell::Zsh,
        "fish" => Shell::Fish,
        "powershell" => Shell::PowerShell,
        _ => return Err(anyhow::anyhow!("Unsupported shell: {}", shell)),
    };

    generate(shell_type, &mut app, "ghostwave", &mut io::stdout());
    Ok(())
}

async fn run_audio_benchmark(config: config::Config) -> Result<()> {
    use std::time::{Duration, Instant};
    use low_latency::*;

    info!("🔬 Running comprehensive audio benchmark");
    info!("Configuration: {}Hz, {} frames, {} channels",
          config.audio.sample_rate, config.audio.buffer_size, config.audio.channels);

    // Set up real-time optimization
    RealTimeScheduler::optimize_thread_for_audio()?;

    let scheduler = RealTimeScheduler::new(config.audio.sample_rate, config.audio.buffer_size as usize);
    let benchmark = AudioBenchmark::new(config.audio.sample_rate, config.audio.buffer_size as usize);

    // Test different buffer sizes for latency analysis
    let test_buffer_sizes = vec![32, 64, 128, 256, 512, 1024];

    info!("📊 Testing optimal buffer sizes for {}ms target latency:", TARGET_LATENCY_MS);
    for &buffer_size in &test_buffer_sizes {
        let latency = scheduler.calculate_latency(buffer_size);
        let latency_ms = latency.as_secs_f64() * 1000.0;
        let status = if latency_ms <= TARGET_LATENCY_MS as f64 { "✅" } else { "⚠️" };
        info!("  {} frames: {:.2}ms {}", buffer_size, latency_ms, status);
    }

    // Recommended buffer size
    let optimal_buffer = RealTimeScheduler::get_optimal_buffer_size(
        config.audio.sample_rate, TARGET_LATENCY_MS
    );
    info!("🎯 Recommended buffer size: {} frames", optimal_buffer);

    // Test lock-free ring buffer performance
    info!("🔄 Testing lock-free ring buffer performance...");
    let ring_buffer = LockFreeAudioBuffer::new(4096, config.audio.sample_rate);
    let test_data = vec![0.5f32; 1024];
    let mut read_buffer = vec![0.0f32; 1024];

    let ring_buffer_start = Instant::now();
    for _ in 0..1000 {
        ring_buffer.write(&test_data)?;
        ring_buffer.read(&mut read_buffer)?;
    }
    let ring_buffer_time = ring_buffer_start.elapsed();
    info!("Ring buffer throughput: {:.2} MB/s",
          (1000 * 1024 * 4) as f64 / ring_buffer_time.as_secs_f64() / 1_000_000.0);

    // Test memory pool performance
    info!("💾 Testing memory pool performance...");
    let memory_pool = AudioMemoryPool::new(config.audio.buffer_size as usize, 16);

    let pool_start = Instant::now();
    for _ in 0..10000 {
        if let Some(buffer) = memory_pool.get_buffer() {
            memory_pool.return_buffer(buffer);
        }
    }
    let pool_time = pool_start.elapsed();
    info!("Memory pool allocation rate: {:.0} allocs/sec",
          10000.0 / pool_time.as_secs_f64());

    // Test real-time audio processing simulation
    info!("🎵 Simulating real-time audio processing for 5 seconds...");
    let mut processor = noise_suppression::NoiseProcessor::new(&config.noise_suppression)?;

    let _frame_duration = Duration::from_micros(
        (config.audio.buffer_size as f64 / config.audio.sample_rate as f64 * 1_000_000.0) as u64
    );

    let test_duration = Duration::from_secs(5);
    let start_time = Instant::now();
    let mut frame_count = 0u64;

    while start_time.elapsed() < test_duration {
        let frame_start = Instant::now();

        // Simulate audio input
        let input = vec![0.1f32; config.audio.buffer_size as usize];
        let mut output = vec![0.0f32; config.audio.buffer_size as usize];

        // Process audio
        processor.process(&input, &mut output)?;

        // Record performance
        let processing_time = frame_start.elapsed();
        benchmark.record_frame_processing(processing_time);

        frame_count += 1;

        // Sleep until next frame
        scheduler.sleep_until_next_frame(frame_start);
    }

    let total_time = start_time.elapsed();
    let expected_frames = (total_time.as_secs_f64() * config.audio.sample_rate as f64 / config.audio.buffer_size as f64) as u64;
    let frame_accuracy = (frame_count as f64 / expected_frames as f64) * 100.0;

    info!("🏆 Benchmark Results:");
    info!("  Duration: {:.2}s", total_time.as_secs_f64());
    info!("  Frames processed: {} (expected: {})", frame_count, expected_frames);
    info!("  Frame timing accuracy: {:.2}%", frame_accuracy);

    benchmark.report_stats();

    let stats = benchmark.get_stats();
    if stats.xrun_count == 0 {
        info!("🎉 Perfect real-time performance achieved!");
        info!("GhostWave is ready for professional audio use");
    } else {
        let xrun_rate = (stats.xrun_count as f64 / stats.total_frames as f64) * 100.0;
        if xrun_rate < 0.1 {
            info!("✅ Excellent performance - {:.3}% XRun rate", xrun_rate);
        } else if xrun_rate < 1.0 {
            info!("⚠️  Good performance - {:.3}% XRun rate", xrun_rate);
            info!("Consider optimizing system for better results");
        } else {
            warn!("❌ Poor performance - {:.3}% XRun rate", xrun_rate);
            warn!("Recommend system tuning or larger buffer sizes");
        }
    }

    info!("💡 Performance Tips:");
    info!("  • Use 'sudo setcap cap_sys_nice+ep ./ghostwave' for RT priority");
    info!("  • Disable CPU frequency scaling for consistent performance");
    info!("  • Close unnecessary applications during audio work");
    info!("  • Consider using a real-time kernel for best results");

    Ok(())
}
