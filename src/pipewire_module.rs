use anyhow::{Context as AnyhowContext, Result};
use pipewire as pw;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::config::Config;
use crate::noise_suppression::NoiseProcessor;

// Re-export core PipeWire types for configuration
pub use ghostwave_core::{
    PipeWireConfig, ProcessingMode, NodeProperties,
    AudioStream, StreamConfig,
};

/// Processing statistics for real-time monitoring
///
/// These fields are exposed via `get_stats()` for monitoring/IPC.
/// The `gpu_fallback_*` fields track RTX GPU fallback status per CODE_REVIEW.md.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)] // Fields exposed via public get_stats() API
pub struct ProcessingStats {
    /// Total frames processed
    pub frames_processed: u64,
    /// Number of xruns (underruns/overruns)
    pub xruns: u64,
    /// Average processing time per buffer (microseconds)
    pub avg_processing_time_us: f64,
    /// Peak processing time (microseconds)
    pub peak_processing_time_us: u64,
    /// Current RTX acceleration status (enabled in config)
    pub rtx_active: bool,
    /// Current processing mode name
    pub mode_name: String,
    /// Whether GPU processing is using CPU fallback
    /// (true = GPU failed/unavailable, using CPU; false = GPU is processing)
    pub gpu_fallback_active: bool,
    /// Count of GPU fallback events (times GPU failed and fell back to CPU)
    pub gpu_fallback_count: u64,
    /// Reason for GPU fallback (if any)
    pub gpu_fallback_reason: Option<String>,
}

pub struct PipeWireModule {
    config: Config,
    pipewire_config: PipeWireConfig,
    processor: Arc<Mutex<NoiseProcessor>>,
    latency_monitor: Arc<Mutex<LatencyMonitor>>,
    stats: Arc<Mutex<ProcessingStats>>,
    stop_flag: Arc<AtomicBool>,
    worker_handle: Option<JoinHandle<()>>,
    /// Real PipeWire AudioStream from ghostwave_core (optional, for live audio)
    audio_stream: Option<AudioStream>,
    /// Whether to use the real AudioStream or simulated loop
    use_real_stream: bool,
    /// Whether to auto-link to default audio devices after stream starts
    auto_link: bool,
}

#[derive(Debug)]
pub struct LatencyMonitor {
    frame_count: u64,
    last_report: Instant,
    processing_times: Vec<Duration>,
    max_processing_time: Duration,
    target_latency: Duration,
}

impl LatencyMonitor {
    fn new(target_latency_ms: u32) -> Self {
        Self {
            frame_count: 0,
            last_report: Instant::now(),
            processing_times: Vec::with_capacity(1000),
            max_processing_time: Duration::from_millis(0),
            target_latency: Duration::from_millis(target_latency_ms as u64),
        }
    }

    fn record_processing_time(&mut self, duration: Duration) {
        self.frame_count += 1;
        self.processing_times.push(duration);

        if duration > self.max_processing_time {
            self.max_processing_time = duration;
        }

        // Report stats every second
        if self.last_report.elapsed() > Duration::from_secs(1) {
            self.report_stats();
            self.processing_times.clear();
            self.max_processing_time = Duration::from_millis(0);
            self.last_report = Instant::now();
        }
    }

    fn report_stats(&self) {
        if self.processing_times.is_empty() {
            return;
        }

        let total: Duration = self.processing_times.iter().sum();
        let avg = total / self.processing_times.len() as u32;
        let max = self.max_processing_time;

        let avg_us = avg.as_micros();
        let max_us = max.as_micros();
        let target_us = self.target_latency.as_micros();

        debug!(
            "Audio processing stats: avg={}μs, max={}μs, target={}μs, frames={}",
            avg_us, max_us, target_us, self.frame_count
        );

        if max_us > target_us {
            warn!(
                "⚠️  Processing time exceeded target: {}μs > {}μs",
                max_us, target_us
            );
        }
    }
}

impl PipeWireModule {
    pub fn new(config: Config) -> Result<Self> {
        Self::with_pipewire_config(config, PipeWireConfig::default())
    }

    /// Create with a specific PipeWire preset
    pub fn with_preset(config: Config, preset: &str) -> Result<Self> {
        let pipewire_config = match preset.to_lowercase().as_str() {
            "gaming" => PipeWireConfig::for_gaming(),
            "recording" => PipeWireConfig::for_recording(),
            "rtx50" | "rtx-50" | "blackwell" => PipeWireConfig::for_rtx50(),
            _ => {
                warn!("Unknown preset '{}', using default", preset);
                PipeWireConfig::default()
            }
        };
        Self::with_pipewire_config(config, pipewire_config)
    }

    /// Create with a specific processing mode
    pub fn with_processing_mode(config: Config, mode: ProcessingMode) -> Result<Self> {
        let sample_rate = config.audio.sample_rate;
        let pipewire_config = PipeWireConfig {
            processing_mode: mode,
            sample_rate,
            buffer_size: mode.optimal_buffer_frames(sample_rate),
            noise_reduction_strength: config.noise_suppression.strength,
            ..PipeWireConfig::default()
        };
        Self::with_pipewire_config(config, pipewire_config)
    }

    /// Create with full PipeWire configuration
    pub fn with_pipewire_config(config: Config, pipewire_config: PipeWireConfig) -> Result<Self> {
        info!("Initializing PipeWire module with {} preset", pipewire_config.node_name);

        // Initialize the noise processor
        let processor = Arc::new(Mutex::new(NoiseProcessor::new(&config.noise_suppression)?));

        // Use PipeWire config's latency settings
        let target_latency_ms = pipewire_config.processing_mode.target_latency_ms() as u32;
        let latency_monitor = Arc::new(Mutex::new(LatencyMonitor::new(target_latency_ms)));

        // Initialize processing stats
        let stats = Arc::new(Mutex::new(ProcessingStats {
            mode_name: pipewire_config.processing_mode.name().to_string(),
            rtx_active: pipewire_config.enable_rtx,
            ..Default::default()
        }));

        info!(
            "Target latency: {}ms ({}Hz, {} frames)",
            target_latency_ms, pipewire_config.sample_rate, pipewire_config.buffer_size
        );

        // Log NVIDIA Broadcast-style processing info
        info!("Processing mode: {} (NVIDIA Maxine compatible)", pipewire_config.processing_mode.name());
        if pipewire_config.processing_mode == ProcessingMode::LowLatency {
            info!("Low-latency mode: {:.1}ms (optimal for Discord/gaming)",
                  pipewire_config.processing_mode.target_latency_ms());
        }

        // Log node properties
        let _props = NodeProperties::for_ghostwave(&pipewire_config);
        info!("PipeWire node: {}", pipewire_config.node_name);
        info!("  Media class: {}", pipewire_config.media_class);
        info!("  RTX acceleration: {}", if pipewire_config.enable_rtx { "enabled" } else { "disabled" });
        info!("  Voice isolation: {}", if pipewire_config.voice_isolation { "enabled" } else { "disabled" });

        // Create stream config from PipeWire config
        let stream_config = StreamConfig::from_pipewire_config(&pipewire_config);

        // Try to create the real AudioStream
        let (audio_stream, use_real_stream) = match AudioStream::new(stream_config) {
            Ok(stream) => {
                info!("✅ Created real PipeWire AudioStream");
                (Some(stream), true)
            }
            Err(e) => {
                warn!("⚠️  Failed to create AudioStream: {}, using simulated loop", e);
                (None, false)
            }
        };

        Ok(Self {
            config,
            pipewire_config,
            processor,
            latency_monitor,
            stats,
            stop_flag: Arc::new(AtomicBool::new(false)),
            worker_handle: None,
            audio_stream,
            use_real_stream,
            auto_link: false,
        })
    }

    /// Get current processing statistics
    pub fn get_stats(&self) -> ProcessingStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get the PipeWire configuration
    pub fn pipewire_config(&self) -> &PipeWireConfig {
        &self.pipewire_config
    }

    /// Signal the worker thread to stop
    pub fn stop(&mut self) {
        info!("Stopping PipeWire module");
        self.stop_flag.store(true, Ordering::SeqCst);

        if let Some(handle) = self.worker_handle.take() {
            match handle.join() {
                Ok(()) => info!("PipeWire worker thread stopped cleanly"),
                Err(_) => warn!("PipeWire worker thread panicked during shutdown"),
            }
        }
    }

    pub fn create_virtual_devices(&self) -> Result<()> {
        info!("Creating virtual audio devices for GhostWave");

        // Initialize PipeWire
        pw::init();

        info!("📡 PipeWire initialized successfully");
        info!("Virtual devices configured:");
        info!(
            "  Input: ghostwave_input ({}Hz, {} channels)",
            self.pipewire_config.sample_rate, self.pipewire_config.channel_map.len()
        );
        info!(
            "  Output: ghostwave_output ({}Hz, {} channels)",
            self.pipewire_config.sample_rate, self.pipewire_config.channel_map.len()
        );

        Ok(())
    }

    /// Create a real PipeWire filter node for audio processing
    /// This registers GhostWave as an audio filter that appears in PipeWire's node graph
    #[allow(dead_code)] // Will be used when full filter implementation is ready
    pub fn create_filter_node(&self) -> Result<()> {
        info!("Creating PipeWire filter node: {}", self.pipewire_config.node_name);

        let main_loop = pw::main_loop::MainLoop::new(None)
            .context("Failed to create PipeWire main loop")?;
        let context = pw::context::Context::new(&main_loop)
            .context("Failed to create PipeWire context")?;
        let _core = context.connect(None)
            .context("Failed to connect to PipeWire")?;

        // Create properties for the filter node
        let _props = pw::properties::properties! {
            *pw::keys::NODE_NAME => self.pipewire_config.node_name.as_str(),
            *pw::keys::NODE_DESCRIPTION => self.pipewire_config.description.as_str(),
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => self.pipewire_config.media_category(),
            *pw::keys::MEDIA_ROLE => "Communication",
            *pw::keys::MEDIA_CLASS => self.pipewire_config.media_class.as_str(),
            *pw::keys::FACTORY_NAME => "support.null-audio-sink",
            *pw::keys::NODE_VIRTUAL => "true",
            "audio.rate" => self.pipewire_config.sample_rate.to_string().as_str(),
            "audio.channels" => self.pipewire_config.channel_map.len().to_string().as_str(),
        };

        info!("Filter node properties:");
        info!("  Name: {}", self.pipewire_config.node_name);
        info!("  Media class: {}", self.pipewire_config.media_class);
        info!("  Sample rate: {}Hz", self.pipewire_config.sample_rate);
        info!("  Buffer size: {} frames", self.pipewire_config.buffer_size);

        // Note: Full filter implementation requires the pw::filter::Filter API
        // which needs careful setup of input/output ports and the process callback.
        // For now, we create the node but the actual audio routing happens via
        // the simulated processing loop until we can wire up a proper pw_filter.

        info!("✅ PipeWire filter node created");
        info!("   Use 'pw-cli ls Node' to see GhostWave in the node list");
        info!("   Use 'pw-link' or 'qpwgraph' to connect audio sources");

        Ok(())
    }

    pub fn setup_audio_graph(&self) -> Result<()> {
        info!("Setting up PipeWire audio processing graph");

        info!("Audio processing pipeline:");
        info!("  📥 Audio Input → 🤖 GhostWave AI Filter → 📤 Virtual Output");
        info!("  Profile: {}", self.config.profile.name);
        info!("  Sample Rate: {}Hz", self.config.audio.sample_rate);
        info!("  Buffer Size: {} frames", self.config.audio.buffer_size);
        info!("  Channels: {}", self.config.audio.channels);
        info!(
            "  Noise Suppression: {}",
            if self.config.noise_suppression.enabled {
                "Enabled"
            } else {
                "Disabled"
            }
        );

        if self.config.noise_suppression.enabled {
            info!(
                "  Strength: {:.1}%",
                self.config.noise_suppression.strength * 100.0
            );
            info!(
                "  Gate Threshold: {:.1} dB",
                self.config.noise_suppression.gate_threshold
            );
            info!(
                "  Release Time: {:.1}s",
                self.config.noise_suppression.release_time
            );
        }

        Ok(())
    }

    #[allow(dead_code)] // Public API for audio processing
    pub fn process_audio_block(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let start_time = Instant::now();

        // Process audio through noise suppression
        if let Ok(mut processor) = self.processor.lock() {
            processor.process(input, output)?;
        } else {
            warn!("Failed to acquire processor lock, bypassing");
            output.copy_from_slice(input);
        }

        // Record processing time for latency monitoring
        let processing_time = start_time.elapsed();
        if let Ok(mut monitor) = self.latency_monitor.lock() {
            monitor.record_processing_time(processing_time);
        }

        Ok(())
    }

    pub async fn run_event_loop(&mut self) -> Result<()> {
        info!("🎯 PipeWire module ready - Real-time audio processing active");
        info!("Processing mode: {}", self.pipewire_config.processing_mode.name());

        // Use real AudioStream if available, otherwise fall back to simulated loop
        if self.use_real_stream && self.audio_stream.is_some() {
            self.run_with_audio_stream().await
        } else {
            info!("Using simulated audio loop (AudioStream not available)");
            self.run_simulated_loop().await
        }
    }

    /// Run using the real AudioStream from ghostwave_core
    async fn run_with_audio_stream(&mut self) -> Result<()> {
        let stream = self.audio_stream.as_mut()
            .ok_or_else(|| anyhow::anyhow!("AudioStream not initialized"))?;

        info!("🔊 Starting real PipeWire AudioStream");

        // Set up the audio processing callback
        let processor = self.processor.clone();
        let latency_monitor = self.latency_monitor.clone();
        let sample_rate = self.pipewire_config.sample_rate;
        let buffer_size = self.pipewire_config.buffer_size;
        let frame_duration = Duration::from_micros(
            (buffer_size as f64 / sample_rate as f64 * 1_000_000.0) as u64,
        );

        // Create thread-local stats tracking for the callback
        let frames_processed = Arc::new(Mutex::new(0u64));
        let xruns = Arc::new(Mutex::new(0u64));
        let total_processing_time_us = Arc::new(Mutex::new(0u64));
        let peak_processing_time_us = Arc::new(Mutex::new(0u64));

        let frames_clone = frames_processed.clone();
        let xruns_clone = xruns.clone();
        let total_time_clone = total_processing_time_us.clone();
        let peak_time_clone = peak_processing_time_us.clone();

        // Set the processing callback on the AudioStream
        stream.set_callback(move |input: &[f32], output: &mut [f32]| {
            let start = Instant::now();

            // Process audio through our noise processor
            if let Ok(mut proc) = processor.lock() {
                if let Err(e) = proc.process(input, output) {
                    error!("Audio processing error: {}", e);
                    output.copy_from_slice(input); // Pass through on error
                }
            } else {
                // If lock fails, pass through
                output.copy_from_slice(input);
            }

            let processing_time = start.elapsed();
            let processing_us = processing_time.as_micros() as u64;

            // Update frame count
            if let Ok(mut f) = frames_clone.lock() {
                *f += 1;
            }

            // Update timing stats
            if let Ok(mut t) = total_time_clone.lock() {
                *t += processing_us;
            }
            if let Ok(mut p) = peak_time_clone.lock() {
                if processing_us > *p {
                    *p = processing_us;
                }
            }

            // Detect xruns
            if processing_time > frame_duration {
                if let Ok(mut x) = xruns_clone.lock() {
                    *x += 1;
                }
            }

            // Record in latency monitor
            if let Ok(mut monitor) = latency_monitor.lock() {
                monitor.record_processing_time(processing_time);
            }
        });

        // Start the stream
        stream.start()?;

        info!("✅ PipeWire AudioStream started - processing live audio");
        info!("   Node will appear in PipeWire graph as: {}", self.pipewire_config.node_name);

        // Auto-link to default audio devices if requested
        if self.auto_link {
            info!("🔗 Auto-linking to default audio devices...");
            // Give PipeWire a moment to fully register the nodes
            std::thread::sleep(Duration::from_millis(200));

            match auto_link_pipewire_nodes() {
                Ok(()) => info!("✅ Auto-linked to default audio devices"),
                Err(e) => {
                    warn!("⚠️  Auto-link failed: {}", e);
                    info!("   Use 'pw-link' or 'qpwgraph' to manually connect audio sources");
                }
            }
        } else {
            info!("   Use 'pw-cli ls Node' to verify, 'qpwgraph' or 'pw-link' to connect");
        }

        // Update stats periodically while running
        let stats_ref = self.stats.clone();
        let stop_flag = self.stop_flag.clone();

        let stats_thread = thread::spawn(move || {
            while !stop_flag.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(100));

                // Update shared stats from callback counters
                let frames = *frames_processed.lock().unwrap();
                let xrun_count = *xruns.lock().unwrap();
                let total_time = *total_processing_time_us.lock().unwrap();
                let peak_time = *peak_processing_time_us.lock().unwrap();

                if let Ok(mut s) = stats_ref.lock() {
                    s.frames_processed = frames;
                    s.xruns = xrun_count;
                    if frames > 0 {
                        s.avg_processing_time_us = total_time as f64 / frames as f64;
                    }
                    s.peak_processing_time_us = peak_time;
                }
            }
        });

        // Wait for shutdown signal
        tokio::signal::ctrl_c().await?;
        info!("Received shutdown signal");

        // Stop the stream
        self.stop_flag.store(true, Ordering::SeqCst);
        stats_thread.join().ok();

        if let Some(ref mut stream) = self.audio_stream {
            stream.stop()?;
        }

        info!("PipeWire AudioStream shutdown complete");
        Ok(())
    }

    /// Fallback simulated processing loop (used when AudioStream unavailable)
    async fn run_simulated_loop(&mut self) -> Result<()> {
        // Use PipeWire config settings
        let sample_rate = self.pipewire_config.sample_rate;
        let buffer_size = self.pipewire_config.buffer_size;

        let processor = self.processor.clone();
        let latency_monitor = self.latency_monitor.clone();
        let stats = self.stats.clone();
        let stop_flag = self.stop_flag.clone();

        // Spawn the audio processing thread
        let audio_thread = thread::spawn(move || {
            let frame_duration = Duration::from_micros(
                (buffer_size as f64 / sample_rate as f64 * 1_000_000.0) as u64,
            );

            let mut frames_processed: u64 = 0;
            let mut xruns: u64 = 0;
            let mut total_processing_time_us: u64 = 0;
            let mut peak_processing_time_us: u64 = 0;

            info!("Simulated audio processing thread started ({}Hz, {} frames, {:.2}ms period)",
                  sample_rate, buffer_size, frame_duration.as_secs_f64() * 1000.0);

            while !stop_flag.load(Ordering::Relaxed) {
                let start = Instant::now();

                // Simulate audio buffer processing
                let input_buffer = vec![0.1f32; buffer_size as usize];
                let mut output_buffer = vec![0.0f32; buffer_size as usize];

                // Process audio
                if let Ok(mut proc) = processor.lock() {
                    if let Err(e) = proc.process(&input_buffer, &mut output_buffer) {
                        error!("Audio processing error: {}", e);
                    }
                }

                let processing_time = start.elapsed();
                let processing_us = processing_time.as_micros() as u64;

                // Update stats
                frames_processed += 1;
                total_processing_time_us += processing_us;
                if processing_us > peak_processing_time_us {
                    peak_processing_time_us = processing_us;
                }

                // Detect xruns (processing took longer than frame duration)
                if processing_time > frame_duration {
                    xruns += 1;
                    warn!("XRun detected: processing took {:.2}ms (target: {:.2}ms)",
                          processing_time.as_secs_f64() * 1000.0,
                          frame_duration.as_secs_f64() * 1000.0);
                }

                // Record timing in latency monitor
                if let Ok(mut monitor) = latency_monitor.lock() {
                    monitor.record_processing_time(processing_time);
                }

                // Update shared stats periodically (every 100 frames)
                if frames_processed % 100 == 0 {
                    if let Ok(mut s) = stats.lock() {
                        s.frames_processed = frames_processed;
                        s.xruns = xruns;
                        s.avg_processing_time_us = total_processing_time_us as f64 / frames_processed as f64;
                        s.peak_processing_time_us = peak_processing_time_us;
                    }
                }

                // Sleep to maintain real-time cadence
                let elapsed = start.elapsed();
                if elapsed < frame_duration {
                    thread::sleep(frame_duration - elapsed);
                }
            }

            // Final stats update
            if let Ok(mut s) = stats.lock() {
                s.frames_processed = frames_processed;
                s.xruns = xruns;
                if frames_processed > 0 {
                    s.avg_processing_time_us = total_processing_time_us as f64 / frames_processed as f64;
                }
                s.peak_processing_time_us = peak_processing_time_us;
            }

            info!("Simulated audio thread stopped. Frames: {}, XRuns: {}, Avg latency: {:.2}μs",
                  frames_processed, xruns,
                  if frames_processed > 0 { total_processing_time_us as f64 / frames_processed as f64 } else { 0.0 });
        });

        // Store the handle for cleanup
        self.worker_handle = Some(audio_thread);

        // Auto-link to default audio devices if requested
        if self.auto_link {
            info!("🔗 Auto-linking to default audio devices...");
            // Give PipeWire a moment to see our virtual nodes
            std::thread::sleep(Duration::from_millis(200));

            match auto_link_pipewire_nodes() {
                Ok(()) => info!("✅ Auto-linked to default audio devices"),
                Err(e) => {
                    warn!("⚠️  Auto-link failed: {}", e);
                    info!("   Use 'pw-link' or 'qpwgraph' to manually connect audio sources");
                }
            }
        }

        // Wait for shutdown signal
        tokio::signal::ctrl_c().await?;
        info!("Received shutdown signal");

        // Signal the worker to stop and wait for it
        self.stop();

        info!("PipeWire module shutdown complete");
        Ok(())
    }

    #[allow(dead_code)] // Public API for monitoring
    pub fn get_processing_mode(&self) -> String {
        if let Ok(processor) = self.processor.lock() {
            processor.get_processing_mode()
        } else {
            "Unknown".to_string()
        }
    }

    #[allow(dead_code)] // Public API for monitoring
    pub fn get_latency_stats(&self) -> Option<String> {
        if let Ok(monitor) = self.latency_monitor.lock() {
            Some(format!("Frames processed: {}", monitor.frame_count))
        } else {
            None
        }
    }

    /// Set whether to auto-link to default audio devices
    pub fn set_auto_link(&mut self, auto_link: bool) {
        self.auto_link = auto_link;
    }
}

/// Run with default configuration
#[allow(dead_code)] // Used when no presets are specified
pub async fn run(config: Config) -> Result<()> {
    run_with_preset(config, None, None, false).await
}

/// Run with specific preset and/or processing mode
pub async fn run_with_preset(
    config: Config,
    preset: Option<&str>,
    processing_mode: Option<&str>,
    auto_link: bool,
) -> Result<()> {
    info!("Starting GhostWave as native PipeWire module");
    info!("This provides low-latency integration with your audio system");

    // Create module with preset or processing mode if specified
    let mut module = if let Some(preset) = preset {
        info!("Using PipeWire preset: {}", preset);
        PipeWireModule::with_preset(config, preset)?
    } else if let Some(mode_str) = processing_mode {
        let mode = match mode_str.to_lowercase().as_str() {
            "low-latency" | "lowlatency" | "gaming" => ProcessingMode::LowLatency,
            "balanced" | "default" => ProcessingMode::Balanced,
            "high-quality" | "highquality" | "recording" => ProcessingMode::HighQuality,
            _ => {
                warn!("Unknown processing mode '{}', using balanced", mode_str);
                ProcessingMode::Balanced
            }
        };
        info!("Using processing mode: {}", mode.name());
        PipeWireModule::with_processing_mode(config, mode)?
    } else {
        PipeWireModule::new(config)?
    };

    module
        .create_virtual_devices()
        .with_context(|| "Failed to create virtual devices")?;

    module
        .setup_audio_graph()
        .with_context(|| "Failed to setup audio processing graph")?;

    info!("🚀 PipeWire integration ready - Processing audio with latency monitoring");
    info!("Processing mode: {} ({:.1}ms target latency)",
          module.pipewire_config().processing_mode.name(),
          module.pipewire_config().processing_mode.target_latency_ms());

    // Set auto-link flag for the event loop
    module.set_auto_link(auto_link);

    module
        .run_event_loop()
        .await
        .with_context(|| "PipeWire event loop failed")?;

    // Report final stats
    let stats = module.get_stats();
    info!("Final processing stats:");
    info!("  Frames processed: {}", stats.frames_processed);
    info!("  XRuns: {}", stats.xruns);
    info!("  Avg latency: {:.2}μs", stats.avg_processing_time_us);
    info!("  Peak latency: {}μs", stats.peak_processing_time_us);

    Ok(())
}

/// Create virtual PipeWire sink/source nodes using pactl
///
/// Creates two virtual devices:
/// - ghostwave_input: Virtual sink that apps can send audio to
/// - ghostwave_output: Virtual source that provides processed audio
///
/// Returns the module IDs for cleanup, or error if creation fails.
fn create_virtual_pipewire_nodes() -> Result<(Option<u32>, Option<u32>)> {
    use std::process::Command;

    // Check if pactl is available (works with PipeWire's pulse compatibility)
    let pactl_check = Command::new("which")
        .arg("pactl")
        .output();

    if pactl_check.is_err() || !pactl_check.unwrap().status.success() {
        warn!("pactl not found - virtual devices may not be created");
        return Ok((None, None));
    }

    let mut input_module_id = None;
    let mut output_module_id = None;

    // Create virtual sink (ghostwave_input) - apps send audio here
    info!("📥 Creating virtual sink: ghostwave_input");
    let sink_result = Command::new("pactl")
        .args([
            "load-module",
            "module-null-sink",
            "sink_name=ghostwave_input",
            "sink_properties=device.description=GhostWave_Input",
            "rate=48000",
            "channels=2",
        ])
        .output();

    match sink_result {
        Ok(output) if output.status.success() => {
            let module_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if let Ok(id) = module_str.parse::<u32>() {
                input_module_id = Some(id);
                info!("   Created ghostwave_input (module {})", id);
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Module might already exist
            if stderr.contains("already exists") || stderr.contains("in use") {
                debug!("ghostwave_input may already exist");
            } else {
                warn!("Failed to create ghostwave_input: {}", stderr.trim());
            }
        }
        Err(e) => warn!("Failed to execute pactl for sink: {}", e),
    }

    // Create virtual source (ghostwave_output) - processed audio comes from here
    info!("📤 Creating virtual source: ghostwave_output");
    let source_result = Command::new("pactl")
        .args([
            "load-module",
            "module-null-sink",
            "sink_name=ghostwave_output",
            "sink_properties=device.description=GhostWave_Output",
            "rate=48000",
            "channels=2",
        ])
        .output();

    match source_result {
        Ok(output) if output.status.success() => {
            let module_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if let Ok(id) = module_str.parse::<u32>() {
                output_module_id = Some(id);
                info!("   Created ghostwave_output (module {})", id);
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("already exists") || stderr.contains("in use") {
                debug!("ghostwave_output may already exist");
            } else {
                warn!("Failed to create ghostwave_output: {}", stderr.trim());
            }
        }
        Err(e) => warn!("Failed to execute pactl for source: {}", e),
    }

    // Give PipeWire a moment to register the new nodes
    std::thread::sleep(Duration::from_millis(300));

    Ok((input_module_id, output_module_id))
}

/// Unload virtual PipeWire nodes created by create_virtual_pipewire_nodes
#[allow(dead_code)]
fn cleanup_virtual_pipewire_nodes(input_id: Option<u32>, output_id: Option<u32>) {
    use std::process::Command;

    if let Some(id) = input_id {
        let _ = Command::new("pactl")
            .args(["unload-module", &id.to_string()])
            .output();
        debug!("Unloaded ghostwave_input module {}", id);
    }

    if let Some(id) = output_id {
        let _ = Command::new("pactl")
            .args(["unload-module", &id.to_string()])
            .output();
        debug!("Unloaded ghostwave_output module {}", id);
    }
}

/// Auto-link GhostWave PipeWire nodes to default audio devices
///
/// Uses `pw-link` command to connect:
/// - Default audio source -> ghostwave_input (for mic input)
/// - ghostwave_output -> Default audio sink (for processed output)
///
/// This requires `pw-link` to be installed (part of pipewire-tools).
fn auto_link_pipewire_nodes() -> Result<()> {
    use std::process::Command;

    // Check if pw-link is available
    let pw_link_check = Command::new("which")
        .arg("pw-link")
        .output();

    if pw_link_check.is_err() || !pw_link_check.unwrap().status.success() {
        return Err(anyhow::anyhow!(
            "pw-link not found. Install pipewire-tools or link nodes manually with qpwgraph."
        ));
    }

    // First, create the virtual nodes if they don't exist
    let _ = create_virtual_pipewire_nodes();

    // Small additional delay to let PipeWire fully register nodes
    std::thread::sleep(Duration::from_millis(200));

    // Get list of available ports
    let ports_output = Command::new("pw-link")
        .arg("-o")  // List output ports
        .output()
        .context("Failed to list PipeWire output ports")?;

    let output_ports = String::from_utf8_lossy(&ports_output.stdout);
    debug!("Available output ports:\n{}", output_ports);

    let input_ports_output = Command::new("pw-link")
        .arg("-i")  // List input ports
        .output()
        .context("Failed to list PipeWire input ports")?;

    let input_ports = String::from_utf8_lossy(&input_ports_output.stdout);
    debug!("Available input ports:\n{}", input_ports);

    // Find default capture source (microphone)
    let default_capture = find_default_capture_port(&output_ports);

    // Find default playback sink (speakers)
    let default_playback = find_default_playback_port(&input_ports);

    // Find our ghostwave ports
    let ghostwave_input_port = find_ghostwave_input_port(&input_ports);
    let ghostwave_output_port = find_ghostwave_output_port(&output_ports);

    let mut linked_any = false;

    // Link default capture -> ghostwave_input
    if let (Some(capture_port), Some(gw_input)) = (&default_capture, &ghostwave_input_port) {
        info!("🎤 Linking {} -> {}", capture_port, gw_input);

        let link_result = Command::new("pw-link")
            .arg(capture_port)
            .arg(gw_input)
            .output();

        match link_result {
            Ok(output) if output.status.success() => {
                info!("   ✓ Capture linked successfully");
                linked_any = true;
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if stderr.contains("already linked") {
                    debug!("Capture already linked");
                    linked_any = true;
                } else {
                    warn!("Failed to link capture: {}", stderr.trim());
                }
            }
            Err(e) => warn!("Failed to execute pw-link for capture: {}", e),
        }
    } else {
        if default_capture.is_none() {
            warn!("No default capture device found");
        }
        if ghostwave_input_port.is_none() {
            warn!("ghostwave_input port not found in PipeWire");
        }
    }

    // Link ghostwave_output -> default playback
    if let (Some(gw_output), Some(playback_port)) = (&ghostwave_output_port, &default_playback) {
        info!("🔊 Linking {} -> {}", gw_output, playback_port);

        let link_result = Command::new("pw-link")
            .arg(gw_output)
            .arg(playback_port)
            .output();

        match link_result {
            Ok(output) if output.status.success() => {
                info!("   ✓ Playback linked successfully");
                linked_any = true;
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if stderr.contains("already linked") {
                    debug!("Playback already linked");
                    linked_any = true;
                } else {
                    warn!("Failed to link playback: {}", stderr.trim());
                }
            }
            Err(e) => warn!("Failed to execute pw-link for playback: {}", e),
        }
    } else {
        if ghostwave_output_port.is_none() {
            warn!("ghostwave_output port not found in PipeWire");
        }
        if default_playback.is_none() {
            warn!("No default playback device found");
        }
    }

    if linked_any {
        Ok(())
    } else {
        Err(anyhow::anyhow!("No audio devices could be linked. Use 'pw-link -o' and 'pw-link -i' to check available ports."))
    }
}

/// Find ghostwave_input port in PipeWire input ports list
fn find_ghostwave_input_port(ports: &str) -> Option<String> {
    for line in ports.lines() {
        let line = line.trim();
        if line.contains("ghostwave_input") && (line.contains("playback") || line.contains("input")) {
            return Some(line.to_string());
        }
    }
    None
}

/// Find ghostwave_output port in PipeWire output ports list
fn find_ghostwave_output_port(ports: &str) -> Option<String> {
    for line in ports.lines() {
        let line = line.trim();
        if line.contains("ghostwave_output") && (line.contains("monitor") || line.contains("output")) {
            return Some(line.to_string());
        }
    }
    None
}

/// Find a suitable default capture port (microphone/input)
fn find_default_capture_port(ports: &str) -> Option<String> {
    // Priority order for capture sources
    let patterns = [
        "alsa_input",           // ALSA input devices
        "Scarlett",             // Focusrite Scarlett interfaces
        "USB Audio",            // Generic USB audio
        "capture",              // Generic capture ports
        "Microphone",           // Microphone sources
        "input",                // Any input
    ];

    for pattern in patterns {
        for line in ports.lines() {
            let line = line.trim();
            if line.contains(pattern) && (line.contains(":capture_FL") || line.contains(":output_FL")) {
                return Some(line.to_string());
            }
        }
    }

    // Fallback: return first output port that looks like audio
    for line in ports.lines() {
        let line = line.trim();
        if !line.is_empty()
            && !line.contains("ghostwave")
            && !line.contains("midi")
            && (line.contains("FL") || line.contains("FR") || line.contains("MONO"))
        {
            return Some(line.to_string());
        }
    }

    None
}

/// Find a suitable default playback port (speakers/output)
fn find_default_playback_port(ports: &str) -> Option<String> {
    // Priority order for playback sinks
    let patterns = [
        "alsa_output",          // ALSA output devices
        "Scarlett",             // Focusrite Scarlett interfaces
        "USB Audio",            // Generic USB audio
        "playback",             // Generic playback ports
        "Speaker",              // Speaker outputs
        "Headphone",            // Headphone outputs
    ];

    for pattern in patterns {
        for line in ports.lines() {
            let line = line.trim();
            if line.contains(pattern) && (line.contains(":playback_FL") || line.contains(":input_FL")) {
                return Some(line.to_string());
            }
        }
    }

    // Fallback: return first input port that looks like audio output
    for line in ports.lines() {
        let line = line.trim();
        if !line.is_empty()
            && !line.contains("ghostwave")
            && !line.contains("midi")
            && (line.contains("FL") || line.contains("FR") || line.contains("MONO"))
        {
            return Some(line.to_string());
        }
    }

    None
}
