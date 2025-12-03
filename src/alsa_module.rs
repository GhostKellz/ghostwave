use alsa::pcm::{Access, Format, HwParams, PCM, State};
use alsa::{Direction, ValueOr};
use anyhow::{Context, Result};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use tracing::{debug, error, info, warn};

use crate::config::Config;
use crate::low_latency::{AudioBenchmark, RealTimeScheduler};
use crate::noise_suppression::NoiseProcessor;

pub struct AlsaModule {
    config: Config,
    processor: Arc<Mutex<NoiseProcessor>>,
    pcm_in: Option<PCM>,
    pcm_out: Option<PCM>,
    scheduler: RealTimeScheduler,
    benchmark: Arc<Mutex<AudioBenchmark>>,
}

impl AlsaModule {
    pub fn new(config: Config) -> Result<Self> {
        info!("Initializing ALSA direct integration module");

        let processor = Arc::new(Mutex::new(NoiseProcessor::new(&config.noise_suppression)?));

        let scheduler =
            RealTimeScheduler::new(config.audio.sample_rate, config.audio.buffer_size as usize);
        let benchmark = Arc::new(Mutex::new(AudioBenchmark::new(
            config.audio.sample_rate,
            config.audio.buffer_size as usize,
        )));

        Ok(Self {
            config,
            processor,
            pcm_in: None,
            pcm_out: None,
            scheduler,
            benchmark,
        })
    }

    pub fn detect_alsa_devices(&self) -> Result<Vec<String>> {
        info!("🔍 Detecting ALSA devices...");

        let mut devices = Vec::new();

        // Check for common ALSA devices
        let device_candidates = vec![
            "default",
            "hw:0,0",
            "hw:1,0",
            "hw:2,0",
            "hw:3,0", // Scarlett Solo is often hw:3,0
            "hw:4,0",
            "plughw:0,0",
            "plughw:1,0",
            "plughw:2,0",
            "plughw:3,0",
            "plughw:4,0",
        ];

        for device_name in device_candidates {
            if let Ok(pcm) = PCM::new(device_name, Direction::Capture, false) {
                devices.push(device_name.to_string());
                debug!("Found ALSA device: {}", device_name);

                // Try to get device info
                if let Ok(info) = pcm.info() {
                    if let Ok(card_name) = info.get_name() {
                        debug!("  Card: {}", card_name);
                    }
                    if let Ok(device_name) = info.get_id() {
                        debug!("  Device: {}", device_name);
                    }
                }
            }
        }

        info!("Found {} ALSA devices", devices.len());
        Ok(devices)
    }

    pub fn open_devices(&mut self) -> Result<()> {
        info!("Opening ALSA audio devices");

        // Determine device names
        let input_device = self
            .config
            .audio
            .input_device
            .as_deref()
            .unwrap_or("default");
        let output_device = self
            .config
            .audio
            .output_device
            .as_deref()
            .unwrap_or("default");

        info!("Input device: {}", input_device);
        info!("Output device: {}", output_device);

        // Open input device
        self.pcm_in = Some(
            PCM::new(input_device, Direction::Capture, false)
                .with_context(|| format!("Failed to open ALSA input device: {}", input_device))?,
        );

        // Open output device
        self.pcm_out = Some(
            PCM::new(output_device, Direction::Playback, false)
                .with_context(|| format!("Failed to open ALSA output device: {}", output_device))?,
        );

        info!("✅ ALSA devices opened successfully");
        Ok(())
    }

    pub fn setup_hardware_params(&mut self) -> Result<()> {
        info!("Configuring ALSA hardware parameters");

        let sample_rate = self.config.audio.sample_rate;
        let channels = self.config.audio.channels;
        let buffer_size = self.config.audio.buffer_size;

        // Configure input device
        if let Some(ref pcm_in) = self.pcm_in {
            let hwp = HwParams::any(pcm_in)?;
            hwp.set_channels(channels.into())?;
            hwp.set_rate(sample_rate, ValueOr::Nearest)?;
            hwp.set_format(Format::float())?;
            hwp.set_access(Access::RWInterleaved)?;

            // Set buffer and period sizes for low latency
            let period_size = buffer_size / 4; // 4 periods per buffer
            hwp.set_buffer_size(buffer_size as i64)?;
            hwp.set_period_size(period_size as i64, ValueOr::Nearest)?;

            pcm_in.hw_params(&hwp)?;

            info!("Input device configured:");
            info!("  Sample rate: {}Hz", hwp.get_rate()?);
            info!("  Channels: {}", hwp.get_channels()?);
            info!("  Buffer size: {} frames", hwp.get_buffer_size()?);
            info!("  Period size: {} frames", hwp.get_period_size()?);
        }

        // Configure output device
        if let Some(ref pcm_out) = self.pcm_out {
            let hwp = HwParams::any(pcm_out)?;
            hwp.set_channels(channels.into())?;
            hwp.set_rate(sample_rate, ValueOr::Nearest)?;
            hwp.set_format(Format::float())?;
            hwp.set_access(Access::RWInterleaved)?;

            let period_size = buffer_size / 4;
            hwp.set_buffer_size(buffer_size as i64)?;
            hwp.set_period_size(period_size as i64, ValueOr::Nearest)?;

            pcm_out.hw_params(&hwp)?;

            info!("Output device configured:");
            info!("  Sample rate: {}Hz", hwp.get_rate()?);
            info!("  Channels: {}", hwp.get_channels()?);
            info!("  Buffer size: {} frames", hwp.get_buffer_size()?);
            info!("  Period size: {} frames", hwp.get_period_size()?);
        }

        info!("✅ ALSA hardware parameters configured");
        Ok(())
    }

    pub fn start_audio_processing(&mut self) -> Result<()> {
        info!("Starting ALSA audio processing");

        let pcm_in = self
            .pcm_in
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Input PCM not initialized"))?;
        let pcm_out = self
            .pcm_out
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Output PCM not initialized"))?;

        // Prepare devices
        pcm_in.prepare()?;
        pcm_out.prepare()?;

        info!("🎵 ALSA real-time audio processing started");
        info!("Processing mode: {}", self.get_processing_mode());

        let buffer_size = self.config.audio.buffer_size as usize;
        let channels = self.config.audio.channels as usize;
        let frame_size = buffer_size * channels;

        let mut input_buffer = vec![0.0f32; frame_size];
        let mut output_buffer = vec![0.0f32; frame_size];
        let mut frame_count = 0u64;

        loop {
            let frame_start = Instant::now();

            // Read from input
            match pcm_in.io_f32() {
                Ok(io) => match io.readi(&mut input_buffer) {
                    Ok(frames_read) => {
                        if frames_read != buffer_size as usize {
                            debug!("Partial read: {} frames", frames_read);
                        }
                    }
                    Err(e) => {
                        warn!("ALSA input error: {}", e);
                        pcm_in.try_recover(e, false)?;
                        continue;
                    }
                },
                Err(e) => {
                    error!("Failed to get input IO: {}", e);
                    break;
                }
            }

            // Process audio
            if let Ok(mut processor) = self.processor.lock() {
                if let Err(e) = processor.process(&input_buffer, &mut output_buffer) {
                    error!("Audio processing error: {}", e);
                }
            } else {
                // Fallback to passthrough
                output_buffer.copy_from_slice(&input_buffer);
            }

            // Write to output
            match pcm_out.io_f32() {
                Ok(io) => match io.writei(&output_buffer) {
                    Ok(frames_written) => {
                        if frames_written != buffer_size as usize {
                            debug!("Partial write: {} frames", frames_written);
                        }
                    }
                    Err(e) => {
                        warn!("ALSA output error: {}", e);
                        pcm_out.try_recover(e, false)?;
                        continue;
                    }
                },
                Err(e) => {
                    error!("Failed to get output IO: {}", e);
                    break;
                }
            }

            // Record performance
            let processing_time = frame_start.elapsed();
            if let Ok(benchmark) = self.benchmark.lock() {
                benchmark.record_frame_processing(processing_time);
            }

            frame_count += 1;

            // Report stats every 1000 frames
            if frame_count % 1000 == 0 {
                if let Ok(benchmark) = self.benchmark.lock() {
                    let stats = benchmark.get_stats();
                    debug!(
                        "ALSA: {} frames processed, {} XRuns",
                        stats.total_frames, stats.xrun_count
                    );
                }
            }

            // Sleep until next frame (if processing was fast enough)
            self.scheduler.sleep_until_next_frame(frame_start);
        }

        Ok(())
    }

    pub fn get_processing_mode(&self) -> String {
        if let Ok(processor) = self.processor.lock() {
            format!("ALSA Direct + {}", processor.get_processing_mode())
        } else {
            "ALSA Direct + Unknown".to_string()
        }
    }

    pub fn stop(&mut self) -> Result<()> {
        info!("Stopping ALSA audio processing");

        if let Some(ref pcm_in) = self.pcm_in {
            if pcm_in.state() == State::Running {
                pcm_in.drop()?;
            }
        }

        if let Some(ref pcm_out) = self.pcm_out {
            if pcm_out.state() == State::Running {
                pcm_out.drop()?;
            }
        }

        // Report final stats
        if let Ok(benchmark) = self.benchmark.lock() {
            benchmark.report_stats();
        }

        info!("✅ ALSA module stopped");
        Ok(())
    }
}

impl Drop for AlsaModule {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Check if ALSA is available on the system
pub fn check_alsa_availability() -> bool {
    // Try to open default device
    match PCM::new("default", Direction::Capture, false) {
        Ok(_) => {
            info!("✅ ALSA is available");
            true
        }
        Err(e) => {
            debug!("ALSA not available: {}", e);
            false
        }
    }
}

/// Auto-detect the best ALSA device for the given config
#[allow(dead_code)] // Public API for external use
pub fn auto_detect_alsa_device(config: &Config) -> Result<Option<String>> {
    let alsa_module = AlsaModule::new(config.clone())?;
    let devices = alsa_module.detect_alsa_devices()?;

    if devices.is_empty() {
        return Ok(None);
    }

    // Prefer hardware devices over plugin devices
    for device in &devices {
        if device.starts_with("hw:") {
            info!("Selected ALSA device: {}", device);
            return Ok(Some(device.clone()));
        }
    }

    // Fall back to first available device
    let device = devices[0].clone();
    info!("Selected ALSA device: {}", device);
    Ok(Some(device))
}

pub async fn run_alsa_mode(config: Config) -> Result<()> {
    info!("🔊 Starting GhostWave in ALSA direct mode");
    info!("This provides low-level hardware access for minimal latency");

    // Check ALSA availability
    if !check_alsa_availability() {
        return Err(anyhow::anyhow!("ALSA is not available on this system"));
    }

    // Set up real-time optimization
    RealTimeScheduler::optimize_thread_for_audio()?;

    // Create and configure ALSA module
    let mut alsa_module = AlsaModule::new(config)?;

    // Detect and open devices
    alsa_module.detect_alsa_devices()?;
    alsa_module
        .open_devices()
        .context("Failed to open ALSA devices")?;

    alsa_module
        .setup_hardware_params()
        .context("Failed to configure ALSA hardware parameters")?;

    info!("🎯 ALSA module ready - Direct hardware access active");

    // Start processing in a separate thread
    let _processing_handle = thread::spawn(move || {
        if let Err(e) = alsa_module.start_audio_processing() {
            error!("ALSA processing error: {}", e);
        }
    });

    // Wait for shutdown signal
    tokio::signal::ctrl_c().await?;
    info!("Received shutdown signal");

    // Note: In a real implementation, you'd send a stop signal to the processing thread
    info!("ALSA module shutdown complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alsa_availability() {
        let available = check_alsa_availability();
        println!("ALSA available: {}", available);
    }

    #[test]
    fn test_device_detection() {
        if check_alsa_availability() {
            let config = Config::load("balanced").unwrap();
            let alsa_module = AlsaModule::new(config).unwrap();
            let devices = alsa_module.detect_alsa_devices().unwrap();
            println!("Found ALSA devices: {:?}", devices);
            assert!(!devices.is_empty());
        }
    }
}
