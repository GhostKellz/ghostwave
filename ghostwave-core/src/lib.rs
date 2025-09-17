//! # GhostWave Core
//!
//! Core audio processing library for GhostWave - Linux RTX Voice alternative.
//!
//! This library provides:
//! - Real-time noise suppression and audio processing
//! - Multiple audio backend support (PipeWire, ALSA, JACK, CPAL)
//! - NVIDIA RTX GPU acceleration for noise reduction
//! - Low-latency audio pipeline optimizations
//! - Lock-free data structures for zero-copy processing
//!
//! ## Example Usage
//!
//! ```rust
//! use ghostwave_core::{GhostWaveProcessor, Config, NoiseSuppressionConfig};
//!
//! // Create configuration
//! let config = Config::default()
//!     .with_sample_rate(48000)
//!     .with_buffer_size(256)
//!     .with_noise_suppression(NoiseSuppressionConfig::default());
//!
//! // Create processor
//! let mut processor = GhostWaveProcessor::new(config)?;
//!
//! // Process audio
//! let input = vec![0.1f32; 256];
//! let mut output = vec![0.0f32; 256];
//! processor.process(&input, &mut output)?;
//! ```

pub mod config;
pub mod noise_suppression;
pub mod low_latency;
pub mod device_detection;

#[cfg(feature = "nvidia-rtx")]
pub mod rtx_acceleration;

#[cfg(feature = "pipewire-backend")]
pub mod pipewire;

#[cfg(feature = "alsa-backend")]
pub mod alsa;

#[cfg(feature = "jack-backend")]
pub mod jack;

#[cfg(feature = "cpal-backend")]
pub mod cpal_backend;

use anyhow::Result;
use std::sync::{Arc, Mutex};
use tracing::{info, debug};

pub use config::{Config, AudioConfig, NoiseSuppressionConfig};
pub use noise_suppression::NoiseProcessor;
pub use low_latency::{LockFreeAudioBuffer, RealTimeScheduler, AudioBenchmark, TARGET_LATENCY_MS};
pub use device_detection::{DeviceDetector, AudioDevice, AudioDeviceType};

#[cfg(feature = "nvidia-rtx")]
pub use rtx_acceleration::{RtxAccelerator, RtxCapabilities};

/// Main GhostWave audio processor
pub struct GhostWaveProcessor {
    config: Config,
    noise_processor: Arc<Mutex<NoiseProcessor>>,

    #[cfg(feature = "nvidia-rtx")]
    rtx_accelerator: Option<RtxAccelerator>,
}

impl GhostWaveProcessor {
    /// Create a new GhostWave processor with the given configuration
    pub fn new(config: Config) -> Result<Self> {
        info!("Initializing GhostWave processor");
        debug!("Config: {}Hz, {} frames, {} channels",
               config.audio.sample_rate, config.audio.buffer_size, config.audio.channels);

        let noise_processor = Arc::new(Mutex::new(
            NoiseProcessor::new(&config.noise_suppression)?
        ));

        #[cfg(feature = "nvidia-rtx")]
        let rtx_accelerator = match RtxAccelerator::new() {
            Ok(accelerator) => {
                if accelerator.is_rtx_available() {
                    info!("✅ RTX acceleration enabled");
                    Some(accelerator)
                } else {
                    info!("💻 RTX not available, using CPU processing");
                    None
                }
            }
            Err(e) => {
                info!("⚠️  RTX initialization failed: {}, using CPU", e);
                None
            }
        };

        #[cfg(not(feature = "nvidia-rtx"))]
        info!("💻 RTX feature not compiled, using CPU processing");

        Ok(Self {
            config,
            noise_processor,
            #[cfg(feature = "nvidia-rtx")]
            rtx_accelerator,
        })
    }

    /// Process audio input through noise suppression
    pub fn process(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(anyhow::anyhow!("Input and output buffer size mismatch"));
        }

        if !self.config.noise_suppression.enabled {
            output.copy_from_slice(input);
            return Ok(());
        }

        // Try RTX acceleration first
        #[cfg(feature = "nvidia-rtx")]
        if let Some(ref rtx) = self.rtx_accelerator {
            if rtx.is_rtx_available() {
                return rtx.process_spectral_denoising(input, output, self.config.noise_suppression.strength);
            }
        }

        // Fall back to CPU processing
        if let Ok(mut processor) = self.noise_processor.lock() {
            processor.process(input, output)
        } else {
            // If we can't lock, pass through
            output.copy_from_slice(input);
            Ok(())
        }
    }

    /// Get the current processing mode (RTX GPU or CPU)
    pub fn get_processing_mode(&self) -> String {
        #[cfg(feature = "nvidia-rtx")]
        if let Some(ref rtx) = self.rtx_accelerator {
            if rtx.is_rtx_available() {
                return rtx.get_processing_mode().to_string();
            }
        }

        if let Ok(processor) = self.noise_processor.lock() {
            processor.get_processing_mode()
        } else {
            "Unknown".to_string()
        }
    }

    /// Check if RTX acceleration is available
    #[cfg(feature = "nvidia-rtx")]
    pub fn has_rtx_acceleration(&self) -> bool {
        self.rtx_accelerator
            .as_ref()
            .map(|rtx| rtx.is_rtx_available())
            .unwrap_or(false)
    }

    #[cfg(not(feature = "nvidia-rtx"))]
    pub fn has_rtx_acceleration(&self) -> bool {
        false
    }

    /// Get RTX capabilities if available
    #[cfg(feature = "nvidia-rtx")]
    pub fn get_rtx_capabilities(&self) -> Option<RtxCapabilities> {
        self.rtx_accelerator
            .as_ref()
            .and_then(|rtx| rtx.get_capabilities())
    }

    #[cfg(not(feature = "nvidia-rtx"))]
    pub fn get_rtx_capabilities(&self) -> Option<()> {
        None
    }

    /// Get the current configuration
    pub fn get_config(&self) -> &Config {
        &self.config
    }

    /// Update noise suppression settings
    pub fn update_noise_suppression(&self, config: NoiseSuppressionConfig) -> Result<()> {
        if let Ok(mut processor) = self.noise_processor.lock() {
            *processor = NoiseProcessor::new(&config)?;
            info!("Updated noise suppression settings");
            Ok(())
        } else {
            Err(anyhow::anyhow!("Failed to acquire processor lock"))
        }
    }
}

/// Audio backend types supported by GhostWave
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioBackend {
    PipeWire,
    Alsa,
    Jack,
    Cpal,
}

impl AudioBackend {
    /// Get all compiled-in backends
    pub fn available_backends() -> Vec<AudioBackend> {
        #[allow(unused_mut)]
        let mut backends = Vec::new();

        #[cfg(feature = "pipewire-backend")]
        backends.push(AudioBackend::PipeWire);

        #[cfg(feature = "alsa-backend")]
        backends.push(AudioBackend::Alsa);

        #[cfg(feature = "jack-backend")]
        backends.push(AudioBackend::Jack);

        #[cfg(feature = "cpal-backend")]
        backends.push(AudioBackend::Cpal);

        backends
    }

    /// Check if a backend is available at runtime
    pub fn is_available(&self) -> bool {
        match self {
            AudioBackend::PipeWire => {
                #[cfg(feature = "pipewire-backend")]
                {
                    // Check if PipeWire is running
                    true // Simplified check
                }
                #[cfg(not(feature = "pipewire-backend"))]
                {
                    false
                }
            }

            AudioBackend::Alsa => {
                #[cfg(feature = "alsa-backend")]
                {
                    crate::alsa::check_alsa_availability()
                }
                #[cfg(not(feature = "alsa-backend"))]
                {
                    false
                }
            }

            AudioBackend::Jack => {
                #[cfg(feature = "jack-backend")]
                {
                    crate::jack::check_jack_availability()
                }
                #[cfg(not(feature = "jack-backend"))]
                {
                    false
                }
            }

            AudioBackend::Cpal => {
                #[cfg(feature = "cpal-backend")]
                {
                    crate::cpal_backend::check_cpal_availability()
                }
                #[cfg(not(feature = "cpal-backend"))]
                {
                    false
                }
            }
        }
    }

    /// Get the recommended backend for the current system
    pub fn recommended() -> Option<AudioBackend> {
        let backends = Self::available_backends();

        // Preference order: JACK (pro audio) > PipeWire (modern) > ALSA (direct) > CPAL (fallback)
        for &backend in &[
            #[cfg(feature = "jack-backend")]
            AudioBackend::Jack,
            #[cfg(feature = "pipewire-backend")]
            AudioBackend::PipeWire,
            #[cfg(feature = "alsa-backend")]
            AudioBackend::Alsa,
            #[cfg(feature = "cpal-backend")]
            AudioBackend::Cpal,
        ] {
            if backends.contains(&backend) && backend.is_available() {
                return Some(backend);
            }
        }

        None
    }
}

impl std::fmt::Display for AudioBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioBackend::PipeWire => write!(f, "PipeWire"),
            AudioBackend::Alsa => write!(f, "ALSA"),
            AudioBackend::Jack => write!(f, "JACK"),
            AudioBackend::Cpal => write!(f, "CPAL"),
        }
    }
}

/// System information and capabilities
pub struct SystemInfo {
    pub available_backends: Vec<AudioBackend>,
    pub recommended_backend: Option<AudioBackend>,
    pub detected_devices: Vec<AudioDevice>,
    pub rtx_available: bool,
    #[cfg(feature = "nvidia-rtx")]
    pub rtx_capabilities: Option<RtxCapabilities>,
}

impl SystemInfo {
    /// Detect system capabilities and available backends
    pub async fn detect() -> Result<Self> {
        info!("Detecting system audio capabilities");

        let available_backends = AudioBackend::available_backends()
            .into_iter()
            .filter(|backend| backend.is_available())
            .collect();

        let recommended_backend = AudioBackend::recommended();

        let detector = DeviceDetector::new();
        let detected_devices = detector.detect_devices().await?;

        #[cfg(feature = "nvidia-rtx")]
        let (rtx_available, rtx_capabilities) = match RtxAccelerator::new() {
            Ok(rtx) => (rtx.is_rtx_available(), rtx.get_capabilities()),
            Err(_) => (false, None),
        };

        #[cfg(not(feature = "nvidia-rtx"))]
        let rtx_available = false;

        info!("System detection complete:");
        info!("  Available backends: {:?}", available_backends);
        info!("  Recommended backend: {:?}", recommended_backend);
        info!("  Detected devices: {}", detected_devices.len());
        info!("  RTX available: {}", rtx_available);

        Ok(SystemInfo {
            available_backends,
            recommended_backend,
            detected_devices,
            rtx_available,
            #[cfg(feature = "nvidia-rtx")]
            rtx_capabilities,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_creation() {
        let config = Config::default();
        let processor = GhostWaveProcessor::new(config);
        assert!(processor.is_ok());
    }

    #[test]
    fn test_audio_processing() {
        let config = Config::default();
        let processor = GhostWaveProcessor::new(config).unwrap();

        let input = vec![0.1f32; 256];
        let mut output = vec![0.0f32; 256];

        let result = processor.process(&input, &mut output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_backend_detection() {
        let backends = AudioBackend::available_backends();
        println!("Available backends: {:?}", backends);

        if let Some(recommended) = AudioBackend::recommended() {
            println!("Recommended backend: {}", recommended);
            assert!(recommended.is_available());
        }
    }

    #[tokio::test]
    async fn test_system_detection() {
        let system_info = SystemInfo::detect().await.unwrap();
        println!("System info: available backends = {:?}", system_info.available_backends);
        println!("RTX available: {}", system_info.rtx_available);
    }
}