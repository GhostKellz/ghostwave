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
pub mod config_v2;
pub mod processor;
pub mod frame_format;
pub mod dsp_pipeline;
pub mod latency_optimizer;
pub mod noise_suppression;
pub mod low_latency;
pub mod device_detection;
pub mod device_manager;
pub mod structured_logging;
pub mod pipewire_integration;
pub mod ipc_server;
pub mod simd_acceleration;
pub mod gpu_acceleration;

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
pub use processor::{AudioProcessor, BypassableProcessor, ProcessingProfile, ParamValue, ParamDescriptor, ProfileParams};
pub use frame_format::{FrameFormat, AudioBuffer, Sample};
pub use dsp_pipeline::DspPipeline;
pub use noise_suppression::NoiseProcessor;
pub use low_latency::{LockFreeAudioBuffer, RealTimeScheduler, AudioBenchmark, TARGET_LATENCY_MS};
pub use device_detection::{DeviceDetector, AudioDevice, AudioDeviceType};
pub use device_manager::{DeviceManager, DeviceManagerBuilder, DeviceSelectionConfig, HotplugEvent};

#[cfg(feature = "nvidia-rtx")]
pub use rtx_acceleration::{RtxAccelerator, RtxCapabilities};

/// Main GhostWave audio processor
pub struct GhostWaveProcessor {
    config: Config,
    profile: ProcessingProfile,
    profile_params: ProfileParams,
    dsp_pipeline: Option<DspPipeline>,
    noise_processor: Arc<Mutex<NoiseProcessor>>,
    initialized: bool,
    sample_rate: u32,
    channels: u32,
    frame_format: FrameFormat,

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

        let profile_params = ProfileParams::new();

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
            profile: ProcessingProfile::default(),
            profile_params,
            dsp_pipeline: None,
            noise_processor,
            initialized: false,
            sample_rate: 48000,
            channels: 1,
            frame_format: FrameFormat::default(),
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

impl AudioProcessor for GhostWaveProcessor {
    fn init(&mut self, sample_rate: u32, channels: u32, max_buffer_size: usize) -> Result<()> {
        info!("Initializing GhostWave processor: {}Hz, {} channels, max {} frames",
              sample_rate, channels, max_buffer_size);

        self.sample_rate = sample_rate;
        self.channels = channels;

        // Create frame format
        self.frame_format = FrameFormat::new(channels as u8, sample_rate, max_buffer_size)?;

        // Update config with new parameters
        self.config.audio.sample_rate = sample_rate;
        self.config.audio.channels = channels as u16;
        self.config.audio.buffer_size = max_buffer_size as u32;

        // Initialize DSP pipeline
        self.dsp_pipeline = Some(DspPipeline::new(self.frame_format, self.profile));

        // Initialize noise processor with updated config
        *self.noise_processor.lock().map_err(|_| anyhow::anyhow!("Lock poisoned"))? =
            NoiseProcessor::new(&self.config.noise_suppression)?;

        self.initialized = true;
        Ok(())
    }

    fn process_inplace(&mut self, buffer: &mut [f32], frames: usize) -> Result<()> {
        use crate::processor::utils;

        if !self.initialized {
            return Err(anyhow::anyhow!("Processor not initialized"));
        }

        // Validate buffer format
        utils::validate_buffer(buffer, self.channels, frames)?;

        // Apply runtime safety guardrails first
        utils::scrub_denormals(buffer);

        // Skip all processing if disabled
        if !self.config.noise_suppression.enabled {
            return Ok(());
        }

        // Use DSP pipeline for primary processing
        if let Some(ref mut pipeline) = self.dsp_pipeline {
            pipeline.process(buffer)?;
        } else {
            // Fallback to legacy processing chain

            // Try RTX acceleration first
            #[cfg(feature = "nvidia-rtx")]
            if let Some(ref rtx) = self.rtx_accelerator {
                if rtx.is_rtx_available() {
                    // For in-place processing, we need a temporary buffer for RTX
                    let mut temp_output = vec![0.0f32; buffer.len()];
                    rtx.process_spectral_denoising(buffer, &mut temp_output, self.config.noise_suppression.strength)?;
                    buffer.copy_from_slice(&temp_output);
                    return Ok(());
                }
            }

            // Fall back to CPU processing
            if let Ok(mut processor) = self.noise_processor.lock() {
                // For in-place processing, we need a temporary input buffer
                let input_buffer = buffer.to_vec();
                processor.process(&input_buffer, buffer)?;
            } else {
                return Err(anyhow::anyhow!("Failed to acquire processor lock"));
            }

            // Apply soft limiting if enabled
            if self.get_param("limiter_enabled").unwrap_or(ParamValue::Bool(false)) == ParamValue::Bool(true) {
                utils::soft_clip_buffer(buffer, 0.95);
            }
        }

        Ok(())
    }

    fn set_profile(&mut self, profile: ProcessingProfile) -> Result<()> {
        info!("Switching to profile: {}", profile);
        self.profile = profile;

        // Update DSP pipeline profile
        if let Some(ref mut pipeline) = self.dsp_pipeline {
            pipeline.set_profile(profile);
        }

        // Apply profile-specific parameters
        self.profile_params.apply_to_processor(self, profile)?;

        Ok(())
    }

    fn get_profile(&self) -> ProcessingProfile {
        self.profile
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        // First try to set parameter in DSP pipeline if available
        if let Some(ref mut pipeline) = self.dsp_pipeline {
            if pipeline.set_param(name, value.clone()).is_ok() {
                self.profile_params.set_profile_param(self.profile, name.to_string(), value);
                debug!("Set parameter {} = {:?} in DSP pipeline", name, value);
                return Ok(());
            }
        }

        // Fallback to legacy parameter handling
        match name {
            "noise_reduction_strength" => {
                if let ParamValue::Float(strength) = value {
                    if strength < 0.0 || strength > 1.0 {
                        return Err(anyhow::anyhow!("Noise reduction strength must be between 0.0 and 1.0"));
                    }
                    self.config.noise_suppression.strength = strength;
                } else {
                    return Err(anyhow::anyhow!("Expected float value for noise_reduction_strength"));
                }
            }
            "voice_enhancement" => {
                if let ParamValue::Float(enhancement) = value {
                    if enhancement < 0.0 || enhancement > 1.0 {
                        return Err(anyhow::anyhow!("Voice enhancement must be between 0.0 and 1.0"));
                    }
                    self.profile_params.set_profile_param(self.profile, name.to_string(), ParamValue::Float(enhancement));
                } else {
                    return Err(anyhow::anyhow!("Expected float value for voice_enhancement"));
                }
            }
            "gate_threshold" => {
                if let ParamValue::Float(threshold) = value {
                    if threshold < -80.0 || threshold > 0.0 {
                        return Err(anyhow::anyhow!("Gate threshold must be between -80.0 and 0.0 dB"));
                    }
                    self.profile_params.set_profile_param(self.profile, name.to_string(), ParamValue::Float(threshold));
                } else {
                    return Err(anyhow::anyhow!("Expected float value for gate_threshold"));
                }
            }
            "limiter_enabled" => {
                if let ParamValue::Bool(enabled) = value {
                    self.profile_params.set_profile_param(self.profile, name.to_string(), ParamValue::Bool(enabled));
                } else {
                    return Err(anyhow::anyhow!("Expected boolean value for limiter_enabled"));
                }
            }
            "highpass_frequency" => {
                if let ParamValue::Float(freq) = value {
                    if freq < 20.0 || freq > 500.0 {
                        return Err(anyhow::anyhow!("Highpass frequency must be between 20.0 and 500.0 Hz"));
                    }
                    self.profile_params.set_profile_param(self.profile, name.to_string(), ParamValue::Float(freq));
                } else {
                    return Err(anyhow::anyhow!("Expected float value for highpass_frequency"));
                }
            }
            _ => return Err(anyhow::anyhow!("Unknown parameter: {}", name)),
        }

        debug!("Set parameter {} = {:?}", name, value);
        Ok(())
    }

    fn get_param(&self, name: &str) -> Result<ParamValue> {
        match name {
            "noise_reduction_strength" => Ok(ParamValue::Float(self.config.noise_suppression.strength)),
            "voice_enhancement" | "gate_threshold" | "limiter_enabled" | "highpass_frequency" => {
                if let Some(params) = self.profile_params.get_profile_params(self.profile) {
                    params.get(name).cloned().ok_or_else(|| anyhow::anyhow!("Parameter not found: {}", name))
                } else {
                    Err(anyhow::anyhow!("No parameters found for current profile"))
                }
            }
            _ => Err(anyhow::anyhow!("Unknown parameter: {}", name)),
        }
    }

    fn get_params(&self) -> std::collections::HashMap<String, ParamDescriptor> {
        use std::collections::HashMap;

        let mut params = HashMap::new();

        // Add noise reduction strength parameter
        params.insert("noise_reduction_strength".to_string(), ParamDescriptor {
            name: "noise_reduction_strength".to_string(),
            description: "Strength of noise reduction processing (0.0 = disabled, 1.0 = maximum)".to_string(),
            value: ParamValue::Float(self.config.noise_suppression.strength),
            default: ParamValue::Float(0.7),
            min: Some(ParamValue::Float(0.0)),
            max: Some(ParamValue::Float(1.0)),
            runtime_adjustable: true,
            category: "Noise Suppression".to_string(),
        });

        // Add other parameters from profile
        if let Some(profile_params) = self.profile_params.get_profile_params(self.profile) {
            for (name, value) in profile_params {
                let descriptor = match name.as_str() {
                    "voice_enhancement" => ParamDescriptor {
                        name: name.clone(),
                        description: "Voice clarity enhancement (0.0 = natural, 1.0 = enhanced)".to_string(),
                        value: value.clone(),
                        default: ParamValue::Float(0.5),
                        min: Some(ParamValue::Float(0.0)),
                        max: Some(ParamValue::Float(1.0)),
                        runtime_adjustable: true,
                        category: "Voice Processing".to_string(),
                    },
                    "gate_threshold" => ParamDescriptor {
                        name: name.clone(),
                        description: "Gate threshold in dB (silence below this level is removed)".to_string(),
                        value: value.clone(),
                        default: ParamValue::Float(-45.0),
                        min: Some(ParamValue::Float(-80.0)),
                        max: Some(ParamValue::Float(0.0)),
                        runtime_adjustable: true,
                        category: "Gate".to_string(),
                    },
                    "limiter_enabled" => ParamDescriptor {
                        name: name.clone(),
                        description: "Enable output limiter to prevent clipping".to_string(),
                        value: value.clone(),
                        default: ParamValue::Bool(true),
                        min: None,
                        max: None,
                        runtime_adjustable: true,
                        category: "Output".to_string(),
                    },
                    "highpass_frequency" => ParamDescriptor {
                        name: name.clone(),
                        description: "High-pass filter frequency in Hz (removes low-frequency noise)".to_string(),
                        value: value.clone(),
                        default: ParamValue::Float(80.0),
                        min: Some(ParamValue::Float(20.0)),
                        max: Some(ParamValue::Float(500.0)),
                        runtime_adjustable: true,
                        category: "Filtering".to_string(),
                    },
                    _ => continue,
                };
                params.insert(name.clone(), descriptor);
            }
        }

        params
    }

    fn name(&self) -> &'static str {
        "GhostWave"
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn reset(&mut self) -> Result<()> {
        info!("Resetting GhostWave processor");

        // Reinitialize noise processor
        *self.noise_processor.lock().map_err(|_| anyhow::anyhow!("Lock poisoned"))? =
            NoiseProcessor::new(&self.config.noise_suppression)?;

        Ok(())
    }

    fn latency_frames(&self) -> usize {
        if let Some(ref pipeline) = self.dsp_pipeline {
            pipeline.latency_frames()
        } else {
            // Fallback estimate
            self.config.audio.buffer_size as usize / 2
        }
    }

    fn cpu_usage(&self) -> f32 {
        // TODO: Implement actual CPU usage measurement
        0.0
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