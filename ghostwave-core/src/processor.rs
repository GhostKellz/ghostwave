//! # GhostWave Audio Processor Trait
//!
//! This module defines the core processing trait and API for GhostWave audio processing.
//! It provides a standardized interface for all audio processing components with support
//! for initialization, in-place processing, profile management, and parameter adjustment.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Audio processing profile types supported by GhostWave
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessingProfile {
    /// Balanced profile for general use - good noise reduction with natural voice
    Balanced,
    /// Optimized for streaming - aggressive noise reduction, optimized for microphones
    Streaming,
    /// Studio quality - minimal processing, maximum fidelity for professional recording
    Studio,
}

impl ProcessingProfile {
    /// Get all available profiles
    pub fn all() -> &'static [ProcessingProfile] {
        &[
            ProcessingProfile::Balanced,
            ProcessingProfile::Streaming,
            ProcessingProfile::Studio,
        ]
    }

    /// Get profile description
    pub fn description(&self) -> &'static str {
        match self {
            ProcessingProfile::Balanced => "Balanced noise reduction for everyday use",
            ProcessingProfile::Streaming => "Aggressive noise reduction optimized for streaming",
            ProcessingProfile::Studio => "Minimal processing for professional recording",
        }
    }
}

impl Default for ProcessingProfile {
    fn default() -> Self {
        ProcessingProfile::Balanced
    }
}

impl std::fmt::Display for ProcessingProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessingProfile::Balanced => write!(f, "balanced"),
            ProcessingProfile::Streaming => write!(f, "streaming"),
            ProcessingProfile::Studio => write!(f, "studio"),
        }
    }
}

impl std::str::FromStr for ProcessingProfile {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "balanced" => Ok(ProcessingProfile::Balanced),
            "streaming" => Ok(ProcessingProfile::Streaming),
            "studio" => Ok(ProcessingProfile::Studio),
            _ => Err(anyhow::anyhow!("Unknown profile: {}", s)),
        }
    }
}

/// Parameter value types supported by the processor
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParamValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    String(String),
}

impl From<f32> for ParamValue {
    fn from(value: f32) -> Self { ParamValue::Float(value) }
}

impl From<i32> for ParamValue {
    fn from(value: i32) -> Self { ParamValue::Int(value) }
}

impl From<bool> for ParamValue {
    fn from(value: bool) -> Self { ParamValue::Bool(value) }
}

impl From<String> for ParamValue {
    fn from(value: String) -> Self { ParamValue::String(value) }
}

impl From<&str> for ParamValue {
    fn from(value: &str) -> Self { ParamValue::String(value.to_string()) }
}

/// Parameter descriptor for runtime parameter discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamDescriptor {
    /// Parameter name
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Current value
    pub value: ParamValue,
    /// Default value
    pub default: ParamValue,
    /// Minimum value (for numeric types)
    pub min: Option<ParamValue>,
    /// Maximum value (for numeric types)
    pub max: Option<ParamValue>,
    /// Whether this parameter can be changed at runtime
    pub runtime_adjustable: bool,
    /// Parameter category for UI grouping
    pub category: String,
}

/// Core audio processing trait that all GhostWave processors must implement
///
/// This trait provides a standardized interface for audio processing components
/// with support for initialization, real-time processing, profile management,
/// and runtime parameter adjustment.
pub trait AudioProcessor: Send + Sync {
    /// Initialize the processor with the given configuration
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate in Hz (e.g., 48000)
    /// * `channels` - Number of audio channels (1 for mono, 2 for stereo)
    /// * `max_buffer_size` - Maximum expected buffer size in frames
    ///
    /// # Returns
    /// * `Ok(())` on successful initialization
    /// * `Err(anyhow::Error)` on initialization failure
    fn init(&mut self, sample_rate: u32, channels: u32, max_buffer_size: usize) -> Result<()>;

    /// Process audio in-place with the current settings
    ///
    /// This is the main processing method called in the audio thread.
    /// It must be real-time safe (no allocations, no blocking operations).
    ///
    /// # Arguments
    /// * `buffer` - Interleaved f32 audio samples (input/output)
    /// * `frames` - Number of frames to process
    ///
    /// # Returns
    /// * `Ok(())` on successful processing
    /// * `Err(anyhow::Error)` on processing error
    ///
    /// # Safety
    /// This method must be real-time safe and should not allocate memory
    /// or perform any blocking operations.
    fn process_inplace(&mut self, buffer: &mut [f32], frames: usize) -> Result<()>;

    /// Set the current processing profile
    ///
    /// This changes the overall processing characteristics and may update
    /// multiple internal parameters. The change should take effect on the
    /// next process_inplace call.
    ///
    /// # Arguments
    /// * `profile` - The processing profile to activate
    ///
    /// # Returns
    /// * `Ok(())` on successful profile change
    /// * `Err(anyhow::Error)` if profile is not supported or change failed
    fn set_profile(&mut self, profile: ProcessingProfile) -> Result<()>;

    /// Get the current processing profile
    fn get_profile(&self) -> ProcessingProfile;

    /// Set a runtime parameter by name
    ///
    /// # Arguments
    /// * `name` - Parameter name (case-sensitive)
    /// * `value` - New parameter value
    ///
    /// # Returns
    /// * `Ok(())` on successful parameter update
    /// * `Err(anyhow::Error)` if parameter doesn't exist or value is invalid
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()>;

    /// Get a parameter value by name
    ///
    /// # Arguments
    /// * `name` - Parameter name (case-sensitive)
    ///
    /// # Returns
    /// * `Ok(ParamValue)` with current parameter value
    /// * `Err(anyhow::Error)` if parameter doesn't exist
    fn get_param(&self, name: &str) -> Result<ParamValue>;

    /// Get all available parameters with their descriptors
    ///
    /// This is used for runtime parameter discovery and UI generation.
    ///
    /// # Returns
    /// * `HashMap<String, ParamDescriptor>` mapping parameter names to descriptors
    fn get_params(&self) -> HashMap<String, ParamDescriptor>;

    /// Get supported processing profiles
    ///
    /// # Returns
    /// * `Vec<ProcessingProfile>` containing all supported profiles
    fn supported_profiles(&self) -> Vec<ProcessingProfile> {
        ProcessingProfile::all().to_vec()
    }

    /// Get processor name for identification
    fn name(&self) -> &'static str;

    /// Get processor version
    fn version(&self) -> &'static str {
        "1.0.0"
    }

    /// Check if processor is initialized and ready
    fn is_initialized(&self) -> bool;

    /// Reset processor to initial state (keeping current profile and parameters)
    fn reset(&mut self) -> Result<()>;

    /// Get current latency in frames
    ///
    /// Returns the processing latency introduced by this processor.
    /// This is used for latency compensation in the audio pipeline.
    fn latency_frames(&self) -> usize {
        0  // Default: no latency
    }

    /// Get current CPU usage estimate (0.0 to 1.0)
    ///
    /// This is an estimate of CPU usage for monitoring and adaptive processing.
    fn cpu_usage(&self) -> f32 {
        0.0  // Default: unknown/no measurement
    }
}

/// Convenience trait for processors that support bypass
pub trait BypassableProcessor: AudioProcessor {
    /// Set bypass state (true = bypass processing, false = process normally)
    fn set_bypass(&mut self, bypass: bool);

    /// Get current bypass state
    fn is_bypassed(&self) -> bool;
}

/// Profile parameter sets for different processing modes
pub struct ProfileParams {
    params: HashMap<ProcessingProfile, HashMap<String, ParamValue>>,
}

impl ProfileParams {
    /// Create new profile parameters with defaults
    pub fn new() -> Self {
        let mut profile_params = Self {
            params: HashMap::new(),
        };

        // Initialize default parameters for each profile
        profile_params.init_profile_defaults();
        profile_params
    }

    /// Initialize default parameters for all profiles
    fn init_profile_defaults(&mut self) {
        // Balanced profile defaults
        let mut balanced = HashMap::new();
        balanced.insert("noise_reduction_strength".to_string(), ParamValue::Float(0.7));
        balanced.insert("voice_enhancement".to_string(), ParamValue::Float(0.5));
        balanced.insert("gate_threshold".to_string(), ParamValue::Float(-45.0));
        balanced.insert("limiter_enabled".to_string(), ParamValue::Bool(true));
        balanced.insert("highpass_frequency".to_string(), ParamValue::Float(80.0));
        self.params.insert(ProcessingProfile::Balanced, balanced);

        // Streaming profile defaults - more aggressive
        let mut streaming = HashMap::new();
        streaming.insert("noise_reduction_strength".to_string(), ParamValue::Float(0.85));
        streaming.insert("voice_enhancement".to_string(), ParamValue::Float(0.7));
        streaming.insert("gate_threshold".to_string(), ParamValue::Float(-40.0));
        streaming.insert("limiter_enabled".to_string(), ParamValue::Bool(true));
        streaming.insert("highpass_frequency".to_string(), ParamValue::Float(100.0));
        self.params.insert(ProcessingProfile::Streaming, streaming);

        // Studio profile defaults - minimal processing
        let mut studio = HashMap::new();
        studio.insert("noise_reduction_strength".to_string(), ParamValue::Float(0.3));
        studio.insert("voice_enhancement".to_string(), ParamValue::Float(0.2));
        studio.insert("gate_threshold".to_string(), ParamValue::Float(-60.0));
        studio.insert("limiter_enabled".to_string(), ParamValue::Bool(false));
        studio.insert("highpass_frequency".to_string(), ParamValue::Float(40.0));
        self.params.insert(ProcessingProfile::Studio, studio);
    }

    /// Get parameters for a specific profile
    pub fn get_profile_params(&self, profile: ProcessingProfile) -> Option<&HashMap<String, ParamValue>> {
        self.params.get(&profile)
    }

    /// Set parameter for a specific profile
    pub fn set_profile_param(&mut self, profile: ProcessingProfile, name: String, value: ParamValue) {
        self.params.entry(profile).or_insert_with(HashMap::new).insert(name, value);
    }

    /// Apply profile parameters to a processor
    pub fn apply_to_processor(&self, processor: &mut dyn AudioProcessor, profile: ProcessingProfile) -> Result<()> {
        if let Some(params) = self.get_profile_params(profile) {
            for (name, value) in params {
                processor.set_param(name, value.clone())?;
            }
        }
        Ok(())
    }
}

impl Default for ProfileParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for audio processing
pub mod utils {
    use super::*;

    /// Validate interleaved audio buffer format
    pub fn validate_buffer(buffer: &[f32], channels: u32, expected_frames: usize) -> Result<()> {
        let expected_samples = expected_frames * channels as usize;
        if buffer.len() != expected_samples {
            return Err(anyhow::anyhow!(
                "Buffer size mismatch: expected {} samples ({} frames × {} channels), got {}",
                expected_samples, expected_frames, channels, buffer.len()
            ));
        }
        Ok(())
    }

    /// Scrub NaN and infinite values from audio buffer
    pub fn scrub_denormals(buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
            }
            // Denormal scrubbing: values very close to zero become zero
            if sample.abs() < 1e-30 {
                *sample = 0.0;
            }
        }
    }

    /// Apply soft clipping to prevent harsh distortion
    pub fn soft_clip(sample: f32, threshold: f32) -> f32 {
        if sample.abs() <= threshold {
            sample
        } else {
            let sign = sample.signum();
            let abs_sample = sample.abs();
            let excess = (abs_sample - threshold) / (1.0 - threshold);
            sign * (threshold + (1.0 - threshold) * excess.tanh())
        }
    }

    /// Apply soft clipping to entire buffer
    pub fn soft_clip_buffer(buffer: &mut [f32], threshold: f32) {
        for sample in buffer.iter_mut() {
            *sample = soft_clip(*sample, threshold);
        }
    }

    /// Convert dB to linear gain
    pub fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    /// Convert linear gain to dB
    pub fn linear_to_db(linear: f32) -> f32 {
        20.0 * linear.max(1e-10).log10()
    }

    /// Simple RMS calculation for level metering
    pub fn calculate_rms(buffer: &[f32]) -> f32 {
        if buffer.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = buffer.iter().map(|&x| x * x).sum();
        (sum_squares / buffer.len() as f32).sqrt()
    }

    /// Peak detection for level metering
    pub fn find_peak(buffer: &[f32]) -> f32 {
        buffer.iter().fold(0.0, |max, &x| max.max(x.abs()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_parsing() {
        assert_eq!("balanced".parse::<ProcessingProfile>().unwrap(), ProcessingProfile::Balanced);
        assert_eq!("streaming".parse::<ProcessingProfile>().unwrap(), ProcessingProfile::Streaming);
        assert_eq!("studio".parse::<ProcessingProfile>().unwrap(), ProcessingProfile::Studio);

        assert!("invalid".parse::<ProcessingProfile>().is_err());
    }

    #[test]
    fn test_param_values() {
        let float_param = ParamValue::from(3.14f32);
        let int_param = ParamValue::from(42i32);
        let bool_param = ParamValue::from(true);
        let string_param = ParamValue::from("test");

        match float_param {
            ParamValue::Float(val) => assert_eq!(val, 3.14),
            _ => panic!("Expected float param"),
        }

        match int_param {
            ParamValue::Int(val) => assert_eq!(val, 42),
            _ => panic!("Expected int param"),
        }

        match bool_param {
            ParamValue::Bool(val) => assert!(val),
            _ => panic!("Expected bool param"),
        }

        match string_param {
            ParamValue::String(val) => assert_eq!(val, "test"),
            _ => panic!("Expected string param"),
        }
    }

    #[test]
    fn test_profile_params() {
        let profile_params = ProfileParams::new();

        // Test that all profiles have default parameters
        for &profile in ProcessingProfile::all() {
            assert!(profile_params.get_profile_params(profile).is_some());

            let params = profile_params.get_profile_params(profile).unwrap();
            assert!(params.contains_key("noise_reduction_strength"));
            assert!(params.contains_key("gate_threshold"));
        }
    }

    #[test]
    fn test_audio_utils() {
        // Test buffer validation
        let buffer = vec![0.0f32; 512]; // 256 frames, 2 channels
        assert!(utils::validate_buffer(&buffer, 2, 256).is_ok());
        assert!(utils::validate_buffer(&buffer, 1, 256).is_err()); // Wrong channel count

        // Test denormal scrubbing
        let mut buffer = vec![1.0, std::f32::NAN, std::f32::INFINITY, 1e-40, 0.5];
        utils::scrub_denormals(&mut buffer);
        assert!(buffer[0] == 1.0);  // Normal value unchanged
        assert!(buffer[1] == 0.0);  // NaN scrubbed
        assert!(buffer[2] == 0.0);  // Infinity scrubbed
        assert!(buffer[3] == 0.0);  // Denormal scrubbed
        assert!(buffer[4] == 0.5);  // Normal value unchanged

        // Test soft clipping
        assert_eq!(utils::soft_clip(0.5, 0.8), 0.5);  // Below threshold
        assert!(utils::soft_clip(1.0, 0.8) < 1.0);    // Above threshold, clipped

        // Test dB conversions
        assert!((utils::db_to_linear(0.0) - 1.0).abs() < 1e-6);
        assert!((utils::db_to_linear(20.0) - 10.0).abs() < 1e-6);
        assert!((utils::linear_to_db(1.0) - 0.0).abs() < 1e-6);
        assert!((utils::linear_to_db(10.0) - 20.0).abs() < 1e-6);

        // Test RMS calculation
        let rms_buffer = vec![1.0, -1.0, 1.0, -1.0];
        assert!((utils::calculate_rms(&rms_buffer) - 1.0).abs() < 1e-6);

        // Test peak detection
        let peak_buffer = vec![0.1, -0.5, 0.3, -0.8, 0.2];
        assert_eq!(utils::find_peak(&peak_buffer), 0.8);
    }
}