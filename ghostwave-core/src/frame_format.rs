//! # GhostWave Audio Frame Format Specification
//!
//! This module defines the stable audio frame format used throughout GhostWave.
//! All audio processing components must use this standardized format for
//! interoperability and consistent performance.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Standard audio sample type used by GhostWave
pub type Sample = f32;

/// Standard frame format for GhostWave audio processing
///
/// This format is designed for optimal performance and compatibility:
/// - f32 samples for high dynamic range and precision
/// - Interleaved channel layout for cache efficiency
/// - Standard sample rates for broad compatibility
/// - Stereo and mono support for different use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FrameFormat {
    /// Number of audio channels
    pub channels: u8,
    /// Audio sample rate in Hz
    pub sample_rate: u32,
    /// Buffer size in frames (samples per channel)
    pub buffer_size: usize,
}

impl FrameFormat {
    /// Default format: 48kHz mono with 128-frame buffer
    pub const DEFAULT: Self = Self {
        channels: 1,
        sample_rate: 48000,
        buffer_size: 128,
    };

    /// Create a new frame format
    pub fn new(channels: u8, sample_rate: u32, buffer_size: usize) -> Result<Self> {
        Self::validate_params(channels, sample_rate, buffer_size)?;
        Ok(Self {
            channels,
            sample_rate,
            buffer_size,
        })
    }

    /// Create mono format
    pub fn mono(sample_rate: u32, buffer_size: usize) -> Result<Self> {
        Self::new(1, sample_rate, buffer_size)
    }

    /// Create stereo format
    pub fn stereo(sample_rate: u32, buffer_size: usize) -> Result<Self> {
        Self::new(2, sample_rate, buffer_size)
    }

    /// Validate frame format parameters
    fn validate_params(channels: u8, sample_rate: u32, buffer_size: usize) -> Result<()> {
        if channels == 0 || channels > 8 {
            return Err(anyhow::anyhow!("Invalid channel count: {} (must be 1-8)", channels));
        }

        if !Self::is_supported_sample_rate(sample_rate) {
            return Err(anyhow::anyhow!("Unsupported sample rate: {} Hz", sample_rate));
        }

        if !Self::is_valid_buffer_size(buffer_size) {
            return Err(anyhow::anyhow!("Invalid buffer size: {} frames", buffer_size));
        }

        Ok(())
    }

    /// Check if sample rate is supported
    pub fn is_supported_sample_rate(sample_rate: u32) -> bool {
        matches!(sample_rate,
            22050 | 24000 | 32000 | 44100 | 48000 | 88200 | 96000 | 176400 | 192000
        )
    }

    /// Check if buffer size is valid (power of 2 between 32 and 4096)
    pub fn is_valid_buffer_size(buffer_size: usize) -> bool {
        buffer_size >= 32 &&
        buffer_size <= 4096 &&
        buffer_size.is_power_of_two()
    }

    /// Get all supported sample rates
    pub fn supported_sample_rates() -> &'static [u32] {
        &[22050, 24000, 32000, 44100, 48000, 88200, 96000, 176400, 192000]
    }

    /// Get recommended buffer sizes for different latency requirements
    pub fn recommended_buffer_sizes() -> &'static [(usize, &'static str)] {
        &[
            (32, "Ultra-low latency (<1ms) - Professional only"),
            (64, "Very low latency (<2ms) - Pro audio"),
            (128, "Low latency (<3ms) - Default/Balanced"),
            (256, "Moderate latency (<6ms) - Streaming"),
            (512, "High latency (<12ms) - Studio"),
            (1024, "Very high latency (<24ms) - Batch processing"),
        ]
    }

    /// Calculate the duration of one buffer in milliseconds
    pub fn buffer_duration_ms(&self) -> f64 {
        (self.buffer_size as f64 / self.sample_rate as f64) * 1000.0
    }

    /// Calculate the total number of samples in a buffer (all channels)
    pub fn samples_per_buffer(&self) -> usize {
        self.buffer_size * self.channels as usize
    }

    /// Calculate buffer size in bytes for f32 samples
    pub fn buffer_size_bytes(&self) -> usize {
        self.samples_per_buffer() * std::mem::size_of::<Sample>()
    }

    /// Get the Nyquist frequency (half the sample rate)
    pub fn nyquist_frequency(&self) -> f32 {
        self.sample_rate as f32 / 2.0
    }

    /// Check if this format is compatible with another format
    pub fn is_compatible_with(&self, other: &FrameFormat) -> bool {
        self.sample_rate == other.sample_rate && self.channels == other.channels
    }

    /// Create a new format with different buffer size
    pub fn with_buffer_size(&self, buffer_size: usize) -> Result<Self> {
        Self::new(self.channels, self.sample_rate, buffer_size)
    }

    /// Create a new format with different channel count
    pub fn with_channels(&self, channels: u8) -> Result<Self> {
        Self::new(channels, self.sample_rate, self.buffer_size)
    }

    /// Create a new format with different sample rate
    pub fn with_sample_rate(&self, sample_rate: u32) -> Result<Self> {
        Self::new(self.channels, sample_rate, self.buffer_size)
    }
}

impl Default for FrameFormat {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl std::fmt::Display for FrameFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let channel_str = match self.channels {
            1 => "mono",
            2 => "stereo",
            n => return write!(f, "{}ch/{}Hz/{}f", n, self.sample_rate, self.buffer_size),
        };
        write!(f, "{}/{}Hz/{}f", channel_str, self.sample_rate, self.buffer_size)
    }
}

/// Audio buffer wrapper that enforces the frame format
#[derive(Debug)]
pub struct AudioBuffer {
    format: FrameFormat,
    data: Vec<Sample>,
}

impl AudioBuffer {
    /// Create a new audio buffer with the specified format
    pub fn new(format: FrameFormat) -> Self {
        let capacity = format.samples_per_buffer();
        Self {
            format,
            data: vec![0.0; capacity],
        }
    }

    /// Create buffer from existing data, validating format
    pub fn from_data(format: FrameFormat, data: Vec<Sample>) -> Result<Self> {
        if data.len() != format.samples_per_buffer() {
            return Err(anyhow::anyhow!(
                "Data length {} doesn't match format {} (expected {} samples)",
                data.len(), format, format.samples_per_buffer()
            ));
        }
        Ok(Self { format, data })
    }

    /// Get the frame format
    pub fn format(&self) -> FrameFormat {
        self.format
    }

    /// Get reference to the audio data
    pub fn data(&self) -> &[Sample] {
        &self.data
    }

    /// Get mutable reference to the audio data
    pub fn data_mut(&mut self) -> &mut [Sample] {
        &mut self.data
    }

    /// Clear the buffer (fill with zeros)
    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    /// Copy data from another buffer, converting format if necessary
    pub fn copy_from(&mut self, other: &AudioBuffer) -> Result<()> {
        if self.format.is_compatible_with(&other.format) {
            if self.format.buffer_size == other.format.buffer_size {
                // Direct copy for identical formats
                self.data.copy_from_slice(&other.data);
            } else {
                // Resize/truncate for different buffer sizes
                let copy_len = self.data.len().min(other.data.len());
                self.data[..copy_len].copy_from_slice(&other.data[..copy_len]);
                if self.data.len() > copy_len {
                    self.data[copy_len..].fill(0.0);
                }
            }
        } else {
            // Format conversion required
            self.convert_from(other)?;
        }
        Ok(())
    }

    /// Convert from another buffer with potentially different format
    fn convert_from(&mut self, other: &AudioBuffer) -> Result<()> {
        // Simple format conversion - in a real implementation, this would
        // include sample rate conversion, channel mixing, etc.

        if self.format.sample_rate != other.format.sample_rate {
            return Err(anyhow::anyhow!(
                "Sample rate conversion not yet implemented ({} Hz -> {} Hz)",
                other.format.sample_rate, self.format.sample_rate
            ));
        }

        match (other.format.channels, self.format.channels) {
            (1, 1) => {
                // Mono to mono - just copy/resize
                let copy_len = self.data.len().min(other.data.len());
                self.data[..copy_len].copy_from_slice(&other.data[..copy_len]);
            }
            (1, 2) => {
                // Mono to stereo - duplicate mono channel
                let frames = self.format.buffer_size.min(other.format.buffer_size);
                for i in 0..frames {
                    let mono_sample = other.data[i];
                    self.data[i * 2] = mono_sample;
                    self.data[i * 2 + 1] = mono_sample;
                }
            }
            (2, 1) => {
                // Stereo to mono - mix channels
                let frames = self.format.buffer_size.min(other.format.buffer_size);
                for i in 0..frames {
                    let left = other.data[i * 2];
                    let right = other.data[i * 2 + 1];
                    self.data[i] = (left + right) * 0.5;
                }
            }
            (2, 2) => {
                // Stereo to stereo - direct copy/resize
                let copy_samples = self.data.len().min(other.data.len());
                self.data[..copy_samples].copy_from_slice(&other.data[..copy_samples]);
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Channel conversion not supported ({} ch -> {} ch)",
                    other.format.channels, self.format.channels
                ));
            }
        }

        Ok(())
    }

    /// Get samples for a specific channel
    pub fn channel_samples(&self, channel: u8) -> Result<Vec<Sample>> {
        if channel >= self.format.channels {
            return Err(anyhow::anyhow!("Channel {} out of range (max {})",
                                     channel, self.format.channels - 1));
        }

        let mut samples = Vec::with_capacity(self.format.buffer_size);
        for frame in 0..self.format.buffer_size {
            let sample_idx = frame * self.format.channels as usize + channel as usize;
            samples.push(self.data[sample_idx]);
        }
        Ok(samples)
    }

    /// Set samples for a specific channel
    pub fn set_channel_samples(&mut self, channel: u8, samples: &[Sample]) -> Result<()> {
        if channel >= self.format.channels {
            return Err(anyhow::anyhow!("Channel {} out of range (max {})",
                                     channel, self.format.channels - 1));
        }

        if samples.len() != self.format.buffer_size {
            return Err(anyhow::anyhow!("Sample count {} doesn't match buffer size {}",
                                     samples.len(), self.format.buffer_size));
        }

        for frame in 0..self.format.buffer_size {
            let sample_idx = frame * self.format.channels as usize + channel as usize;
            self.data[sample_idx] = samples[frame];
        }
        Ok(())
    }

    /// Calculate RMS level for all channels
    pub fn rms_level(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = self.data.iter().map(|&x| x * x).sum();
        (sum_squares / self.data.len() as f32).sqrt()
    }

    /// Calculate peak level for all channels
    pub fn peak_level(&self) -> f32 {
        self.data.iter().fold(0.0, |max, &x| max.max(x.abs()))
    }
}

/// Profile-specific format recommendations
impl FrameFormat {
    /// Get recommended format for Balanced profile
    pub fn balanced() -> Self {
        Self {
            channels: 1,
            sample_rate: 48000,
            buffer_size: 128,  // ~2.7ms latency
        }
    }

    /// Get recommended format for Streaming profile
    pub fn streaming() -> Self {
        Self {
            channels: 1,
            sample_rate: 48000,
            buffer_size: 256,  // ~5.3ms latency, good for encoding
        }
    }

    /// Get recommended format for Studio profile
    pub fn studio() -> Self {
        Self {
            channels: 2,        // Stereo for studio work
            sample_rate: 96000, // High sample rate for quality
            buffer_size: 256,   // ~2.7ms latency at 96kHz
        }
    }

    /// Get format optimized for ultra-low latency
    pub fn ultra_low_latency() -> Self {
        Self {
            channels: 1,
            sample_rate: 48000,
            buffer_size: 32,    // ~0.67ms latency
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_format_validation() {
        // Valid formats
        assert!(FrameFormat::new(1, 48000, 128).is_ok());
        assert!(FrameFormat::new(2, 44100, 256).is_ok());

        // Invalid channel counts
        assert!(FrameFormat::new(0, 48000, 128).is_err());
        assert!(FrameFormat::new(9, 48000, 128).is_err());

        // Invalid sample rates
        assert!(FrameFormat::new(1, 12000, 128).is_err());
        assert!(FrameFormat::new(1, 50000, 128).is_err());

        // Invalid buffer sizes
        assert!(FrameFormat::new(1, 48000, 31).is_err());   // Too small
        assert!(FrameFormat::new(1, 48000, 100).is_err());  // Not power of 2
        assert!(FrameFormat::new(1, 48000, 8192).is_err()); // Too large
    }

    #[test]
    fn test_format_calculations() {
        let format = FrameFormat::new(2, 48000, 128).unwrap();

        assert_eq!(format.buffer_duration_ms(), 128.0 / 48000.0 * 1000.0);
        assert_eq!(format.samples_per_buffer(), 256); // 128 frames * 2 channels
        assert_eq!(format.buffer_size_bytes(), 256 * 4); // 256 samples * 4 bytes
        assert_eq!(format.nyquist_frequency(), 24000.0);
    }

    #[test]
    fn test_format_compatibility() {
        let format1 = FrameFormat::new(1, 48000, 128).unwrap();
        let format2 = FrameFormat::new(1, 48000, 256).unwrap(); // Different buffer size
        let format3 = FrameFormat::new(2, 48000, 128).unwrap(); // Different channels

        assert!(format1.is_compatible_with(&format2));
        assert!(!format1.is_compatible_with(&format3));
    }

    #[test]
    fn test_audio_buffer() {
        let format = FrameFormat::mono(48000, 128).unwrap();
        let mut buffer = AudioBuffer::new(format);

        assert_eq!(buffer.data().len(), 128);
        assert_eq!(buffer.format().channels, 1);

        // Test clearing
        buffer.data_mut()[0] = 1.0;
        buffer.clear();
        assert_eq!(buffer.data()[0], 0.0);

        // Test levels
        buffer.data_mut().fill(0.5);
        assert!((buffer.rms_level() - 0.5).abs() < 1e-6);
        assert_eq!(buffer.peak_level(), 0.5);
    }

    #[test]
    fn test_channel_operations() {
        let format = FrameFormat::stereo(48000, 4).unwrap();
        let mut buffer = AudioBuffer::new(format);

        // Set left channel data
        let left_samples = vec![1.0, 2.0, 3.0, 4.0];
        buffer.set_channel_samples(0, &left_samples).unwrap();

        // Set right channel data
        let right_samples = vec![0.1, 0.2, 0.3, 0.4];
        buffer.set_channel_samples(1, &right_samples).unwrap();

        // Verify interleaved layout
        assert_eq!(buffer.data(), &[1.0, 0.1, 2.0, 0.2, 3.0, 0.3, 4.0, 0.4]);

        // Extract channel data
        assert_eq!(buffer.channel_samples(0).unwrap(), left_samples);
        assert_eq!(buffer.channel_samples(1).unwrap(), right_samples);
    }

    #[test]
    fn test_format_conversion() {
        // Mono to stereo
        let mono_format = FrameFormat::mono(48000, 2).unwrap();
        let mut mono_buffer = AudioBuffer::new(mono_format);
        mono_buffer.data_mut()[0] = 1.0;
        mono_buffer.data_mut()[1] = 2.0;

        let stereo_format = FrameFormat::stereo(48000, 2).unwrap();
        let mut stereo_buffer = AudioBuffer::new(stereo_format);

        stereo_buffer.copy_from(&mono_buffer).unwrap();
        assert_eq!(stereo_buffer.data(), &[1.0, 1.0, 2.0, 2.0]);

        // Stereo to mono
        let mut mono_buffer2 = AudioBuffer::new(mono_format);
        stereo_buffer.data_mut()[0] = 1.0; // Left
        stereo_buffer.data_mut()[1] = 3.0; // Right
        stereo_buffer.data_mut()[2] = 2.0; // Left
        stereo_buffer.data_mut()[3] = 4.0; // Right

        mono_buffer2.copy_from(&stereo_buffer).unwrap();
        assert_eq!(mono_buffer2.data(), &[2.0, 3.0]); // Mixed: (1+3)/2, (2+4)/2
    }

    #[test]
    fn test_profile_formats() {
        let balanced = FrameFormat::balanced();
        assert_eq!(balanced.channels, 1);
        assert_eq!(balanced.sample_rate, 48000);
        assert_eq!(balanced.buffer_size, 128);

        let streaming = FrameFormat::streaming();
        assert_eq!(streaming.buffer_size, 256);

        let studio = FrameFormat::studio();
        assert_eq!(studio.channels, 2);
        assert_eq!(studio.sample_rate, 96000);

        let ultra = FrameFormat::ultra_low_latency();
        assert_eq!(ultra.buffer_size, 32);
    }
}