//! Adapter wrapping the `nnnoiseless` crate for integration with GhostWave's DSP pipeline.
//!
//! Handles frame-size alignment (480 samples required by nnnoiseless vs arbitrary pipeline
//! buffer sizes), sample format scaling ([-1,1] normalized ↔ [-32768,32767] i16-scale),
//! and first-frame warmup artifacts.

use anyhow::Result;
use nnnoiseless::DenoiseState;

/// nnnoiseless requires exactly 480 samples per frame at 48kHz.
const NNNOISE_FRAME_SIZE: usize = 480;

/// Scale factor: nnnoiseless expects i16-range floats.
const SCALE_IN: f32 = 32768.0;
const SCALE_OUT: f32 = 1.0 / 32768.0;

/// Adapter that bridges nnnoiseless's fixed 480-sample frame processing to
/// GhostWave's variable-size buffer pipeline.
pub struct NnnoiseDenoiser {
    state: Box<DenoiseState<'static>>,

    // Ring buffer for accumulating input samples until we have a full 480-sample frame
    input_ring: Vec<f32>,
    input_pos: usize,

    // Ring buffer for denoised output samples waiting to be consumed
    output_ring: Vec<f32>,
    output_read: usize,
    output_write: usize,
    output_count: usize,

    // VAD probability from most recent frame
    vad_probability: f32,

    // First frame produces warmup artifacts — discard its output
    frames_processed: usize,

    // Blend between denoised and original (0.0 = bypass, 1.0 = full denoise)
    strength: f32,
}

impl NnnoiseDenoiser {
    pub fn new() -> Self {
        // Allocate enough output ring buffer to handle large pipeline buffers.
        // 4800 = 10 nnnoiseless frames = 100ms at 48kHz, more than any reasonable
        // pipeline buffer (typically 128-1024 samples).
        let ring_capacity = NNNOISE_FRAME_SIZE * 10;

        Self {
            state: DenoiseState::new(),
            input_ring: vec![0.0; NNNOISE_FRAME_SIZE],
            input_pos: 0,
            output_ring: vec![0.0; ring_capacity],
            output_read: 0,
            output_write: 0,
            output_count: 0,
            vad_probability: 0.0,
            frames_processed: 0,
            strength: 1.0,
        }
    }

    /// Process audio in-place. Accepts any buffer size — internally accumulates
    /// to 480-sample frames for nnnoiseless, then drains denoised output back.
    ///
    /// During startup (before the first full frame is processed), input is passed
    /// through with gentle attenuation to avoid silence gaps.
    pub fn process(&mut self, buffer: &mut [f32], _voice_active: bool) -> Result<()> {
        // Keep a copy of the original for strength blending
        let original: Vec<f32> = if self.strength < 1.0 {
            buffer.to_vec()
        } else {
            Vec::new()
        };

        for sample in buffer.iter_mut() {
            // Feed input sample (scaled to i16 range) into the accumulation buffer
            self.input_ring[self.input_pos] = *sample * SCALE_IN;
            self.input_pos += 1;

            // When we have a full frame, process it
            if self.input_pos >= NNNOISE_FRAME_SIZE {
                self.process_one_frame();
                self.input_pos = 0;
            }

            // Drain denoised output if available
            if self.output_count > 0 {
                let denoised = self.output_ring[self.output_read] * SCALE_OUT;
                self.output_read = (self.output_read + 1) % self.output_ring.len();
                self.output_count -= 1;
                *sample = denoised;
            } else {
                // Startup latency: no denoised output yet, pass through attenuated
                *sample *= 0.5;
            }
        }

        // Blend with original based on strength
        if self.strength < 1.0 && !original.is_empty() {
            let dry = 1.0 - self.strength;
            for (i, sample) in buffer.iter_mut().enumerate() {
                *sample = *sample * self.strength + original[i] * dry;
            }
        }

        Ok(())
    }

    /// Process one accumulated 480-sample frame through nnnoiseless.
    fn process_one_frame(&mut self) {
        let mut output_frame = [0.0f32; NNNOISE_FRAME_SIZE];

        let vad = self.state.process_frame(&mut output_frame, &self.input_ring);
        self.vad_probability = vad;
        self.frames_processed += 1;

        // Discard first frame output (warmup artifacts), push silence instead
        if self.frames_processed == 1 {
            output_frame.fill(0.0);
        }

        // Push denoised output into ring buffer
        let cap = self.output_ring.len();
        for &sample in &output_frame {
            if self.output_count < cap {
                self.output_ring[self.output_write] = sample;
                self.output_write = (self.output_write + 1) % cap;
                self.output_count += 1;
            }
            // If ring is full, we drop samples (shouldn't happen with proper sizing)
        }
    }

    /// VAD probability from the most recently processed frame (0.0 to ~1.0).
    pub fn vad_probability(&self) -> f32 {
        self.vad_probability
    }

    /// Set denoise strength (0.0 = bypass / dry, 1.0 = full denoise).
    pub fn set_strength(&mut self, strength: f32) {
        self.strength = strength.clamp(0.0, 1.0);
    }

    /// Get current denoise strength.
    pub fn strength(&self) -> f32 {
        self.strength
    }

    /// Reset internal state for a clean start (e.g., after sample rate change).
    pub fn reset(&mut self) {
        self.state = DenoiseState::new();
        self.input_pos = 0;
        self.output_read = 0;
        self.output_write = 0;
        self.output_count = 0;
        self.vad_probability = 0.0;
        self.frames_processed = 0;
    }

    /// Number of frames processed so far.
    pub fn frames_processed(&self) -> usize {
        self.frames_processed
    }

    /// Latency introduced by the frame accumulation, in samples.
    /// One full nnnoiseless frame (480 samples = 10ms at 48kHz).
    pub fn latency_samples(&self) -> usize {
        NNNOISE_FRAME_SIZE
    }
}

impl Default for NnnoiseDenoiser {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for NnnoiseDenoiser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NnnoiseDenoiser")
            .field("input_pos", &self.input_pos)
            .field("output_count", &self.output_count)
            .field("vad_probability", &self.vad_probability)
            .field("frames_processed", &self.frames_processed)
            .field("strength", &self.strength)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_denoiser() {
        let d = NnnoiseDenoiser::new();
        assert_eq!(d.frames_processed(), 0);
        assert_eq!(d.vad_probability(), 0.0);
        assert!((d.strength() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_process_small_buffer() {
        let mut d = NnnoiseDenoiser::new();
        let mut buf = vec![0.1; 128];
        assert!(d.process(&mut buf, true).is_ok());
    }

    #[test]
    fn test_process_exact_frame() {
        let mut d = NnnoiseDenoiser::new();
        let mut buf = vec![0.05; 480];
        assert!(d.process(&mut buf, true).is_ok());
        assert_eq!(d.frames_processed(), 1);
    }

    #[test]
    fn test_process_large_buffer() {
        let mut d = NnnoiseDenoiser::new();
        let mut buf = vec![0.05; 1024];
        assert!(d.process(&mut buf, true).is_ok());
        assert!(d.frames_processed() >= 2);
    }

    #[test]
    fn test_strength_blend() {
        let mut d = NnnoiseDenoiser::new();
        d.set_strength(0.0);

        // With strength 0.0, output should match input (bypass)
        let input = vec![0.3; 128];
        let mut buf = input.clone();
        d.process(&mut buf, true).unwrap();

        // During startup (no denoised frames yet), bypass means: attenuated * 0 + original * 1
        for (i, &sample) in buf.iter().enumerate() {
            assert!(
                (sample - input[i]).abs() < 0.01,
                "sample {} diverged: {} vs {}",
                i,
                sample,
                input[i]
            );
        }
    }

    #[test]
    fn test_reset() {
        let mut d = NnnoiseDenoiser::new();
        let mut buf = vec![0.1; 960];
        d.process(&mut buf, true).unwrap();
        assert!(d.frames_processed() > 0);

        d.reset();
        assert_eq!(d.frames_processed(), 0);
        assert_eq!(d.vad_probability(), 0.0);
    }

    #[test]
    fn test_latency_samples() {
        let d = NnnoiseDenoiser::new();
        assert_eq!(d.latency_samples(), 480);
    }
}
