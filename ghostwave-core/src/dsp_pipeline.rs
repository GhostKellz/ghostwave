//! # GhostWave DSP Pipeline
//!
//! Complete audio processing pipeline implementing:
//! HPF → VAD → spectral denoise → expander/gate → soft-clip/limiter
//!
//! This module provides the core DSP chain that transforms raw microphone input
//! into clean, processed audio suitable for streaming or recording.

use anyhow::Result;
use crate::frame_format::{FrameFormat, Sample};
use crate::processor::{ProcessingProfile, ParamValue};
use std::collections::VecDeque;

/// High-pass filter to remove low-frequency noise and rumble
#[derive(Debug)]
pub struct HighPassFilter {
    cutoff_hz: f32,
    sample_rate: f32,
    x1: f32,
    y1: f32,
    a: f32,
    b: f32,
}

impl HighPassFilter {
    pub fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let mut filter = Self {
            cutoff_hz,
            sample_rate,
            x1: 0.0,
            y1: 0.0,
            a: 0.0,
            b: 0.0,
        };
        filter.calculate_coefficients();
        filter
    }

    fn calculate_coefficients(&mut self) {
        let omega = 2.0 * std::f32::consts::PI * self.cutoff_hz / self.sample_rate;
        let alpha = omega / (omega + 1.0);
        self.a = alpha;
        self.b = alpha;
    }

    pub fn set_cutoff(&mut self, cutoff_hz: f32) {
        self.cutoff_hz = cutoff_hz;
        self.calculate_coefficients();
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let output = self.a * (input - self.x1) + self.b * self.y1;
        self.x1 = input;
        self.y1 = output;
        output
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }
}

/// Voice Activity Detector to distinguish speech from background noise
#[derive(Debug)]
pub struct VoiceActivityDetector {
    energy_threshold: f32,
    zcr_threshold: f32,
    energy_history: VecDeque<f32>,
    history_size: usize,
    is_voice_active: bool,
    hangover_frames: usize,
    current_hangover: usize,
}

impl VoiceActivityDetector {
    pub fn new(sample_rate: f32, frame_size: usize) -> Self {
        Self {
            energy_threshold: -40.0, // dB
            zcr_threshold: 0.1,
            energy_history: VecDeque::with_capacity(20),
            history_size: 20,
            is_voice_active: false,
            hangover_frames: (sample_rate * 0.2) as usize / frame_size, // 200ms hangover
            current_hangover: 0,
        }
    }

    pub fn process(&mut self, buffer: &[Sample]) -> bool {
        let energy_db = self.calculate_energy_db(buffer);
        let zcr = self.calculate_zero_crossing_rate(buffer);

        // Update energy history
        self.energy_history.push_back(energy_db);
        if self.energy_history.len() > self.history_size {
            self.energy_history.pop_front();
        }

        // Adaptive threshold based on recent energy
        let avg_energy = self.energy_history.iter().sum::<f32>() / self.energy_history.len() as f32;
        let adaptive_threshold = avg_energy + 6.0; // 6dB above average

        let voice_detected = energy_db > adaptive_threshold.max(self.energy_threshold) && zcr < self.zcr_threshold;

        if voice_detected {
            self.is_voice_active = true;
            self.current_hangover = self.hangover_frames;
        } else if self.current_hangover > 0 {
            self.current_hangover -= 1;
            self.is_voice_active = true;
        } else {
            self.is_voice_active = false;
        }

        self.is_voice_active
    }

    fn calculate_energy_db(&self, buffer: &[Sample]) -> f32 {
        if buffer.is_empty() {
            return -80.0;
        }

        let rms = (buffer.iter().map(|&x| x * x).sum::<f32>() / buffer.len() as f32).sqrt();
        20.0 * rms.max(1e-10).log10()
    }

    fn calculate_zero_crossing_rate(&self, buffer: &[Sample]) -> f32 {
        if buffer.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..buffer.len() {
            if (buffer[i] >= 0.0) != (buffer[i-1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (buffer.len() - 1) as f32
    }

    pub fn is_voice_active(&self) -> bool {
        self.is_voice_active
    }

    pub fn set_sensitivity(&mut self, sensitivity: f32) {
        // sensitivity: 0.0 = very sensitive, 1.0 = less sensitive
        self.energy_threshold = -60.0 + sensitivity * 30.0; // Range: -60dB to -30dB
        self.zcr_threshold = 0.05 + sensitivity * 0.15; // Range: 0.05 to 0.2
    }
}

/// Spectral noise reduction using simple spectral subtraction
#[derive(Debug)]
pub struct SpectralDenoiser {
    strength: f32,
    noise_estimate: Vec<f32>,
    over_subtraction: f32,
}

impl SpectralDenoiser {
    pub fn new(_sample_rate: f32, frame_size: usize) -> Self {
        Self {
            strength: 0.7,
            noise_estimate: vec![0.0; frame_size / 2 + 1],
            over_subtraction: 2.0,
        }
    }

    pub fn process(&mut self, buffer: &mut [Sample], voice_active: bool) -> Result<()> {
        // Simple time-domain noise reduction for now
        // In a full implementation, this would use FFT-based spectral subtraction

        if !voice_active {
            // Update noise estimate when no voice is detected
            let energy = buffer.iter().map(|&x| x * x).sum::<f32>() / buffer.len() as f32;
            let noise_level = energy.sqrt();

            // Simple noise floor estimation
            if self.noise_estimate.is_empty() || noise_level < self.noise_estimate[0] {
                self.noise_estimate = vec![noise_level; 1];
            }
        } else if !self.noise_estimate.is_empty() {
            // Apply noise reduction when voice is active
            let noise_level = self.noise_estimate[0];
            let threshold = noise_level * self.over_subtraction;

            for sample in buffer.iter_mut() {
                let magnitude = sample.abs();
                if magnitude > threshold {
                    let reduction_factor = 1.0 - (self.strength * threshold / magnitude);
                    *sample *= reduction_factor.max(0.1); // Minimum 10% of original
                } else {
                    *sample *= 0.1; // Heavily attenuate below threshold
                }
            }
        }

        Ok(())
    }

    pub fn set_strength(&mut self, strength: f32) {
        self.strength = strength.clamp(0.0, 1.0);
    }
}

/// Expander/Gate to remove quiet background noise
#[derive(Debug)]
pub struct ExpanderGate {
    threshold_db: f32,
    ratio: f32,
    attack_samples: usize,
    release_samples: usize,
    envelope: f32,
    gain_reduction: f32,
}

impl ExpanderGate {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            threshold_db: -45.0,
            ratio: 4.0, // 4:1 expansion below threshold
            attack_samples: (sample_rate * 0.001) as usize, // 1ms attack
            release_samples: (sample_rate * 0.1) as usize,  // 100ms release
            envelope: 0.0,
            gain_reduction: 1.0,
        }
    }

    pub fn process(&mut self, buffer: &mut [Sample]) {
        for sample in buffer.iter_mut() {
            let input_level = sample.abs();

            // Envelope follower
            if input_level > self.envelope {
                // Attack
                let attack_coeff = 1.0 / self.attack_samples as f32;
                self.envelope += (input_level - self.envelope) * attack_coeff;
            } else {
                // Release
                let release_coeff = 1.0 / self.release_samples as f32;
                self.envelope += (input_level - self.envelope) * release_coeff;
            }

            // Convert to dB for threshold comparison
            let envelope_db = 20.0 * self.envelope.max(1e-10).log10();

            // Calculate gain reduction
            if envelope_db < self.threshold_db {
                let over_threshold = envelope_db - self.threshold_db;
                let expanded_over = over_threshold * (self.ratio - 1.0) / self.ratio;
                let target_gain_db = expanded_over;
                let target_gain = 10.0_f32.powf(target_gain_db / 20.0);

                // Smooth gain changes
                let smoothing = 0.01;
                self.gain_reduction += (target_gain - self.gain_reduction) * smoothing;
            } else {
                // Above threshold - no reduction
                let smoothing = 0.01;
                self.gain_reduction += (1.0 - self.gain_reduction) * smoothing;
            }

            *sample *= self.gain_reduction;
        }
    }

    pub fn set_threshold(&mut self, threshold_db: f32) {
        self.threshold_db = threshold_db;
    }

    pub fn set_ratio(&mut self, ratio: f32) {
        self.ratio = ratio.max(1.0);
    }
}

/// Soft clipper and limiter to prevent harsh distortion
#[derive(Debug)]
pub struct SoftLimiter {
    threshold: f32,
    knee_width: f32,
    makeup_gain: f32,
    lookahead_samples: usize,
    delay_buffer: VecDeque<Sample>,
}

impl SoftLimiter {
    pub fn new(sample_rate: f32) -> Self {
        let lookahead_ms = 5.0; // 5ms lookahead
        let lookahead_samples = (sample_rate * lookahead_ms / 1000.0) as usize;

        Self {
            threshold: 0.95,
            knee_width: 0.1,
            makeup_gain: 1.0,
            lookahead_samples,
            delay_buffer: VecDeque::with_capacity(lookahead_samples),
        }
    }

    pub fn process(&mut self, buffer: &mut [Sample]) {
        for sample in buffer.iter_mut() {
            // Add to delay buffer
            if self.delay_buffer.len() >= self.lookahead_samples {
                self.delay_buffer.pop_front();
            }
            self.delay_buffer.push_back(*sample);

            // Process delayed sample
            if let Some(&delayed_sample) = self.delay_buffer.front() {
                let processed = self.apply_limiting(delayed_sample);
                *sample = processed * self.makeup_gain;
            }
        }
    }

    fn apply_limiting(&mut self, input: Sample) -> Sample {
        let input_abs = input.abs();

        // Soft knee compression/limiting
        let knee_start = self.threshold - self.knee_width / 2.0;
        let knee_end = self.threshold + self.knee_width / 2.0;

        let gain = if input_abs <= knee_start {
            1.0
        } else if input_abs >= knee_end {
            self.threshold / input_abs
        } else {
            // Smooth transition in knee region
            let knee_ratio = (input_abs - knee_start) / self.knee_width;
            let knee_gain = 1.0 - knee_ratio * (1.0 - self.threshold / input_abs);
            knee_gain
        };

        input * gain
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.1, 1.0);
    }

    pub fn set_makeup_gain(&mut self, gain_db: f32) {
        self.makeup_gain = 10.0_f32.powf(gain_db / 20.0);
    }
}

/// Complete DSP pipeline combining all processing stages
#[derive(Debug)]
pub struct DspPipeline {
    format: FrameFormat,
    profile: ProcessingProfile,

    // Processing stages
    highpass_filter: HighPassFilter,
    voice_detector: VoiceActivityDetector,
    spectral_denoiser: SpectralDenoiser,
    expander_gate: ExpanderGate,
    soft_limiter: SoftLimiter,

    // State
    enabled: bool,
    bypass_vad: bool,
}

impl DspPipeline {
    pub fn new(format: FrameFormat, profile: ProcessingProfile) -> Self {
        let sample_rate = format.sample_rate as f32;
        let buffer_size = format.buffer_size;

        let mut pipeline = Self {
            format,
            profile,
            highpass_filter: HighPassFilter::new(80.0, sample_rate),
            voice_detector: VoiceActivityDetector::new(sample_rate, buffer_size),
            spectral_denoiser: SpectralDenoiser::new(sample_rate, buffer_size),
            expander_gate: ExpanderGate::new(sample_rate),
            soft_limiter: SoftLimiter::new(sample_rate),
            enabled: true,
            bypass_vad: false,
        };

        pipeline.apply_profile_settings();
        pipeline
    }

    /// Apply profile-specific settings to all DSP components
    fn apply_profile_settings(&mut self) {
        match self.profile {
            ProcessingProfile::Balanced => {
                self.highpass_filter.set_cutoff(80.0);
                self.voice_detector.set_sensitivity(0.5);
                self.spectral_denoiser.set_strength(0.7);
                self.expander_gate.set_threshold(-45.0);
                self.expander_gate.set_ratio(3.0);
                self.soft_limiter.set_threshold(0.95);
                self.soft_limiter.set_makeup_gain(0.0);
            }
            ProcessingProfile::Streaming => {
                self.highpass_filter.set_cutoff(100.0);
                self.voice_detector.set_sensitivity(0.3); // More aggressive
                self.spectral_denoiser.set_strength(0.85);
                self.expander_gate.set_threshold(-40.0);
                self.expander_gate.set_ratio(4.0);
                self.soft_limiter.set_threshold(0.90);
                self.soft_limiter.set_makeup_gain(1.0);
            }
            ProcessingProfile::Studio => {
                self.highpass_filter.set_cutoff(40.0); // Less aggressive
                self.voice_detector.set_sensitivity(0.7);
                self.spectral_denoiser.set_strength(0.3); // Minimal processing
                self.expander_gate.set_threshold(-60.0);
                self.expander_gate.set_ratio(2.0);
                self.soft_limiter.set_threshold(0.98);
                self.soft_limiter.set_makeup_gain(-1.0); // Slight reduction
            }
        }
    }

    /// Process audio buffer through the complete DSP pipeline
    pub fn process(&mut self, buffer: &mut [Sample]) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // Ensure buffer is the right size
        if buffer.len() != self.format.samples_per_buffer() {
            return Err(anyhow::anyhow!(
                "Buffer size mismatch: expected {}, got {}",
                self.format.samples_per_buffer(), buffer.len()
            ));
        }

        // Stage 1: High-pass filtering
        for sample in buffer.iter_mut() {
            *sample = self.highpass_filter.process(*sample);
        }

        // Stage 2: Voice Activity Detection
        let voice_active = if self.bypass_vad {
            true
        } else {
            self.voice_detector.process(buffer)
        };

        // Stage 3: Spectral noise reduction
        self.spectral_denoiser.process(buffer, voice_active)?;

        // Stage 4: Expander/Gate
        if voice_active || self.profile == ProcessingProfile::Studio {
            self.expander_gate.process(buffer);
        } else {
            // Heavy attenuation when no voice detected
            for sample in buffer.iter_mut() {
                *sample *= 0.05;
            }
        }

        // Stage 5: Soft limiting
        self.soft_limiter.process(buffer);

        Ok(())
    }

    /// Set processing parameter by name
    pub fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()> {
        match name {
            "highpass_frequency" => {
                if let ParamValue::Float(freq) = value {
                    self.highpass_filter.set_cutoff(freq);
                } else {
                    return Err(anyhow::anyhow!("Expected float value for highpass_frequency"));
                }
            }
            "noise_reduction_strength" => {
                if let ParamValue::Float(strength) = value {
                    self.spectral_denoiser.set_strength(strength);
                } else {
                    return Err(anyhow::anyhow!("Expected float value for noise_reduction_strength"));
                }
            }
            "gate_threshold" => {
                if let ParamValue::Float(threshold) = value {
                    self.expander_gate.set_threshold(threshold);
                } else {
                    return Err(anyhow::anyhow!("Expected float value for gate_threshold"));
                }
            }
            "limiter_threshold" => {
                if let ParamValue::Float(threshold) = value {
                    self.soft_limiter.set_threshold(threshold);
                } else {
                    return Err(anyhow::anyhow!("Expected float value for limiter_threshold"));
                }
            }
            "vad_sensitivity" => {
                if let ParamValue::Float(sensitivity) = value {
                    self.voice_detector.set_sensitivity(sensitivity);
                } else {
                    return Err(anyhow::anyhow!("Expected float value for vad_sensitivity"));
                }
            }
            "bypass_vad" => {
                if let ParamValue::Bool(bypass) = value {
                    self.bypass_vad = bypass;
                } else {
                    return Err(anyhow::anyhow!("Expected bool value for bypass_vad"));
                }
            }
            _ => return Err(anyhow::anyhow!("Unknown parameter: {}", name)),
        }
        Ok(())
    }

    /// Get current processing parameter
    pub fn get_param(&self, name: &str) -> Result<ParamValue> {
        match name {
            "highpass_frequency" => Ok(ParamValue::Float(self.highpass_filter.cutoff_hz)),
            "noise_reduction_strength" => Ok(ParamValue::Float(self.spectral_denoiser.strength)),
            "gate_threshold" => Ok(ParamValue::Float(self.expander_gate.threshold_db)),
            "limiter_threshold" => Ok(ParamValue::Float(self.soft_limiter.threshold)),
            "bypass_vad" => Ok(ParamValue::Bool(self.bypass_vad)),
            _ => Err(anyhow::anyhow!("Unknown parameter: {}", name)),
        }
    }

    /// Set processing profile and apply settings
    pub fn set_profile(&mut self, profile: ProcessingProfile) {
        self.profile = profile;
        self.apply_profile_settings();
    }

    pub fn get_profile(&self) -> ProcessingProfile {
        self.profile
    }

    /// Enable/disable entire pipeline
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Check if voice is currently detected
    pub fn is_voice_active(&self) -> bool {
        self.voice_detector.is_voice_active()
    }

    /// Reset all processing stages
    pub fn reset(&mut self) {
        self.highpass_filter.reset();
        // Other components don't have explicit reset methods yet
    }

    /// Get processing latency in frames
    pub fn latency_frames(&self) -> usize {
        // HPF: ~0 frames
        // VAD: 0 frames (current frame analysis)
        // Spectral denoiser: 0 frames (simple version)
        // Expander: 0 frames
        // Limiter: lookahead samples
        self.soft_limiter.lookahead_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highpass_filter() {
        let mut hpf = HighPassFilter::new(100.0, 48000.0);

        // Test DC removal
        let dc_signal = vec![1.0; 100];
        let filtered: Vec<f32> = dc_signal.iter().map(|&x| hpf.process(x)).collect();

        // After settling, DC should be mostly removed
        let final_average = filtered[50..].iter().sum::<f32>() / 50.0;
        assert!(final_average.abs() < 0.1);
    }

    #[test]
    fn test_voice_activity_detector() {
        let mut vad = VoiceActivityDetector::new(48000.0, 256);

        // Silence should not trigger VAD
        let silence = vec![0.0; 256];
        assert!(!vad.process(&silence));

        // Loud signal should trigger VAD
        let speech = vec![0.5; 256];
        assert!(vad.process(&speech));
    }

    #[test]
    fn test_dsp_pipeline() {
        let format = FrameFormat::balanced();
        let mut pipeline = DspPipeline::new(format, ProcessingProfile::Balanced);

        let mut buffer = vec![0.1; format.samples_per_buffer()];
        assert!(pipeline.process(&mut buffer).is_ok());

        // Test parameter setting
        assert!(pipeline.set_param("highpass_frequency", ParamValue::Float(120.0)).is_ok());
        assert_eq!(pipeline.get_param("highpass_frequency").unwrap(), ParamValue::Float(120.0));
    }

    #[test]
    fn test_profile_switching() {
        let format = FrameFormat::balanced();
        let mut pipeline = DspPipeline::new(format, ProcessingProfile::Balanced);

        // Switch to streaming profile
        pipeline.set_profile(ProcessingProfile::Streaming);
        assert_eq!(pipeline.get_profile(), ProcessingProfile::Streaming);

        // Verify parameters changed
        let hpf_freq = pipeline.get_param("highpass_frequency").unwrap();
        assert_eq!(hpf_freq, ParamValue::Float(100.0)); // Streaming profile setting
    }
}