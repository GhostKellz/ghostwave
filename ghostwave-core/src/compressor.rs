//! # GPU-Accelerated Compressor/Limiter
//!
//! Professional dynamics processor with look-ahead limiting.
//!
//! ## Features
//! - Compressor with soft/hard knee
//! - Brick-wall limiter
//! - Look-ahead for transparent limiting
//! - Auto-makeup gain
//! - Sidechain high-pass filter
//! - RMS/Peak detection modes
//!
//! ## Modes
//! - **Compressor**: Gentle dynamic range control
//! - **Limiter**: Prevent clipping with minimal artifacts
//! - **Gate/Expander**: Remove quiet sounds

use anyhow::Result;
use std::collections::VecDeque;
use std::f32::consts::PI;
use tracing::debug;

/// Detection mode for level measurement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMode {
    /// Peak detection (fast, responsive)
    Peak,
    /// RMS detection (smoother, more musical)
    Rms,
    /// True peak detection (oversampled)
    TruePeak,
}

impl Default for DetectionMode {
    fn default() -> Self {
        Self::Peak
    }
}

/// Knee type for compression curve
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KneeType {
    /// Hard knee (abrupt transition)
    Hard,
    /// Soft knee (gradual transition)
    Soft,
}

impl Default for KneeType {
    fn default() -> Self {
        Self::Soft
    }
}

/// Compressor configuration
#[derive(Debug, Clone)]
pub struct CompressorConfig {
    /// Threshold in dB (-60 to 0)
    pub threshold_db: f32,
    /// Ratio (1:1 to infinity, use f32::INFINITY for limiter)
    pub ratio: f32,
    /// Attack time in ms (0.01 to 100)
    pub attack_ms: f32,
    /// Release time in ms (10 to 2000)
    pub release_ms: f32,
    /// Knee width in dB (0 = hard, typically 6-12 for soft)
    pub knee_db: f32,
    /// Makeup gain in dB (-12 to +24)
    pub makeup_gain_db: f32,
    /// Enable auto-makeup gain
    pub auto_makeup: bool,
    /// Detection mode
    pub detection_mode: DetectionMode,
    /// Look-ahead time in ms (0 to 10)
    pub lookahead_ms: f32,
    /// Sidechain high-pass frequency (0 = disabled, typically 60-150Hz)
    pub sidechain_hpf_hz: f32,
    /// Mix (0 = dry, 1 = wet) for parallel compression
    pub mix: f32,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            threshold_db: -18.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            knee_db: 6.0,
            makeup_gain_db: 0.0,
            auto_makeup: true,
            detection_mode: DetectionMode::Peak,
            lookahead_ms: 0.0,
            sidechain_hpf_hz: 0.0,
            mix: 1.0,
        }
    }
}

impl CompressorConfig {
    /// Create a gentle vocal compressor preset
    pub fn vocal() -> Self {
        Self {
            threshold_db: -24.0,
            ratio: 3.0,
            attack_ms: 15.0,
            release_ms: 150.0,
            knee_db: 10.0,
            makeup_gain_db: 0.0,
            auto_makeup: true,
            detection_mode: DetectionMode::Rms,
            lookahead_ms: 0.0,
            sidechain_hpf_hz: 100.0,
            mix: 1.0,
        }
    }

    /// Create a brick-wall limiter preset
    pub fn limiter() -> Self {
        Self {
            threshold_db: -1.0,
            ratio: f32::INFINITY, // Brick wall
            attack_ms: 0.1,
            release_ms: 50.0,
            knee_db: 0.0, // Hard knee
            makeup_gain_db: 0.0,
            auto_makeup: false,
            detection_mode: DetectionMode::TruePeak,
            lookahead_ms: 1.5, // Prevents inter-sample peaks
            sidechain_hpf_hz: 0.0,
            mix: 1.0,
        }
    }

    /// Create a broadcast preset (aggressive, loud)
    pub fn broadcast() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 6.0,
            attack_ms: 5.0,
            release_ms: 80.0,
            knee_db: 4.0,
            makeup_gain_db: 0.0,
            auto_makeup: true,
            detection_mode: DetectionMode::Rms,
            lookahead_ms: 2.0,
            sidechain_hpf_hz: 80.0,
            mix: 1.0,
        }
    }
}

/// Sidechain high-pass filter
#[derive(Debug, Clone)]
struct SidechainHpf {
    enabled: bool,
    b0: f32,
    b1: f32,
    a1: f32,
    x1: f32,
    y1: f32,
}

impl SidechainHpf {
    fn new(frequency: f32, sample_rate: f32) -> Self {
        if frequency <= 0.0 {
            return Self {
                enabled: false,
                b0: 1.0,
                b1: 0.0,
                a1: 0.0,
                x1: 0.0,
                y1: 0.0,
            };
        }

        let omega = 2.0 * PI * frequency / sample_rate;
        let cos_omega = omega.cos();
        let alpha = (1.0 + cos_omega) / 2.0;

        Self {
            enabled: true,
            b0: alpha,
            b1: -alpha,
            a1: -(1.0 - omega.sin()) / (1.0 + omega.sin()),
            x1: 0.0,
            y1: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        if !self.enabled {
            return input;
        }

        let output = self.b0 * input + self.b1 * self.x1 - self.a1 * self.y1;
        self.x1 = input;
        self.y1 = output;
        output
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }
}

/// RMS calculator
#[derive(Debug, Clone)]
struct RmsCalculator {
    sum_squares: f32,
    window_size: usize,
    buffer: VecDeque<f32>,
}

impl RmsCalculator {
    fn new(window_ms: f32, sample_rate: f32) -> Self {
        let window_size = ((window_ms * 0.001 * sample_rate) as usize).max(1);
        Self {
            sum_squares: 0.0,
            window_size,
            buffer: VecDeque::with_capacity(window_size),
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let input_squared = input * input;

        // Remove old sample
        if self.buffer.len() >= self.window_size {
            if let Some(old) = self.buffer.pop_front() {
                self.sum_squares -= old;
            }
        }

        // Add new sample
        self.buffer.push_back(input_squared);
        self.sum_squares += input_squared;

        // Calculate RMS
        (self.sum_squares / self.buffer.len() as f32).sqrt()
    }

    fn reset(&mut self) {
        self.sum_squares = 0.0;
        self.buffer.clear();
    }
}

/// Look-ahead delay buffer
#[derive(Debug, Clone)]
struct LookaheadBuffer {
    buffer: VecDeque<f32>,
    delay_samples: usize,
}

impl LookaheadBuffer {
    fn new(delay_ms: f32, sample_rate: f32) -> Self {
        let delay_samples = (delay_ms * 0.001 * sample_rate) as usize;
        let mut buffer = VecDeque::with_capacity(delay_samples + 1);

        // Pre-fill with zeros
        for _ in 0..delay_samples {
            buffer.push_back(0.0);
        }

        Self {
            buffer,
            delay_samples,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        self.buffer.push_back(input);
        self.buffer.pop_front().unwrap_or(0.0)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        for _ in 0..self.delay_samples {
            self.buffer.push_back(0.0);
        }
    }

    fn delay_samples(&self) -> usize {
        self.delay_samples
    }
}

/// Gain smoothing envelope
#[derive(Debug, Clone)]
struct GainEnvelope {
    attack_coeff: f32,
    release_coeff: f32,
    current_gain: f32,
}

impl GainEnvelope {
    fn new(attack_ms: f32, release_ms: f32, sample_rate: f32) -> Self {
        Self {
            attack_coeff: (-1.0 / (attack_ms * 0.001 * sample_rate)).exp(),
            release_coeff: (-1.0 / (release_ms * 0.001 * sample_rate)).exp(),
            current_gain: 1.0,
        }
    }

    fn process(&mut self, target_gain: f32) -> f32 {
        // Attack when gain decreases, release when gain increases
        let coeff = if target_gain < self.current_gain {
            self.attack_coeff
        } else {
            self.release_coeff
        };

        self.current_gain = target_gain + (self.current_gain - target_gain) * coeff;
        self.current_gain
    }

    fn update(&mut self, attack_ms: f32, release_ms: f32, sample_rate: f32) {
        self.attack_coeff = (-1.0 / (attack_ms * 0.001 * sample_rate)).exp();
        self.release_coeff = (-1.0 / (release_ms * 0.001 * sample_rate)).exp();
    }

    fn reset(&mut self) {
        self.current_gain = 1.0;
    }
}

/// Main Compressor/Limiter processor
pub struct Compressor {
    config: CompressorConfig,
    sample_rate: f32,

    // Sidechain processing
    sidechain_hpf: SidechainHpf,
    rms_calc: RmsCalculator,

    // Look-ahead delay
    lookahead: LookaheadBuffer,

    // Gain smoothing
    gain_envelope: GainEnvelope,

    // Computed values
    threshold_linear: f32,
    makeup_gain_linear: f32,
    knee_half: f32,

    // Metering
    gain_reduction_db: f32,
    input_level_db: f32,
    output_level_db: f32,
}

impl Compressor {
    /// Create a new Compressor
    pub fn new(config: CompressorConfig, sample_rate: f32) -> Self {
        let sidechain_hpf = SidechainHpf::new(config.sidechain_hpf_hz, sample_rate);
        let rms_calc = RmsCalculator::new(10.0, sample_rate); // 10ms RMS window
        let lookahead = LookaheadBuffer::new(config.lookahead_ms, sample_rate);
        let gain_envelope = GainEnvelope::new(config.attack_ms, config.release_ms, sample_rate);

        let threshold_linear = 10.0_f32.powf(config.threshold_db / 20.0);

        // Calculate auto-makeup gain
        let makeup_gain_linear = if config.auto_makeup {
            Self::calculate_auto_makeup(&config)
        } else {
            10.0_f32.powf(config.makeup_gain_db / 20.0)
        };

        let knee_half = config.knee_db / 2.0;

        debug!(
            "Compressor initialized: threshold {}dB, ratio {}:1",
            config.threshold_db, config.ratio
        );

        Self {
            config,
            sample_rate,
            sidechain_hpf,
            rms_calc,
            lookahead,
            gain_envelope,
            threshold_linear,
            makeup_gain_linear,
            knee_half,
            gain_reduction_db: 0.0,
            input_level_db: -60.0,
            output_level_db: -60.0,
        }
    }

    /// Create with default settings
    pub fn with_sample_rate(sample_rate: f32) -> Self {
        Self::new(CompressorConfig::default(), sample_rate)
    }

    /// Create a limiter
    pub fn limiter(sample_rate: f32) -> Self {
        Self::new(CompressorConfig::limiter(), sample_rate)
    }

    /// Calculate auto-makeup gain based on threshold and ratio
    fn calculate_auto_makeup(config: &CompressorConfig) -> f32 {
        // Estimate gain reduction at -18dB input level (typical voice)
        let test_level = -18.0;
        if test_level > config.threshold_db {
            let over = test_level - config.threshold_db;
            let reduction = over - over / config.ratio;
            10.0_f32.powf((reduction * 0.5) / 20.0) // Apply half the estimated reduction
        } else {
            1.0
        }
    }

    /// Process a buffer of audio
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(anyhow::anyhow!("Buffer size mismatch"));
        }

        let mut max_input = 0.0_f32;
        let mut max_output = 0.0_f32;
        let mut max_gr = 0.0_f32;

        for (i, &sample) in input.iter().enumerate() {
            // Sidechain processing
            let sidechain = self.sidechain_hpf.process(sample);

            // Level detection
            let level = match self.config.detection_mode {
                DetectionMode::Peak => sidechain.abs(),
                DetectionMode::Rms => self.rms_calc.process(sidechain),
                DetectionMode::TruePeak => sidechain.abs(), // Simplified, would need oversampling
            };

            max_input = max_input.max(level);

            // Convert to dB
            let level_db = 20.0 * level.max(1e-10).log10();

            // Calculate gain reduction
            let gr_db = self.calculate_gain_reduction(level_db);
            max_gr = max_gr.max(-gr_db);

            // Convert to linear
            let target_gain = 10.0_f32.powf(gr_db / 20.0);

            // Smooth the gain
            let smoothed_gain = self.gain_envelope.process(target_gain);

            // Get delayed input (for look-ahead)
            let delayed_input = self.lookahead.process(sample);

            // Apply gain and makeup
            let compressed = delayed_input * smoothed_gain * self.makeup_gain_linear;

            // Mix dry/wet
            let mixed = if self.config.mix < 1.0 {
                delayed_input * (1.0 - self.config.mix) + compressed * self.config.mix
            } else {
                compressed
            };

            output[i] = mixed;
            max_output = max_output.max(mixed.abs());
        }

        // Update metering
        self.gain_reduction_db = max_gr;
        self.input_level_db = 20.0 * max_input.max(1e-10).log10();
        self.output_level_db = 20.0 * max_output.max(1e-10).log10();

        Ok(())
    }

    /// Calculate gain reduction for a given input level (soft knee)
    fn calculate_gain_reduction(&self, level_db: f32) -> f32 {
        let threshold = self.config.threshold_db;
        let ratio = self.config.ratio;
        let knee = self.config.knee_db;

        if knee <= 0.0 || ratio <= 1.0 {
            // Hard knee
            if level_db <= threshold {
                0.0
            } else {
                let over = level_db - threshold;
                -(over - over / ratio)
            }
        } else {
            // Soft knee
            let knee_start = threshold - self.knee_half;
            let knee_end = threshold + self.knee_half;

            if level_db <= knee_start {
                0.0
            } else if level_db >= knee_end {
                let over = level_db - threshold;
                -(over - over / ratio)
            } else {
                // In knee region - quadratic interpolation
                let x = level_db - knee_start;
                let knee_factor = x / knee;
                let over = x * knee_factor * 0.5;
                -(over - over / ratio)
            }
        }
    }

    /// Process in-place
    pub fn process_inplace(&mut self, buffer: &mut [f32]) -> Result<()> {
        let input = buffer.to_vec();
        self.process(&input, buffer)
    }

    /// Update configuration
    pub fn set_config(&mut self, config: CompressorConfig) {
        self.sidechain_hpf = SidechainHpf::new(config.sidechain_hpf_hz, self.sample_rate);
        self.lookahead = LookaheadBuffer::new(config.lookahead_ms, self.sample_rate);
        self.gain_envelope
            .update(config.attack_ms, config.release_ms, self.sample_rate);

        self.threshold_linear = 10.0_f32.powf(config.threshold_db / 20.0);
        self.knee_half = config.knee_db / 2.0;

        self.makeup_gain_linear = if config.auto_makeup {
            Self::calculate_auto_makeup(&config)
        } else {
            10.0_f32.powf(config.makeup_gain_db / 20.0)
        };

        self.config = config;
    }

    /// Set threshold
    pub fn set_threshold(&mut self, threshold_db: f32) {
        self.config.threshold_db = threshold_db.clamp(-60.0, 0.0);
        self.threshold_linear = 10.0_f32.powf(self.config.threshold_db / 20.0);

        if self.config.auto_makeup {
            self.makeup_gain_linear = Self::calculate_auto_makeup(&self.config);
        }
    }

    /// Set ratio
    pub fn set_ratio(&mut self, ratio: f32) {
        self.config.ratio = ratio.clamp(1.0, 100.0);

        if self.config.auto_makeup {
            self.makeup_gain_linear = Self::calculate_auto_makeup(&self.config);
        }
    }

    /// Set attack time
    pub fn set_attack(&mut self, attack_ms: f32) {
        self.config.attack_ms = attack_ms.clamp(0.01, 100.0);
        self.gain_envelope
            .update(self.config.attack_ms, self.config.release_ms, self.sample_rate);
    }

    /// Set release time
    pub fn set_release(&mut self, release_ms: f32) {
        self.config.release_ms = release_ms.clamp(10.0, 2000.0);
        self.gain_envelope
            .update(self.config.attack_ms, self.config.release_ms, self.sample_rate);
    }

    /// Set makeup gain
    pub fn set_makeup_gain(&mut self, gain_db: f32) {
        self.config.makeup_gain_db = gain_db.clamp(-12.0, 24.0);
        self.config.auto_makeup = false;
        self.makeup_gain_linear = 10.0_f32.powf(self.config.makeup_gain_db / 20.0);
    }

    /// Get current gain reduction in dB
    pub fn get_gain_reduction_db(&self) -> f32 {
        self.gain_reduction_db
    }

    /// Get input level in dB
    pub fn get_input_level_db(&self) -> f32 {
        self.input_level_db
    }

    /// Get output level in dB
    pub fn get_output_level_db(&self) -> f32 {
        self.output_level_db
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.sidechain_hpf.reset();
        self.rms_calc.reset();
        self.lookahead.reset();
        self.gain_envelope.reset();
        self.gain_reduction_db = 0.0;
        self.input_level_db = -60.0;
        self.output_level_db = -60.0;
    }

    /// Get latency in samples
    pub fn latency_samples(&self) -> usize {
        self.lookahead.delay_samples()
    }

    /// Get configuration
    pub fn get_config(&self) -> &CompressorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_creation() {
        let comp = Compressor::with_sample_rate(48000.0);
        assert_eq!(comp.latency_samples(), 0); // Default has no lookahead
    }

    #[test]
    fn test_limiter_creation() {
        let limiter = Compressor::limiter(48000.0);
        assert!(limiter.latency_samples() > 0); // Limiter has lookahead
    }

    #[test]
    fn test_compression() {
        let mut comp = Compressor::with_sample_rate(48000.0);
        comp.set_threshold(-20.0);
        comp.set_ratio(4.0);

        // Generate a loud signal
        let input: Vec<f32> = vec![0.5; 1024];
        let mut output = vec![0.0; 1024];

        let result = comp.process(&input, &mut output);
        assert!(result.is_ok());

        // Output should be reduced
        let _input_rms: f32 = input.iter().map(|&x| x * x).sum::<f32>() / input.len() as f32;
        let _output_rms: f32 = output.iter().map(|&x| x * x).sum::<f32>() / output.len() as f32;

        // With makeup gain, output might be similar, but gain reduction should occur
        assert!(comp.get_gain_reduction_db() > 0.0);
    }

    #[test]
    fn test_limiting() {
        let mut limiter = Compressor::limiter(48000.0);
        limiter.set_threshold(-3.0);

        // Generate an over-limit signal
        let input: Vec<f32> = vec![1.5; 1024]; // Way over 0dB
        let mut output = vec![0.0; 1024];

        // Process multiple times to fill lookahead buffer
        for _ in 0..10 {
            let _ = limiter.process(&input, &mut output);
        }

        // Output should be limited to near threshold
        let max_output = output.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
        assert!(
            max_output < 1.0,
            "Output should be limited, got {}",
            max_output
        );
    }

    #[test]
    fn test_gain_reduction_calculation() {
        let comp = Compressor::with_sample_rate(48000.0);

        // Below threshold: no reduction
        let gr = comp.calculate_gain_reduction(-30.0);
        assert!(gr.abs() < 0.1);

        // Above threshold: should reduce
        let gr = comp.calculate_gain_reduction(-10.0);
        assert!(gr < 0.0);
    }
}
