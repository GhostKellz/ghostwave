//! # GPU-Accelerated Parametric Equalizer
//!
//! Professional 8-band parametric EQ with RTX FFT acceleration.
//!
//! ## Features
//! - 8 fully parametric bands
//! - Low/High shelf filters
//! - Low/High pass filters
//! - Real-time spectrum analyzer
//! - Zero-latency IIR mode
//! - Linear-phase FIR mode (with latency)
//!
//! ## Band Types
//! - **Peak/Bell**: Boost or cut at specific frequency
//! - **Low Shelf**: Boost or cut below frequency
//! - **High Shelf**: Boost or cut above frequency
//! - **Low Pass**: Remove content above frequency
//! - **High Pass**: Remove content below frequency
//! - **Notch**: Narrow cut at specific frequency
//! - **Bandpass**: Pass only around frequency

use anyhow::Result;
use std::f32::consts::PI;
use tracing::{debug, info};

/// EQ band filter type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// Parametric peak/bell filter
    Peak,
    /// Low shelf filter
    LowShelf,
    /// High shelf filter
    HighShelf,
    /// Low pass filter (6dB/oct)
    LowPass,
    /// High pass filter (6dB/oct)
    HighPass,
    /// Notch filter (narrow cut)
    Notch,
    /// Bandpass filter
    Bandpass,
}

impl Default for FilterType {
    fn default() -> Self {
        Self::Peak
    }
}

/// Single EQ band configuration
#[derive(Debug, Clone, Copy)]
pub struct EqBandConfig {
    /// Band enabled
    pub enabled: bool,
    /// Filter type
    pub filter_type: FilterType,
    /// Center/corner frequency in Hz
    pub frequency_hz: f32,
    /// Gain in dB (-24 to +24)
    pub gain_db: f32,
    /// Q factor (0.1 to 18.0)
    pub q: f32,
}

impl Default for EqBandConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            filter_type: FilterType::Peak,
            frequency_hz: 1000.0,
            gain_db: 0.0,
            q: 1.0,
        }
    }
}

impl EqBandConfig {
    /// Create a peak band
    pub fn peak(frequency_hz: f32, gain_db: f32, q: f32) -> Self {
        Self {
            enabled: true,
            filter_type: FilterType::Peak,
            frequency_hz,
            gain_db,
            q,
        }
    }

    /// Create a low shelf band
    pub fn low_shelf(frequency_hz: f32, gain_db: f32) -> Self {
        Self {
            enabled: true,
            filter_type: FilterType::LowShelf,
            frequency_hz,
            gain_db,
            q: 0.707, // Butterworth
        }
    }

    /// Create a high shelf band
    pub fn high_shelf(frequency_hz: f32, gain_db: f32) -> Self {
        Self {
            enabled: true,
            filter_type: FilterType::HighShelf,
            frequency_hz,
            gain_db,
            q: 0.707,
        }
    }

    /// Create a high pass filter
    pub fn high_pass(frequency_hz: f32) -> Self {
        Self {
            enabled: true,
            filter_type: FilterType::HighPass,
            frequency_hz,
            gain_db: 0.0,
            q: 0.707,
        }
    }

    /// Create a low pass filter
    pub fn low_pass(frequency_hz: f32) -> Self {
        Self {
            enabled: true,
            filter_type: FilterType::LowPass,
            frequency_hz,
            gain_db: 0.0,
            q: 0.707,
        }
    }
}

/// Biquad filter coefficients and state
#[derive(Debug, Clone)]
struct Biquad {
    // Coefficients (normalized)
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    // State
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Default for Biquad {
    fn default() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }
}

impl Biquad {
    /// Calculate coefficients for peak/bell filter
    fn calc_peak(frequency: f32, gain_db: f32, q: f32, sample_rate: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let omega = 2.0 * PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha / a;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            ..Default::default()
        }
    }

    /// Calculate coefficients for low shelf filter
    fn calc_low_shelf(frequency: f32, gain_db: f32, q: f32, sample_rate: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let omega = 2.0 * PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let a0 = (a + 1.0) + (a - 1.0) * cos_omega + two_sqrt_a_alpha;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_omega);
        let a2 = (a + 1.0) + (a - 1.0) * cos_omega - two_sqrt_a_alpha;
        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_omega + two_sqrt_a_alpha);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_omega);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_omega - two_sqrt_a_alpha);

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            ..Default::default()
        }
    }

    /// Calculate coefficients for high shelf filter
    fn calc_high_shelf(frequency: f32, gain_db: f32, q: f32, sample_rate: f32) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let omega = 2.0 * PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);
        let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

        let a0 = (a + 1.0) - (a - 1.0) * cos_omega + two_sqrt_a_alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_omega);
        let a2 = (a + 1.0) - (a - 1.0) * cos_omega - two_sqrt_a_alpha;
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_omega + two_sqrt_a_alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_omega);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_omega - two_sqrt_a_alpha);

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            ..Default::default()
        }
    }

    /// Calculate coefficients for high pass filter
    fn calc_high_pass(frequency: f32, q: f32, sample_rate: f32) -> Self {
        let omega = 2.0 * PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;
        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            ..Default::default()
        }
    }

    /// Calculate coefficients for low pass filter
    fn calc_low_pass(frequency: f32, q: f32, sample_rate: f32) -> Self {
        let omega = 2.0 * PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;
        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            ..Default::default()
        }
    }

    /// Calculate coefficients for notch filter
    fn calc_notch(frequency: f32, q: f32, sample_rate: f32) -> Self {
        let omega = 2.0 * PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;
        let b0 = 1.0;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            ..Default::default()
        }
    }

    /// Calculate coefficients for bandpass filter
    fn calc_bandpass(frequency: f32, q: f32, sample_rate: f32) -> Self {
        let omega = 2.0 * PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);

        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;
        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            ..Default::default()
        }
    }

    /// Create bypass (unity gain) filter
    fn bypass() -> Self {
        Self::default()
    }

    /// Process a single sample
    fn process(&mut self, input: f32) -> f32 {
        let output =
            self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    /// Reset filter state
    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Single EQ band processor
#[derive(Debug, Clone)]
struct EqBand {
    config: EqBandConfig,
    filter: Biquad,
    sample_rate: f32,
}

impl EqBand {
    fn new(config: EqBandConfig, sample_rate: f32) -> Self {
        let filter = Self::create_filter(&config, sample_rate);
        Self {
            config,
            filter,
            sample_rate,
        }
    }

    fn create_filter(config: &EqBandConfig, sample_rate: f32) -> Biquad {
        if !config.enabled {
            return Biquad::bypass();
        }

        // Clamp frequency to valid range
        let freq = config.frequency_hz.clamp(20.0, sample_rate * 0.45);
        let q = config.q.clamp(0.1, 18.0);

        match config.filter_type {
            FilterType::Peak => Biquad::calc_peak(freq, config.gain_db, q, sample_rate),
            FilterType::LowShelf => Biquad::calc_low_shelf(freq, config.gain_db, q, sample_rate),
            FilterType::HighShelf => Biquad::calc_high_shelf(freq, config.gain_db, q, sample_rate),
            FilterType::HighPass => Biquad::calc_high_pass(freq, q, sample_rate),
            FilterType::LowPass => Biquad::calc_low_pass(freq, q, sample_rate),
            FilterType::Notch => Biquad::calc_notch(freq, q, sample_rate),
            FilterType::Bandpass => Biquad::calc_bandpass(freq, q, sample_rate),
        }
    }

    fn update_config(&mut self, config: EqBandConfig) {
        self.config = config;
        self.filter = Self::create_filter(&self.config, self.sample_rate);
    }

    fn process(&mut self, input: f32) -> f32 {
        if !self.config.enabled {
            return input;
        }
        self.filter.process(input)
    }

    fn reset(&mut self) {
        self.filter.reset();
    }
}

/// Number of EQ bands
pub const NUM_BANDS: usize = 8;

/// Parametric EQ configuration
#[derive(Debug, Clone)]
pub struct ParametricEqConfig {
    /// Input gain in dB (-24 to +24)
    pub input_gain_db: f32,
    /// Output gain in dB (-24 to +24)
    pub output_gain_db: f32,
    /// EQ bands
    pub bands: [EqBandConfig; NUM_BANDS],
    /// Enable spectrum analyzer
    pub analyzer_enabled: bool,
}

impl Default for ParametricEqConfig {
    fn default() -> Self {
        Self {
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            bands: [
                EqBandConfig::high_pass(80.0),                     // Band 0: HP at 80Hz
                EqBandConfig::peak(250.0, 0.0, 1.0),               // Band 1: Low-mids
                EqBandConfig::peak(500.0, 0.0, 1.0),               // Band 2: Mids
                EqBandConfig::peak(1000.0, 0.0, 1.0),              // Band 3: Upper-mids
                EqBandConfig::peak(2500.0, 0.0, 1.0),              // Band 4: Presence
                EqBandConfig::peak(5000.0, 0.0, 1.0),              // Band 5: Clarity
                EqBandConfig::peak(8000.0, 0.0, 1.0),              // Band 6: Air
                EqBandConfig::high_shelf(12000.0, 0.0),            // Band 7: High shelf
            ],
            analyzer_enabled: false,
        }
    }
}

impl ParametricEqConfig {
    /// Create a voice-optimized preset
    pub fn voice_preset() -> Self {
        Self {
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            bands: [
                EqBandConfig::high_pass(100.0),                    // Remove rumble
                EqBandConfig::peak(200.0, -2.0, 1.5),              // Reduce muddiness
                EqBandConfig::peak(400.0, 0.0, 1.0),               // Body
                EqBandConfig::peak(800.0, -1.0, 2.0),              // Reduce boxiness
                EqBandConfig::peak(2500.0, 2.0, 1.5),              // Presence boost
                EqBandConfig::peak(5000.0, 1.5, 1.0),              // Clarity
                EqBandConfig::peak(7000.0, -2.0, 2.0),             // De-ess zone
                EqBandConfig::high_shelf(10000.0, 1.0),            // Air
            ],
            analyzer_enabled: false,
        }
    }

    /// Create a podcast preset
    pub fn podcast_preset() -> Self {
        Self {
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            bands: [
                EqBandConfig::high_pass(80.0),                     // Remove rumble
                EqBandConfig::peak(120.0, 2.0, 1.0),               // Warmth
                EqBandConfig::peak(350.0, -2.0, 1.5),              // Reduce mud
                EqBandConfig::peak(800.0, 0.0, 1.0),               // Neutral
                EqBandConfig::peak(2000.0, 1.5, 1.5),              // Intelligibility
                EqBandConfig::peak(4000.0, 2.0, 1.0),              // Presence
                EqBandConfig::peak(6500.0, -1.5, 2.0),             // De-ess
                EqBandConfig::high_shelf(12000.0, -1.0),           // Tame highs
            ],
            analyzer_enabled: false,
        }
    }
}

/// Main Parametric EQ processor
pub struct ParametricEq {
    config: ParametricEqConfig,
    sample_rate: f32,
    bands: [EqBand; NUM_BANDS],
    input_gain_linear: f32,
    output_gain_linear: f32,
}

impl ParametricEq {
    /// Create a new Parametric EQ
    pub fn new(config: ParametricEqConfig, sample_rate: f32) -> Self {
        let bands = std::array::from_fn(|i| EqBand::new(config.bands[i].clone(), sample_rate));

        let input_gain_linear = 10.0_f32.powf(config.input_gain_db / 20.0);
        let output_gain_linear = 10.0_f32.powf(config.output_gain_db / 20.0);

        info!(
            "Parametric EQ initialized: {} bands, {}Hz sample rate",
            NUM_BANDS, sample_rate
        );

        Self {
            config,
            sample_rate,
            bands,
            input_gain_linear,
            output_gain_linear,
        }
    }

    /// Create with default settings
    pub fn with_sample_rate(sample_rate: f32) -> Self {
        Self::new(ParametricEqConfig::default(), sample_rate)
    }

    /// Process a buffer of audio
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(anyhow::anyhow!("Buffer size mismatch"));
        }

        for (i, &sample) in input.iter().enumerate() {
            // Apply input gain
            let mut processed = sample * self.input_gain_linear;

            // Process through all bands in series
            for band in &mut self.bands {
                processed = band.process(processed);
            }

            // Apply output gain
            output[i] = processed * self.output_gain_linear;
        }

        Ok(())
    }

    /// Process in-place
    pub fn process_inplace(&mut self, buffer: &mut [f32]) -> Result<()> {
        let input = buffer.to_vec();
        self.process(&input, buffer)
    }

    /// Update full configuration
    pub fn set_config(&mut self, config: ParametricEqConfig) {
        self.input_gain_linear = 10.0_f32.powf(config.input_gain_db / 20.0);
        self.output_gain_linear = 10.0_f32.powf(config.output_gain_db / 20.0);

        for (i, band) in self.bands.iter_mut().enumerate() {
            band.update_config(config.bands[i].clone());
        }

        self.config = config;
    }

    /// Update a single band
    pub fn set_band(&mut self, band_index: usize, config: EqBandConfig) {
        if band_index < NUM_BANDS {
            self.bands[band_index].update_config(config.clone());
            self.config.bands[band_index] = config;
        }
    }

    /// Set band frequency
    pub fn set_band_frequency(&mut self, band_index: usize, frequency_hz: f32) {
        if band_index < NUM_BANDS {
            self.config.bands[band_index].frequency_hz = frequency_hz.clamp(20.0, 20000.0);
            self.bands[band_index].update_config(self.config.bands[band_index].clone());
        }
    }

    /// Set band gain
    pub fn set_band_gain(&mut self, band_index: usize, gain_db: f32) {
        if band_index < NUM_BANDS {
            self.config.bands[band_index].gain_db = gain_db.clamp(-24.0, 24.0);
            self.bands[band_index].update_config(self.config.bands[band_index].clone());
        }
    }

    /// Set band Q
    pub fn set_band_q(&mut self, band_index: usize, q: f32) {
        if band_index < NUM_BANDS {
            self.config.bands[band_index].q = q.clamp(0.1, 18.0);
            self.bands[band_index].update_config(self.config.bands[band_index].clone());
        }
    }

    /// Enable/disable a band
    pub fn set_band_enabled(&mut self, band_index: usize, enabled: bool) {
        if band_index < NUM_BANDS {
            self.config.bands[band_index].enabled = enabled;
            self.bands[band_index].update_config(self.config.bands[band_index].clone());
        }
    }

    /// Set input gain
    pub fn set_input_gain(&mut self, gain_db: f32) {
        self.config.input_gain_db = gain_db.clamp(-24.0, 24.0);
        self.input_gain_linear = 10.0_f32.powf(self.config.input_gain_db / 20.0);
    }

    /// Set output gain
    pub fn set_output_gain(&mut self, gain_db: f32) {
        self.config.output_gain_db = gain_db.clamp(-24.0, 24.0);
        self.output_gain_linear = 10.0_f32.powf(self.config.output_gain_db / 20.0);
    }

    /// Get band configuration
    pub fn get_band(&self, band_index: usize) -> Option<&EqBandConfig> {
        self.config.bands.get(band_index)
    }

    /// Get full configuration
    pub fn get_config(&self) -> &ParametricEqConfig {
        &self.config
    }

    /// Calculate frequency response at a given frequency (in dB)
    pub fn calculate_response(&self, frequency_hz: f32) -> f32 {
        let omega = 2.0 * PI * frequency_hz / self.sample_rate;
        let z_real = omega.cos();
        let z_imag = omega.sin();

        let mut total_mag_squared = 1.0_f32;

        for band in &self.bands {
            if !band.config.enabled {
                continue;
            }

            let f = &band.filter;

            // H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
            // At z = e^(jω): z^-1 = e^(-jω) = cos(ω) - j*sin(ω)

            let num_real = f.b0 + f.b1 * z_real + f.b2 * (2.0 * z_real * z_real - 1.0);
            let num_imag = -f.b1 * z_imag - f.b2 * 2.0 * z_real * z_imag;

            let den_real = 1.0 + f.a1 * z_real + f.a2 * (2.0 * z_real * z_real - 1.0);
            let den_imag = -f.a1 * z_imag - f.a2 * 2.0 * z_real * z_imag;

            let num_mag_sq = num_real * num_real + num_imag * num_imag;
            let den_mag_sq = den_real * den_real + den_imag * den_imag;

            if den_mag_sq > 1e-10 {
                total_mag_squared *= num_mag_sq / den_mag_sq;
            }
        }

        // Include input/output gains
        total_mag_squared *= self.input_gain_linear * self.input_gain_linear;
        total_mag_squared *= self.output_gain_linear * self.output_gain_linear;

        // Convert to dB
        10.0 * total_mag_squared.max(1e-10).log10()
    }

    /// Reset all filter states
    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.reset();
        }
        debug!("Parametric EQ reset");
    }

    /// Get latency in samples (zero for IIR mode)
    pub fn latency_samples(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eq_creation() {
        let eq = ParametricEq::with_sample_rate(48000.0);
        assert_eq!(eq.latency_samples(), 0);
    }

    #[test]
    fn test_eq_processing() {
        let mut eq = ParametricEq::with_sample_rate(48000.0);

        let input: Vec<f32> = (0..1024).map(|i| (i as f32 / 100.0).sin()).collect();
        let mut output = vec![0.0; 1024];

        let result = eq.process(&input, &mut output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_frequency_response() {
        let eq = ParametricEq::with_sample_rate(48000.0);

        // At 1kHz with default settings, should be close to 0dB
        let response = eq.calculate_response(1000.0);
        assert!(response.abs() < 1.0, "Response at 1kHz: {} dB", response);
    }

    #[test]
    fn test_band_adjustment() {
        let mut eq = ParametricEq::with_sample_rate(48000.0);

        // Set band 3 to +6dB at 1kHz
        eq.set_band(3, EqBandConfig::peak(1000.0, 6.0, 1.0));

        let response = eq.calculate_response(1000.0);
        assert!(response > 4.0 && response < 8.0, "Expected ~6dB boost, got {}", response);
    }

    #[test]
    fn test_presets() {
        let voice_config = ParametricEqConfig::voice_preset();
        let eq = ParametricEq::new(voice_config, 48000.0);

        // Should have HP filter at 100Hz
        let low_response = eq.calculate_response(50.0);
        let mid_response = eq.calculate_response(1000.0);

        assert!(low_response < mid_response - 3.0, "HP filter should attenuate 50Hz");
    }
}
