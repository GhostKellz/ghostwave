//! Audio processing chain for GhostWave VST
//!
//! Integrates all DSP modules from ghostwave-core:
//! - Noise suppression
//! - Echo removal
//! - De-esser
//! - Parametric EQ
//! - Compressor
//! - Limiter

use crate::params::GhostWaveParams;
use ghostwave_core::{
    compressor::{Compressor, CompressorConfig, DetectionMode},
    de_esser::DeEsser,
    parametric_eq::{EqBandConfig, FilterType, ParametricEq, ParametricEqConfig, NUM_BANDS},
    NoiseProcessor, NoiseSuppressionConfig,
};

/// Audio processing state
pub struct GhostWaveProcessing {
    #[allow(dead_code)] // Will be used for sample rate dependent calculations
    sample_rate: f32,

    // Processing modules
    noise_processor: Option<NoiseProcessor>,
    de_esser: DeEsser,
    eq: ParametricEq,
    compressor: Compressor,
    limiter: Compressor,

    // Parameter cache to detect changes
    cached_noise_strength: f32,
    cached_deesser_threshold: f32,
    cached_deesser_freq: f32,
    cached_comp_threshold: f32,
    cached_comp_ratio: f32,

    // Scratch buffer for processing
    scratch_buffer: Vec<f32>,
}

impl GhostWaveProcessing {
    /// Create new processing chain
    pub fn new(sample_rate: f32) -> Self {
        // Initialize noise processor
        let noise_config = NoiseSuppressionConfig {
            enabled: true,
            strength: 0.7,
            gate_threshold: -45.0,
            release_time: 0.1,
        };
        let noise_processor = NoiseProcessor::new(&noise_config).ok();

        // Initialize de-esser with defaults
        let de_esser = DeEsser::with_sample_rate(sample_rate);

        // Initialize EQ with 5-band voice preset (8 bands available but we use 5)
        let mut bands = [EqBandConfig::default(); NUM_BANDS];

        // Low shelf
        bands[0] = EqBandConfig {
            enabled: true,
            filter_type: FilterType::LowShelf,
            frequency_hz: 100.0,
            gain_db: 0.0,
            q: 0.707,
        };
        // Low-mid
        bands[1] = EqBandConfig {
            enabled: true,
            filter_type: FilterType::Peak,
            frequency_hz: 350.0,
            gain_db: 0.0,
            q: 1.0,
        };
        // Mid
        bands[2] = EqBandConfig {
            enabled: true,
            filter_type: FilterType::Peak,
            frequency_hz: 1000.0,
            gain_db: 0.0,
            q: 1.0,
        };
        // High-mid
        bands[3] = EqBandConfig {
            enabled: true,
            filter_type: FilterType::Peak,
            frequency_hz: 3500.0,
            gain_db: 0.0,
            q: 1.0,
        };
        // High shelf
        bands[4] = EqBandConfig {
            enabled: true,
            filter_type: FilterType::HighShelf,
            frequency_hz: 8000.0,
            gain_db: 0.0,
            q: 0.707,
        };

        let eq_config = ParametricEqConfig {
            bands,
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            analyzer_enabled: false,
        };
        let eq = ParametricEq::new(eq_config, sample_rate);

        // Initialize compressor with vocal preset
        let compressor = Compressor::new(CompressorConfig::vocal(), sample_rate);

        // Initialize limiter
        let limiter = Compressor::limiter(sample_rate);

        Self {
            sample_rate,
            noise_processor,
            de_esser,
            eq,
            compressor,
            limiter,
            cached_noise_strength: 0.7,
            cached_deesser_threshold: -20.0,
            cached_deesser_freq: 6500.0,
            cached_comp_threshold: -18.0,
            cached_comp_ratio: 4.0,
            scratch_buffer: Vec::with_capacity(8192),
        }
    }

    /// Reset all processing state
    pub fn reset(&mut self) {
        self.de_esser.reset();
        self.eq.reset();
        self.compressor.reset();
        self.limiter.reset();
    }

    /// Update parameters from plugin params
    pub fn update_params(&mut self, params: &GhostWaveParams) {
        // Update noise processor
        if let Some(ref mut processor) = self.noise_processor {
            let strength = params.noise_strength.value();
            if (strength - self.cached_noise_strength).abs() > 0.001 {
                processor.set_strength(strength);
                self.cached_noise_strength = strength;
            }
        }

        // Update de-esser
        let deesser_threshold = params.deesser_threshold.value();
        let deesser_freq = params.deesser_frequency.value();
        if (deesser_threshold - self.cached_deesser_threshold).abs() > 0.1 {
            self.de_esser.set_threshold(deesser_threshold);
            self.cached_deesser_threshold = deesser_threshold;
        }
        if (deesser_freq - self.cached_deesser_freq).abs() > 10.0 {
            self.de_esser.set_frequency(deesser_freq);
            self.cached_deesser_freq = deesser_freq;
        }
        self.de_esser.set_ratio(params.deesser_ratio.value());

        // Update EQ bands
        self.update_eq_params(params);

        // Update compressor
        let comp_threshold = params.comp_threshold.value();
        let comp_ratio = params.comp_ratio.value();
        if (comp_threshold - self.cached_comp_threshold).abs() > 0.1
            || (comp_ratio - self.cached_comp_ratio).abs() > 0.1
        {
            let config = CompressorConfig {
                threshold_db: comp_threshold,
                ratio: comp_ratio,
                attack_ms: params.comp_attack.value(),
                release_ms: params.comp_release.value(),
                knee_db: params.comp_knee.value(),
                makeup_gain_db: params.comp_makeup.value(),
                auto_makeup: false,
                lookahead_ms: 5.0,
                detection_mode: DetectionMode::Peak,
                sidechain_hpf_hz: 80.0,
                mix: 1.0,
            };
            self.compressor.set_config(config);
            self.cached_comp_threshold = comp_threshold;
            self.cached_comp_ratio = comp_ratio;
        }

        // Update limiter ceiling
        let mut limiter_config = CompressorConfig::limiter();
        limiter_config.threshold_db = params.limiter_ceiling.value();
        self.limiter.set_config(limiter_config);
    }

    /// Update EQ parameters
    fn update_eq_params(&mut self, params: &GhostWaveParams) {
        // Band 1: Low Shelf
        self.eq.set_band_enabled(0, params.eq_band1_enabled.value());
        self.eq.set_band_frequency(0, params.eq_band1_freq.value());
        self.eq.set_band_gain(0, params.eq_band1_gain.value());

        // Band 2: Low-Mid
        self.eq.set_band_enabled(1, params.eq_band2_enabled.value());
        self.eq.set_band_frequency(1, params.eq_band2_freq.value());
        self.eq.set_band_gain(1, params.eq_band2_gain.value());
        self.eq.set_band_q(1, params.eq_band2_q.value());

        // Band 3: Mid
        self.eq.set_band_enabled(2, params.eq_band3_enabled.value());
        self.eq.set_band_frequency(2, params.eq_band3_freq.value());
        self.eq.set_band_gain(2, params.eq_band3_gain.value());
        self.eq.set_band_q(2, params.eq_band3_q.value());

        // Band 4: High-Mid
        self.eq.set_band_enabled(3, params.eq_band4_enabled.value());
        self.eq.set_band_frequency(3, params.eq_band4_freq.value());
        self.eq.set_band_gain(3, params.eq_band4_gain.value());
        self.eq.set_band_q(3, params.eq_band4_q.value());

        // Band 5: High Shelf
        self.eq.set_band_enabled(4, params.eq_band5_enabled.value());
        self.eq.set_band_frequency(4, params.eq_band5_freq.value());
        self.eq.set_band_gain(4, params.eq_band5_gain.value());

        // Output gain
        let output_db = params.output_gain.value();
        self.eq.set_output_gain(output_db);
    }

    /// Process a block of audio samples
    pub fn process_block(&mut self, buffer: &mut [f32]) {
        if buffer.is_empty() {
            return;
        }

        // Ensure scratch buffer is large enough
        if self.scratch_buffer.len() < buffer.len() {
            self.scratch_buffer.resize(buffer.len(), 0.0);
        }

        // 1. Noise Suppression (first in chain)
        if let Some(ref mut processor) = self.noise_processor {
            self.scratch_buffer[..buffer.len()].copy_from_slice(buffer);
            if processor
                .process(&self.scratch_buffer[..buffer.len()], buffer)
                .is_err()
            {
                // On error, leave buffer unchanged
            }
        }

        // 2. De-Esser
        let _ = self.de_esser.process_inplace(buffer);

        // 3. EQ
        let _ = self.eq.process_inplace(buffer);

        // 4. Compressor
        let _ = self.compressor.process_inplace(buffer);

        // 5. Limiter (last in chain)
        let _ = self.limiter.process_inplace(buffer);
    }
}
