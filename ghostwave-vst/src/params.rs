//! Plugin parameters for GhostWave VST
//!
//! Organized into sections:
//! - Noise Suppression (NVIDIA Broadcast-style)
//! - Echo Removal
//! - De-Esser
//! - Parametric EQ (8 bands)
//! - Compressor/Limiter
//! - Output

use nih_plug::prelude::*;
use nih_plug_egui::EguiState;
use std::sync::Arc;

/// Main plugin parameters
#[derive(Params)]
pub struct GhostWaveParams {
    /// Editor state for the GUI
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,

    // ═══════════════════════════════════════════════════════════════════════
    // NOISE SUPPRESSION
    // ═══════════════════════════════════════════════════════════════════════
    #[id = "noise_enabled"]
    pub noise_enabled: BoolParam,

    #[id = "noise_strength"]
    pub noise_strength: FloatParam,

    #[id = "noise_gate_threshold"]
    pub noise_gate_threshold: FloatParam,

    // ═══════════════════════════════════════════════════════════════════════
    // ECHO REMOVAL
    // ═══════════════════════════════════════════════════════════════════════
    #[id = "echo_enabled"]
    pub echo_enabled: BoolParam,

    #[id = "echo_strength"]
    pub echo_strength: FloatParam,

    // ═══════════════════════════════════════════════════════════════════════
    // DE-ESSER
    // ═══════════════════════════════════════════════════════════════════════
    #[id = "deesser_enabled"]
    pub deesser_enabled: BoolParam,

    #[id = "deesser_threshold"]
    pub deesser_threshold: FloatParam,

    #[id = "deesser_frequency"]
    pub deesser_frequency: FloatParam,

    #[id = "deesser_ratio"]
    pub deesser_ratio: FloatParam,

    // ═══════════════════════════════════════════════════════════════════════
    // PARAMETRIC EQ
    // ═══════════════════════════════════════════════════════════════════════
    #[id = "eq_enabled"]
    pub eq_enabled: BoolParam,

    // Band 1: Low Shelf (typically 80-120 Hz)
    #[id = "eq_band1_enabled"]
    pub eq_band1_enabled: BoolParam,
    #[id = "eq_band1_freq"]
    pub eq_band1_freq: FloatParam,
    #[id = "eq_band1_gain"]
    pub eq_band1_gain: FloatParam,

    // Band 2: Low-Mid (typically 200-500 Hz)
    #[id = "eq_band2_enabled"]
    pub eq_band2_enabled: BoolParam,
    #[id = "eq_band2_freq"]
    pub eq_band2_freq: FloatParam,
    #[id = "eq_band2_gain"]
    pub eq_band2_gain: FloatParam,
    #[id = "eq_band2_q"]
    pub eq_band2_q: FloatParam,

    // Band 3: Mid (typically 500-2000 Hz)
    #[id = "eq_band3_enabled"]
    pub eq_band3_enabled: BoolParam,
    #[id = "eq_band3_freq"]
    pub eq_band3_freq: FloatParam,
    #[id = "eq_band3_gain"]
    pub eq_band3_gain: FloatParam,
    #[id = "eq_band3_q"]
    pub eq_band3_q: FloatParam,

    // Band 4: High-Mid (typically 2000-5000 Hz)
    #[id = "eq_band4_enabled"]
    pub eq_band4_enabled: BoolParam,
    #[id = "eq_band4_freq"]
    pub eq_band4_freq: FloatParam,
    #[id = "eq_band4_gain"]
    pub eq_band4_gain: FloatParam,
    #[id = "eq_band4_q"]
    pub eq_band4_q: FloatParam,

    // Band 5: High Shelf (typically 8000+ Hz)
    #[id = "eq_band5_enabled"]
    pub eq_band5_enabled: BoolParam,
    #[id = "eq_band5_freq"]
    pub eq_band5_freq: FloatParam,
    #[id = "eq_band5_gain"]
    pub eq_band5_gain: FloatParam,

    // ═══════════════════════════════════════════════════════════════════════
    // COMPRESSOR
    // ═══════════════════════════════════════════════════════════════════════
    #[id = "comp_enabled"]
    pub comp_enabled: BoolParam,

    #[id = "comp_threshold"]
    pub comp_threshold: FloatParam,

    #[id = "comp_ratio"]
    pub comp_ratio: FloatParam,

    #[id = "comp_attack"]
    pub comp_attack: FloatParam,

    #[id = "comp_release"]
    pub comp_release: FloatParam,

    #[id = "comp_knee"]
    pub comp_knee: FloatParam,

    #[id = "comp_makeup"]
    pub comp_makeup: FloatParam,

    // ═══════════════════════════════════════════════════════════════════════
    // LIMITER
    // ═══════════════════════════════════════════════════════════════════════
    #[id = "limiter_enabled"]
    pub limiter_enabled: BoolParam,

    #[id = "limiter_ceiling"]
    pub limiter_ceiling: FloatParam,

    // ═══════════════════════════════════════════════════════════════════════
    // OUTPUT
    // ═══════════════════════════════════════════════════════════════════════
    #[id = "output_gain"]
    pub output_gain: FloatParam,

    #[id = "dry_wet"]
    pub dry_wet: FloatParam,
}

impl Default for GhostWaveParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(800, 600),

            // ═══════════════════════════════════════════════════════════════
            // NOISE SUPPRESSION
            // ═══════════════════════════════════════════════════════════════
            noise_enabled: BoolParam::new("Noise Suppression", true),
            noise_strength: FloatParam::new(
                "Noise Strength",
                0.7,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),

            noise_gate_threshold: FloatParam::new(
                "Gate Threshold",
                -45.0,
                FloatRange::Linear {
                    min: -80.0,
                    max: 0.0,
                },
            )
            .with_unit(" dB"),

            // ═══════════════════════════════════════════════════════════════
            // ECHO REMOVAL
            // ═══════════════════════════════════════════════════════════════
            echo_enabled: BoolParam::new("Echo Removal", false),
            echo_strength: FloatParam::new(
                "Echo Strength",
                0.5,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),

            // ═══════════════════════════════════════════════════════════════
            // DE-ESSER
            // ═══════════════════════════════════════════════════════════════
            deesser_enabled: BoolParam::new("De-Esser", false),
            deesser_threshold: FloatParam::new(
                "De-Esser Threshold",
                -20.0,
                FloatRange::Linear {
                    min: -60.0,
                    max: 0.0,
                },
            )
            .with_unit(" dB"),
            deesser_frequency: FloatParam::new(
                "De-Esser Frequency",
                6500.0,
                FloatRange::Skewed {
                    min: 2000.0,
                    max: 12000.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_unit(" Hz"),
            deesser_ratio: FloatParam::new(
                "De-Esser Ratio",
                4.0,
                FloatRange::Skewed {
                    min: 1.0,
                    max: 20.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(":1"),

            // ═══════════════════════════════════════════════════════════════
            // PARAMETRIC EQ
            // ═══════════════════════════════════════════════════════════════
            eq_enabled: BoolParam::new("EQ", false),

            // Band 1: Low Shelf
            eq_band1_enabled: BoolParam::new("EQ Band 1", true),
            eq_band1_freq: FloatParam::new(
                "EQ Band 1 Freq",
                100.0,
                FloatRange::Skewed {
                    min: 20.0,
                    max: 500.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" Hz"),
            eq_band1_gain: FloatParam::new(
                "EQ Band 1 Gain",
                0.0,
                FloatRange::Linear {
                    min: -18.0,
                    max: 18.0,
                },
            )
            .with_unit(" dB"),

            // Band 2: Low-Mid
            eq_band2_enabled: BoolParam::new("EQ Band 2", true),
            eq_band2_freq: FloatParam::new(
                "EQ Band 2 Freq",
                350.0,
                FloatRange::Skewed {
                    min: 100.0,
                    max: 1000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" Hz"),
            eq_band2_gain: FloatParam::new(
                "EQ Band 2 Gain",
                0.0,
                FloatRange::Linear {
                    min: -18.0,
                    max: 18.0,
                },
            )
            .with_unit(" dB"),
            eq_band2_q: FloatParam::new(
                "EQ Band 2 Q",
                1.0,
                FloatRange::Skewed {
                    min: 0.1,
                    max: 10.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            ),

            // Band 3: Mid
            eq_band3_enabled: BoolParam::new("EQ Band 3", true),
            eq_band3_freq: FloatParam::new(
                "EQ Band 3 Freq",
                1000.0,
                FloatRange::Skewed {
                    min: 300.0,
                    max: 3000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" Hz"),
            eq_band3_gain: FloatParam::new(
                "EQ Band 3 Gain",
                0.0,
                FloatRange::Linear {
                    min: -18.0,
                    max: 18.0,
                },
            )
            .with_unit(" dB"),
            eq_band3_q: FloatParam::new(
                "EQ Band 3 Q",
                1.0,
                FloatRange::Skewed {
                    min: 0.1,
                    max: 10.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            ),

            // Band 4: High-Mid
            eq_band4_enabled: BoolParam::new("EQ Band 4", true),
            eq_band4_freq: FloatParam::new(
                "EQ Band 4 Freq",
                3500.0,
                FloatRange::Skewed {
                    min: 1000.0,
                    max: 8000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" Hz"),
            eq_band4_gain: FloatParam::new(
                "EQ Band 4 Gain",
                0.0,
                FloatRange::Linear {
                    min: -18.0,
                    max: 18.0,
                },
            )
            .with_unit(" dB"),
            eq_band4_q: FloatParam::new(
                "EQ Band 4 Q",
                1.0,
                FloatRange::Skewed {
                    min: 0.1,
                    max: 10.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            ),

            // Band 5: High Shelf
            eq_band5_enabled: BoolParam::new("EQ Band 5", true),
            eq_band5_freq: FloatParam::new(
                "EQ Band 5 Freq",
                8000.0,
                FloatRange::Skewed {
                    min: 4000.0,
                    max: 20000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" Hz"),
            eq_band5_gain: FloatParam::new(
                "EQ Band 5 Gain",
                0.0,
                FloatRange::Linear {
                    min: -18.0,
                    max: 18.0,
                },
            )
            .with_unit(" dB"),

            // ═══════════════════════════════════════════════════════════════
            // COMPRESSOR
            // ═══════════════════════════════════════════════════════════════
            comp_enabled: BoolParam::new("Compressor", false),
            comp_threshold: FloatParam::new(
                "Comp Threshold",
                -18.0,
                FloatRange::Linear {
                    min: -60.0,
                    max: 0.0,
                },
            )
            .with_unit(" dB"),
            comp_ratio: FloatParam::new(
                "Comp Ratio",
                4.0,
                FloatRange::Skewed {
                    min: 1.0,
                    max: 20.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(":1"),
            comp_attack: FloatParam::new(
                "Comp Attack",
                10.0,
                FloatRange::Skewed {
                    min: 0.1,
                    max: 100.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" ms"),
            comp_release: FloatParam::new(
                "Comp Release",
                100.0,
                FloatRange::Skewed {
                    min: 10.0,
                    max: 1000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" ms"),
            comp_knee: FloatParam::new(
                "Comp Knee",
                6.0,
                FloatRange::Linear { min: 0.0, max: 24.0 },
            )
            .with_unit(" dB"),
            comp_makeup: FloatParam::new(
                "Comp Makeup",
                0.0,
                FloatRange::Linear {
                    min: -12.0,
                    max: 24.0,
                },
            )
            .with_unit(" dB"),

            // ═══════════════════════════════════════════════════════════════
            // LIMITER
            // ═══════════════════════════════════════════════════════════════
            limiter_enabled: BoolParam::new("Limiter", true),
            limiter_ceiling: FloatParam::new(
                "Limiter Ceiling",
                -0.3,
                FloatRange::Linear {
                    min: -12.0,
                    max: 0.0,
                },
            )
            .with_unit(" dB"),

            // ═══════════════════════════════════════════════════════════════
            // OUTPUT
            // ═══════════════════════════════════════════════════════════════
            output_gain: FloatParam::new(
                "Output Gain",
                0.0,
                FloatRange::Linear {
                    min: -24.0,
                    max: 24.0,
                },
            )
            .with_unit(" dB"),
            dry_wet: FloatParam::new("Dry/Wet", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit(" %")
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
        }
    }
}
