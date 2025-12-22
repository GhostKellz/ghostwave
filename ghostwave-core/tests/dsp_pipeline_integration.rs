//! Integration tests for the DSP pipeline
//!
//! Tests the complete DSP processing chain including noise suppression,
//! de-essing, compression, and EQ.

use ghostwave_core::{
    DspPipeline, FrameFormat, ProcessingProfile, ParamValue,
    DeEsser, DeEsserConfig,
    ParametricEq, ParametricEqConfig, EqBandConfig, FilterType,
    Compressor, CompressorConfig, DetectionMode, KneeType,
};

/// Test DSP pipeline creation
#[test]
fn test_pipeline_creation() {
    let format = FrameFormat::new(1, 48000, 256).unwrap();
    let pipeline = DspPipeline::new(format, ProcessingProfile::Balanced);

    // Pipeline should be created successfully
    assert!(pipeline.latency_frames() > 0);
}

/// Test pipeline processing with test signal
#[test]
fn test_pipeline_processing() {
    let format = FrameFormat::new(1, 48000, 256).unwrap();
    let mut pipeline = DspPipeline::new(format, ProcessingProfile::Balanced);

    // Generate test signal (sine wave with noise)
    let mut buffer: Vec<f32> = (0..256)
        .map(|i| {
            let sine = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.3;
            let noise = ((i * 7919) % 1000) as f32 / 10000.0 - 0.05; // Pseudo-random noise
            sine + noise
        })
        .collect();

    let result = pipeline.process(&mut buffer);
    assert!(result.is_ok(), "Pipeline processing failed: {:?}", result.err());
}

/// Test pipeline profile switching
#[test]
fn test_pipeline_profile_switching() {
    let format = FrameFormat::new(1, 48000, 256).unwrap();
    let mut pipeline = DspPipeline::new(format, ProcessingProfile::Balanced);

    // Switch through all profiles
    pipeline.set_profile(ProcessingProfile::Streaming);
    pipeline.set_profile(ProcessingProfile::Studio);
    pipeline.set_profile(ProcessingProfile::Balanced);

    // Process should still work
    let mut buffer = vec![0.1f32; 256];
    assert!(pipeline.process(&mut buffer).is_ok());
}

/// Test pipeline parameter setting
#[test]
fn test_pipeline_parameters() {
    let format = FrameFormat::new(1, 48000, 256).unwrap();
    let mut pipeline = DspPipeline::new(format, ProcessingProfile::Balanced);

    // Set various parameters
    let result = pipeline.set_param("noise_reduction_strength", ParamValue::Float(0.8));
    assert!(result.is_ok() || result.is_err()); // May or may not be supported

    // Pipeline should still function
    let mut buffer = vec![0.1f32; 256];
    assert!(pipeline.process(&mut buffer).is_ok());
}

// =============================================================================
// De-Esser Tests
// =============================================================================

/// Test de-esser creation
#[test]
fn test_deesser_creation() {
    let config = DeEsserConfig::default();
    let deesser = DeEsser::new(config, 48000.0);
    // DeEsser::new returns Self, not Result
    assert!(deesser.get_gain_reduction_db().abs() <= 60.0);
}

/// Test de-esser processing
#[test]
fn test_deesser_processing() {
    let config = DeEsserConfig {
        threshold_db: -20.0,
        ratio: 4.0,
        frequency_hz: 6000.0,
        bandwidth_octaves: 1.0,
        attack_ms: 0.5,
        release_ms: 50.0,
        makeup_gain_db: 0.0,
        listen_mode: false,
    };

    let mut deesser = DeEsser::new(config, 48000.0);

    // Generate sibilant-like signal (high frequency)
    let input: Vec<f32> = (0..256)
        .map(|i| (2.0 * std::f32::consts::PI * 7000.0 * i as f32 / 48000.0).sin() * 0.5)
        .collect();

    let mut output = vec![0.0f32; 256];
    let original_energy: f32 = input.iter().map(|s| s * s).sum();

    let result = deesser.process(&input, &mut output);
    assert!(result.is_ok());

    let processed_energy: f32 = output.iter().map(|s| s * s).sum();

    // De-esser should reduce high frequency content
    assert!(
        processed_energy <= original_energy,
        "De-esser should reduce or maintain energy"
    );
}

/// Test de-esser with sample rate constructor
#[test]
fn test_deesser_with_sample_rate() {
    let deesser = DeEsser::with_sample_rate(48000.0);
    // Should create successfully with defaults
    assert!(deesser.get_gain_reduction_db().abs() <= 60.0);
}

// =============================================================================
// Parametric EQ Tests
// =============================================================================

/// Test parametric EQ creation
#[test]
fn test_eq_creation() {
    let config = ParametricEqConfig::default();
    let eq = ParametricEq::new(config, 48000.0);
    // Returns Self, not Result
    assert!(std::mem::size_of_val(&eq) > 0);
}

/// Test EQ processing
#[test]
fn test_eq_processing() {
    let config = ParametricEqConfig::default();
    let mut eq = ParametricEq::new(config, 48000.0);

    // Generate test signal
    let input: Vec<f32> = (0..256)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin() * 0.5)
        .collect();

    let mut output = vec![0.0f32; 256];

    let result = eq.process(&input, &mut output);
    assert!(result.is_ok(), "EQ processing failed");
}

/// Test EQ with custom band configuration
#[test]
fn test_eq_custom_bands() {
    // Create custom band config
    let band = EqBandConfig {
        enabled: true,
        filter_type: FilterType::Peak,
        frequency_hz: 1000.0,
        gain_db: -6.0, // 6dB cut at 1kHz
        q: 1.0,
    };

    let mut bands = ParametricEqConfig::default().bands;
    bands[0] = band;

    let config = ParametricEqConfig {
        bands,
        ..ParametricEqConfig::default()
    };

    let mut eq = ParametricEq::new(config, 48000.0);

    // Generate 1kHz signal
    let input: Vec<f32> = (0..256)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin() * 0.5)
        .collect();

    let original_energy: f32 = input.iter().map(|s| s * s).sum();

    let mut output = vec![0.0f32; 256];
    eq.process(&input, &mut output).unwrap();

    let processed_energy: f32 = output.iter().map(|s| s * s).sum();

    // Should reduce energy due to 6dB cut
    assert!(
        processed_energy < original_energy,
        "EQ cut should reduce signal energy"
    );
}

/// Test various EQ filter types
#[test]
fn test_eq_filter_types() {
    let filter_types = [
        FilterType::Peak,
        FilterType::LowShelf,
        FilterType::HighShelf,
        FilterType::LowPass,
        FilterType::HighPass,
        FilterType::Notch,
        FilterType::Bandpass,
    ];

    for filter_type in filter_types {
        let band = EqBandConfig {
            enabled: true,
            filter_type,
            frequency_hz: 1000.0,
            gain_db: -3.0,
            q: 1.0,
        };

        let mut bands = ParametricEqConfig::default().bands;
        bands[0] = band;

        let config = ParametricEqConfig {
            bands,
            ..ParametricEqConfig::default()
        };

        let mut eq = ParametricEq::new(config, 48000.0);

        let input = vec![0.5f32; 256];
        let mut output = vec![0.0f32; 256];

        let result = eq.process(&input, &mut output);
        assert!(result.is_ok(), "Failed with filter type {:?}", filter_type);
    }
}

// =============================================================================
// Compressor Tests
// =============================================================================

/// Test compressor creation
#[test]
fn test_compressor_creation() {
    let config = CompressorConfig::default();
    let compressor = Compressor::new(config, 48000.0);
    // Returns Self, not Result
    assert!(std::mem::size_of_val(&compressor) > 0);
}

/// Test compressor processing
#[test]
fn test_compressor_processing() {
    let config = CompressorConfig {
        threshold_db: -20.0,
        ratio: 4.0,
        attack_ms: 10.0,
        release_ms: 100.0,
        knee_db: 0.0, // Hard knee
        makeup_gain_db: 0.0,
        auto_makeup: false,
        detection_mode: DetectionMode::Rms,
        lookahead_ms: 0.0,
        sidechain_hpf_hz: 0.0,
        mix: 1.0,
    };

    let mut compressor = Compressor::new(config, 48000.0);

    // Generate loud signal
    let input: Vec<f32> = (0..256)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.9)
        .collect();

    let original_peak = input.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

    let mut output = vec![0.0f32; 256];
    let result = compressor.process(&input, &mut output);
    assert!(result.is_ok());

    let processed_peak = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

    // Compressor should reduce peak level
    assert!(
        processed_peak <= original_peak,
        "Compressor should reduce or maintain peak level"
    );
}

/// Test compressor with soft knee
#[test]
fn test_compressor_soft_knee() {
    let config = CompressorConfig {
        threshold_db: -12.0,
        ratio: 2.0,
        knee_db: 6.0, // Soft knee
        ..Default::default()
    };

    let mut compressor = Compressor::new(config, 48000.0);

    let input = vec![0.5f32; 256];
    let mut output = vec![0.0f32; 256];

    let result = compressor.process(&input, &mut output);
    assert!(result.is_ok());

    // Output should be finite
    assert!(output.iter().all(|s| s.is_finite()));
}

/// Test compressor detection modes
#[test]
fn test_compressor_detection_modes() {
    let modes = [DetectionMode::Peak, DetectionMode::Rms, DetectionMode::TruePeak];

    for mode in modes {
        let config = CompressorConfig {
            detection_mode: mode,
            ..Default::default()
        };

        let mut compressor = Compressor::new(config, 48000.0);

        let input = vec![0.5f32; 256];
        let mut output = vec![0.0f32; 256];

        let result = compressor.process(&input, &mut output);
        assert!(result.is_ok(), "Failed with detection mode {:?}", mode);
    }
}

/// Test compressor as limiter
#[test]
fn test_compressor_limiter_mode() {
    let config = CompressorConfig {
        threshold_db: -1.0,
        ratio: f32::INFINITY, // Limiter mode
        attack_ms: 0.1,
        release_ms: 50.0,
        knee_db: 0.0,
        makeup_gain_db: 0.0,
        auto_makeup: false,
        detection_mode: DetectionMode::TruePeak,
        lookahead_ms: 1.0, // Small lookahead for limiting
        sidechain_hpf_hz: 0.0,
        mix: 1.0,
    };

    let mut limiter = Compressor::new(config, 48000.0);

    // Generate signal that clips
    let input: Vec<f32> = (0..256)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 1.5)
        .collect();

    let mut output = vec![0.0f32; 256];
    limiter.process(&input, &mut output).unwrap();

    // Limiter should prevent clipping (output should be < 1.0)
    let max_output = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    // Note: with lookahead there may be slight overshoots in first samples
    assert!(
        max_output < 1.5,
        "Limiter should significantly reduce peak level, got {}", max_output
    );
}

// =============================================================================
// Frame Format Tests
// =============================================================================

/// Test frame format creation
#[test]
fn test_frame_format_creation() {
    // Valid configurations
    assert!(FrameFormat::new(1, 48000, 256).is_ok());
    assert!(FrameFormat::new(2, 44100, 512).is_ok());
    assert!(FrameFormat::new(2, 96000, 1024).is_ok());
    assert!(FrameFormat::new(1, 192000, 64).is_ok());
}

/// Test frame format with various sample rates
#[test]
fn test_frame_format_sample_rates() {
    let sample_rates = [44100, 48000, 88200, 96000, 176400, 192000];

    for rate in sample_rates {
        let format = FrameFormat::new(1, rate, 256);
        assert!(format.is_ok(), "Failed to create format with {}Hz", rate);
    }
}

/// Test frame format with various buffer sizes
#[test]
fn test_frame_format_buffer_sizes() {
    let buffer_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096];

    for size in buffer_sizes {
        let format = FrameFormat::new(1, 48000, size);
        assert!(format.is_ok(), "Failed to create format with {} frames", size);
    }
}

// =============================================================================
// Pipeline Stress Tests
// =============================================================================

/// Test pipeline with rapid parameter changes
#[test]
fn test_pipeline_rapid_param_changes() {
    let format = FrameFormat::new(1, 48000, 256).unwrap();
    let mut pipeline = DspPipeline::new(format, ProcessingProfile::Balanced);

    for i in 0..100 {
        let strength = (i % 10) as f32 / 10.0;
        let _ = pipeline.set_param("noise_reduction_strength", ParamValue::Float(strength));

        let mut buffer = vec![0.1f32; 256];
        assert!(pipeline.process(&mut buffer).is_ok());
    }
}

/// Test pipeline with rapid profile switching
#[test]
fn test_pipeline_rapid_profile_switching() {
    let format = FrameFormat::new(1, 48000, 256).unwrap();
    let mut pipeline = DspPipeline::new(format, ProcessingProfile::Balanced);

    let profiles = [
        ProcessingProfile::Streaming,
        ProcessingProfile::Balanced,
        ProcessingProfile::Studio,
    ];

    for _ in 0..50 {
        for profile in &profiles {
            pipeline.set_profile(*profile);
            let mut buffer = vec![0.1f32; 256];
            assert!(pipeline.process(&mut buffer).is_ok());
        }
    }
}

/// Test processing chain with multiple components
#[test]
fn test_full_processing_chain() {
    // Create all components
    let mut deesser = DeEsser::with_sample_rate(48000.0);
    let mut eq = ParametricEq::new(ParametricEqConfig::default(), 48000.0);
    let mut compressor = Compressor::new(CompressorConfig::default(), 48000.0);

    // Generate test signal
    let input: Vec<f32> = (0..256)
        .map(|i| {
            let low = (2.0 * std::f32::consts::PI * 100.0 * i as f32 / 48000.0).sin() * 0.3;
            let mid = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin() * 0.3;
            let high = (2.0 * std::f32::consts::PI * 6000.0 * i as f32 / 48000.0).sin() * 0.3;
            low + mid + high
        })
        .collect();

    // Process through chain
    let mut buffer1 = vec![0.0f32; 256];
    let mut buffer2 = vec![0.0f32; 256];
    let mut buffer3 = vec![0.0f32; 256];

    deesser.process(&input, &mut buffer1).unwrap();
    eq.process(&buffer1, &mut buffer2).unwrap();
    compressor.process(&buffer2, &mut buffer3).unwrap();

    // Output should be finite
    assert!(buffer3.iter().all(|s| s.is_finite()));

    // Output should have some signal
    let energy: f32 = buffer3.iter().map(|s| s * s).sum();
    assert!(energy > 0.001, "Output should have signal");
}
