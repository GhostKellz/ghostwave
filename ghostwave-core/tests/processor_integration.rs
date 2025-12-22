//! Integration tests for GhostWave audio processor
//!
//! These tests verify the complete processing pipeline works correctly,
//! including initialization, audio processing, and parameter management.

use ghostwave_core::{
    Config, GhostWaveProcessor, AudioProcessor, ProcessingProfile, ParamValue,
};

/// Test basic processor creation with default config
#[test]
fn test_processor_creation_default() {
    let config = Config::default();
    let processor = GhostWaveProcessor::new(config);
    assert!(processor.is_ok(), "Failed to create processor with default config");

    let processor = processor.unwrap();
    assert_eq!(processor.name(), "GhostWave");
    assert!(!processor.version().is_empty());
}

/// Test processor creation with different profiles
#[test]
fn test_processor_creation_profiles() {
    for profile in ["balanced", "streaming", "studio"] {
        let config = Config::load(profile).expect("Failed to load profile");
        let processor = GhostWaveProcessor::new(config);
        assert!(processor.is_ok(), "Failed to create processor with {} profile", profile);
    }
}

/// Test processor initialization
#[test]
fn test_processor_initialization() {
    let config = Config::default();
    let mut processor = GhostWaveProcessor::new(config).unwrap();

    // Processor should not be initialized yet
    assert!(!processor.is_initialized());

    // Initialize with standard parameters
    let result = processor.init(48000, 1, 256);
    assert!(result.is_ok(), "Initialization failed: {:?}", result.err());

    // Now should be initialized
    assert!(processor.is_initialized());
}

/// Test audio processing with silence
#[test]
fn test_process_silence() {
    let config = Config::default();
    let processor = GhostWaveProcessor::new(config).unwrap();

    let input = vec![0.0f32; 256];
    let mut output = vec![1.0f32; 256]; // Initialize with non-zero to verify change

    let result = processor.process(&input, &mut output);
    assert!(result.is_ok());

    // Output should be near-silent (noise gate may affect slightly)
    for sample in &output {
        assert!(sample.abs() < 0.01, "Expected silence, got: {}", sample);
    }
}

/// Test audio processing with sine wave
#[test]
fn test_process_sine_wave() {
    let mut config = Config::default();
    config.noise_suppression.enabled = true;
    config.noise_suppression.strength = 0.5;

    let processor = GhostWaveProcessor::new(config).unwrap();

    // Generate a 440Hz sine wave at 48kHz
    let sample_rate = 48000.0f32;
    let frequency = 440.0f32;
    let input: Vec<f32> = (0..256)
        .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / sample_rate).sin() * 0.5)
        .collect();

    let mut output = vec![0.0f32; 256];

    let result = processor.process(&input, &mut output);
    assert!(result.is_ok());

    // Output should have some signal (not completely zeroed)
    let output_energy: f32 = output.iter().map(|s| s * s).sum();
    assert!(output_energy > 0.01, "Expected signal output, got near-silence");
}

/// Test buffer size mismatch handling
#[test]
fn test_buffer_size_mismatch() {
    let config = Config::default();
    let processor = GhostWaveProcessor::new(config).unwrap();

    let input = vec![0.1f32; 256];
    let mut output = vec![0.0f32; 128]; // Mismatched size

    let result = processor.process(&input, &mut output);
    assert!(result.is_err(), "Should fail with mismatched buffer sizes");
}

/// Test processing with disabled noise suppression
#[test]
fn test_process_bypass() {
    let mut config = Config::default();
    config.noise_suppression.enabled = false;

    let processor = GhostWaveProcessor::new(config).unwrap();

    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.001).collect();
    let mut output = vec![0.0f32; 256];

    let result = processor.process(&input, &mut output);
    assert!(result.is_ok());

    // With bypass, output should exactly match input
    for (i, (inp, out)) in input.iter().zip(output.iter()).enumerate() {
        assert!((inp - out).abs() < f32::EPSILON,
            "Sample {} mismatch: input={}, output={}", i, inp, out);
    }
}

/// Test in-place processing
#[test]
fn test_process_inplace() {
    let config = Config::default();
    let mut processor = GhostWaveProcessor::new(config).unwrap();

    // Initialize processor first
    processor.init(48000, 1, 256).unwrap();

    // Generate test signal
    let mut buffer: Vec<f32> = (0..256)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.3)
        .collect();

    let result = processor.process_inplace(&mut buffer, 256);
    assert!(result.is_ok(), "In-place processing failed: {:?}", result.err());
}

/// Test profile switching
#[test]
fn test_profile_switching() {
    let config = Config::default();
    let mut processor = GhostWaveProcessor::new(config).unwrap();

    // Initialize first
    processor.init(48000, 1, 256).unwrap();

    // Test each profile
    for profile in [ProcessingProfile::Streaming, ProcessingProfile::Balanced, ProcessingProfile::Studio] {
        let result = processor.set_profile(profile);
        assert!(result.is_ok(), "Failed to set profile {:?}", profile);
        assert_eq!(processor.get_profile(), profile);
    }
}

/// Test parameter setting and getting
#[test]
fn test_parameter_management() {
    let config = Config::default();
    let mut processor = GhostWaveProcessor::new(config).unwrap();

    // Test setting noise reduction strength before init (stored in config)
    let result = processor.set_param("noise_reduction_strength", ParamValue::Float(0.9));
    assert!(result.is_ok(), "Failed to set noise_reduction_strength");

    // Verify the param is retrievable (may be from config or profile_params)
    let value = processor.get_param("noise_reduction_strength");
    assert!(value.is_ok(), "Failed to get noise_reduction_strength");

    if let ParamValue::Float(strength) = value.unwrap() {
        // Value should be in valid range
        assert!(strength >= 0.0 && strength <= 1.0, "Strength out of range: {}", strength);
    } else {
        panic!("Expected Float value");
    }

    // After initialization, params may be routed to DSP pipeline
    processor.init(48000, 1, 256).unwrap();

    // Set again after init
    let result = processor.set_param("noise_reduction_strength", ParamValue::Float(0.8));
    assert!(result.is_ok(), "Failed to set noise_reduction_strength after init");
}

/// Test parameter validation
#[test]
fn test_parameter_validation() {
    let config = Config::default();
    let mut processor = GhostWaveProcessor::new(config).unwrap();

    // Out of range value should fail
    let result = processor.set_param("noise_reduction_strength", ParamValue::Float(1.5));
    assert!(result.is_err(), "Should reject out-of-range value");

    let result = processor.set_param("noise_reduction_strength", ParamValue::Float(-0.1));
    assert!(result.is_err(), "Should reject negative value");

    // Unknown parameter should fail
    let result = processor.set_param("unknown_param", ParamValue::Float(0.5));
    assert!(result.is_err(), "Should reject unknown parameter");
}

/// Test processor reset
#[test]
fn test_processor_reset() {
    let config = Config::default();
    let mut processor = GhostWaveProcessor::new(config).unwrap();

    // Initialize
    processor.init(48000, 1, 256).unwrap();

    // Process some audio
    let mut buffer = vec![0.1f32; 256];
    processor.process_inplace(&mut buffer, 256).unwrap();

    // Reset should succeed
    let result = processor.reset();
    assert!(result.is_ok(), "Reset failed: {:?}", result.err());

    // Should still be able to process after reset
    let mut buffer = vec![0.1f32; 256];
    let result = processor.process_inplace(&mut buffer, 256);
    assert!(result.is_ok(), "Processing after reset failed");
}

/// Test getting parameter descriptors
#[test]
fn test_get_params() {
    let config = Config::default();
    let processor = GhostWaveProcessor::new(config).unwrap();

    let params = processor.get_params();

    // Should have noise_reduction_strength at minimum
    assert!(params.contains_key("noise_reduction_strength"), "Missing noise_reduction_strength");

    // Verify descriptor properties
    let nr_param = params.get("noise_reduction_strength").unwrap();
    assert!(nr_param.runtime_adjustable);
    assert_eq!(nr_param.category, "Noise Suppression");
}

/// Test latency reporting
#[test]
fn test_latency_frames() {
    let config = Config::default();
    let mut processor = GhostWaveProcessor::new(config).unwrap();

    // Initialize with 256 frame buffer
    processor.init(48000, 1, 256).unwrap();

    let latency = processor.latency_frames();
    assert!(latency > 0, "Latency should be > 0");
    assert!(latency <= 256, "Latency should not exceed buffer size");
}

/// Test processing mode reporting
#[test]
fn test_processing_mode() {
    let config = Config::default();
    let processor = GhostWaveProcessor::new(config).unwrap();

    let mode = processor.get_processing_mode();
    assert!(!mode.is_empty(), "Processing mode should not be empty");

    // Should be either RTX or CPU mode
    assert!(
        mode.contains("RTX") || mode.contains("CPU") || mode.contains("Spectral"),
        "Unexpected processing mode: {}", mode
    );
}

/// Test concurrent processing safety (basic)
#[test]
fn test_concurrent_process_calls() {
    use std::sync::Arc;
    use std::thread;

    let config = Config::default();
    let processor = Arc::new(GhostWaveProcessor::new(config).unwrap());

    let mut handles = vec![];

    for _ in 0..4 {
        let processor = Arc::clone(&processor);
        handles.push(thread::spawn(move || {
            let input = vec![0.1f32; 256];
            let mut output = vec![0.0f32; 256];
            for _ in 0..10 {
                let _ = processor.process(&input, &mut output);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test large buffer processing
#[test]
fn test_large_buffer_processing() {
    let mut config = Config::default();
    config.audio.buffer_size = 4096;

    let processor = GhostWaveProcessor::new(config).unwrap();

    let input = vec![0.1f32; 4096];
    let mut output = vec![0.0f32; 4096];

    let result = processor.process(&input, &mut output);
    assert!(result.is_ok(), "Large buffer processing failed");
}

/// Test small buffer processing
#[test]
fn test_small_buffer_processing() {
    let mut config = Config::default();
    config.audio.buffer_size = 64;

    let processor = GhostWaveProcessor::new(config).unwrap();

    let input = vec![0.1f32; 64];
    let mut output = vec![0.0f32; 64];

    let result = processor.process(&input, &mut output);
    assert!(result.is_ok(), "Small buffer processing failed");
}

/// Test RTX acceleration availability check
#[test]
fn test_rtx_availability() {
    let config = Config::default();
    let processor = GhostWaveProcessor::new(config).unwrap();

    // Just verify it doesn't panic - RTX may or may not be available
    let has_rtx = processor.has_rtx_acceleration();
    println!("RTX acceleration available: {}", has_rtx);
}

/// Test config access
#[test]
fn test_config_access() {
    let config = Config::load("studio").unwrap();
    let processor = GhostWaveProcessor::new(config).unwrap();

    let config = processor.get_config();
    assert_eq!(config.audio.sample_rate, 96000); // Studio profile uses 96kHz
}
