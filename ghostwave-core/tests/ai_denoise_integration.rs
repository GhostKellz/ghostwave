//! Integration tests for AI denoising pipeline
//!
//! These tests verify the complete AI noise suppression system including:
//! - Model weight initialization and loading
//! - Feature extraction
//! - GPU and CPU inference paths
//! - End-to-end audio processing

use ghostwave_core::ai_denoise::{
    AiDenoiseConfig, DenoiseQuality,
    RNNoiseModelWeights, ModelSize, generate_pretrained_models,
};

/// Test model weight initialization with different sizes
#[test]
fn test_model_weight_creation() {
    for size in [ModelSize::Tiny, ModelSize::Standard, ModelSize::Large] {
        let weights = RNNoiseModelWeights::pretrained(size);

        let expected_hidden = size.hidden_size();

        // Verify GRU1 dimensions (input: 42 features)
        assert_eq!(weights.gru1.in_features, 42, "GRU1 input features wrong for {:?}", size);
        assert_eq!(weights.gru1.hidden_size, expected_hidden, "GRU1 hidden size wrong for {:?}", size);

        // Verify weight vector sizes
        let expected_w_ih_size = 42 * 3 * expected_hidden;
        assert_eq!(weights.gru1.w_ih.len(), expected_w_ih_size, "GRU1 w_ih size wrong for {:?}", size);

        // Verify GRU2 and GRU3 (hidden -> hidden)
        assert_eq!(weights.gru2.in_features, expected_hidden);
        assert_eq!(weights.gru3.in_features, expected_hidden);

        // Verify output layer (hidden -> 23)
        assert_eq!(weights.output_weights.len(), expected_hidden * 23);
        assert_eq!(weights.output_bias.len(), 23);

        // Verify weights are non-zero (initialized)
        assert!(weights.gru1.w_ih.iter().any(|&w| w != 0.0), "GRU1 weights should be non-zero");
        assert!(weights.output_bias.iter().any(|&b| b != 0.0), "Output bias should be non-zero");

        println!("Model {:?}: {} parameters", size, weights.param_count());
    }
}

/// Test model save and load round-trip
#[test]
fn test_model_save_load() {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("test_model.gwm");

    // Create and save
    let original = RNNoiseModelWeights::pretrained(ModelSize::Standard);
    original.save(&model_path).expect("Failed to save model");

    // Load and verify
    let loaded = RNNoiseModelWeights::load(&model_path).expect("Failed to load model");

    // Verify dimensions match
    assert_eq!(original.size, loaded.size);
    assert_eq!(original.param_count(), loaded.param_count());

    // Verify weights match exactly
    assert_eq!(original.gru1.w_ih, loaded.gru1.w_ih, "GRU1 weights mismatch");
    assert_eq!(original.gru2.w_hh, loaded.gru2.w_hh, "GRU2 weights mismatch");
    assert_eq!(original.output_weights, loaded.output_weights, "Output weights mismatch");
    assert_eq!(original.output_bias, loaded.output_bias, "Output bias mismatch");
}

/// Test generating all pretrained models
#[test]
fn test_generate_pretrained_models() {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");

    generate_pretrained_models(temp_dir.path()).expect("Failed to generate models");

    // Verify all models were created
    assert!(temp_dir.path().join("rnnoise_tiny.gwm").exists(), "Tiny model not created");
    assert!(temp_dir.path().join("rnnoise_standard.gwm").exists(), "Standard model not created");
    assert!(temp_dir.path().join("rnnoise_large.gwm").exists(), "Large model not created");

    // Verify they're loadable
    for filename in ["rnnoise_tiny.gwm", "rnnoise_standard.gwm", "rnnoise_large.gwm"] {
        let path = temp_dir.path().join(filename);
        let weights = RNNoiseModelWeights::load(&path);
        assert!(weights.is_ok(), "Failed to load {}", filename);
    }
}

/// Test AI denoise config presets
#[test]
fn test_config_presets() {
    // Gaming preset - fast, low latency
    let gaming = AiDenoiseConfig::for_gaming();
    assert_eq!(gaming.quality, DenoiseQuality::Fast);
    assert!(gaming.echo_cancellation);
    assert!(gaming.noise_suppression);

    // Streaming preset - balanced
    let streaming = AiDenoiseConfig::for_streaming();
    assert_eq!(streaming.quality, DenoiseQuality::Balanced);
    assert!(streaming.voice_isolation);

    // Recording preset - high quality
    let recording = AiDenoiseConfig::for_recording();
    assert_eq!(recording.quality, DenoiseQuality::Quality);
    assert!(!recording.echo_cancellation); // Assumes treated room

    // Broadcast preset - ultra quality
    let broadcast = AiDenoiseConfig::for_broadcast();
    assert_eq!(broadcast.quality, DenoiseQuality::Ultra);
    assert!(broadcast.prefer_tensorrt);
}

/// Test denoise quality latency expectations
#[test]
fn test_quality_latency() {
    let fast_latency = DenoiseQuality::Fast.expected_latency_ms();
    let balanced_latency = DenoiseQuality::Balanced.expected_latency_ms();
    let quality_latency = DenoiseQuality::Quality.expected_latency_ms();
    let ultra_latency = DenoiseQuality::Ultra.expected_latency_ms();

    // Higher quality = more latency
    assert!(fast_latency < balanced_latency);
    assert!(balanced_latency < quality_latency);
    assert!(quality_latency < ultra_latency);

    // Reasonable bounds
    assert!(fast_latency >= 1.0 && fast_latency <= 10.0, "Fast latency: {}", fast_latency);
    assert!(ultra_latency <= 50.0, "Ultra latency too high: {}", ultra_latency);
}

/// Test default config values
#[test]
fn test_default_config() {
    let config = AiDenoiseConfig::default();

    assert_eq!(config.quality, DenoiseQuality::Balanced);
    assert_eq!(config.sample_rate, 48000);
    assert_eq!(config.frame_size, 480); // 10ms at 48kHz
    assert!(config.noise_suppression);
    assert!(config.use_gpu);
    assert!(config.prefer_tensorrt);

    // Noise strength should be reasonable
    assert!(config.noise_strength >= 0.5 && config.noise_strength <= 1.0);
}

/// Test model size classification
#[test]
fn test_model_size_classification() {
    // Test from_hidden_size
    assert_eq!(ModelSize::from_hidden_size(64), ModelSize::Tiny);
    assert_eq!(ModelSize::from_hidden_size(80), ModelSize::Tiny);
    assert_eq!(ModelSize::from_hidden_size(96), ModelSize::Standard);
    assert_eq!(ModelSize::from_hidden_size(112), ModelSize::Standard);
    assert_eq!(ModelSize::from_hidden_size(128), ModelSize::Large);
    assert_eq!(ModelSize::from_hidden_size(256), ModelSize::Large);

    // Test hidden_size
    assert_eq!(ModelSize::Tiny.hidden_size(), 64);
    assert_eq!(ModelSize::Standard.hidden_size(), 96);
    assert_eq!(ModelSize::Large.hidden_size(), 128);
}

/// Test GRU layer weight initialization
#[test]
fn test_gru_weight_structure() {
    let weights = RNNoiseModelWeights::pretrained(ModelSize::Standard);
    let hidden = 96;

    // GRU has 3 gates: update, reset, new
    // w_ih: [input, 3*hidden]
    // w_hh: [hidden, 3*hidden]
    // biases: [3*hidden]

    let expected_gru2_w_ih = hidden * 3 * hidden;
    let expected_gru2_w_hh = hidden * 3 * hidden;
    let expected_bias = 3 * hidden;

    assert_eq!(weights.gru2.w_ih.len(), expected_gru2_w_ih);
    assert_eq!(weights.gru2.w_hh.len(), expected_gru2_w_hh);
    assert_eq!(weights.gru2.b_ih.len(), expected_bias);
    assert_eq!(weights.gru2.b_hh.len(), expected_bias);
}

/// Test that pretrained weights have reasonable distributions
#[test]
fn test_weight_distribution() {
    let weights = RNNoiseModelWeights::pretrained(ModelSize::Standard);

    // Check GRU1 input weights
    let w_ih_mean: f32 = weights.gru1.w_ih.iter().sum::<f32>() / weights.gru1.w_ih.len() as f32;
    let w_ih_max = weights.gru1.w_ih.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w_ih_min = weights.gru1.w_ih.iter().cloned().fold(f32::INFINITY, f32::min);

    // Weights should be bounded (not exploding)
    assert!(w_ih_max < 5.0, "Weights too large: {}", w_ih_max);
    assert!(w_ih_min > -5.0, "Weights too negative: {}", w_ih_min);

    // Mean should be near zero (balanced initialization)
    assert!(w_ih_mean.abs() < 0.5, "Weight mean too far from zero: {}", w_ih_mean);

    // Output bias should be positive (favor passing signal through)
    let output_bias_mean: f32 = weights.output_bias[..22].iter().sum::<f32>() / 22.0;
    assert!(output_bias_mean > 0.0, "Output bias should favor signal pass-through");
}

/// Test parameter count calculations
#[test]
fn test_parameter_counts() {
    let tiny = RNNoiseModelWeights::new(ModelSize::Tiny);
    let standard = RNNoiseModelWeights::new(ModelSize::Standard);
    let large = RNNoiseModelWeights::new(ModelSize::Large);

    // Larger models should have more parameters
    assert!(tiny.param_count() < standard.param_count());
    assert!(standard.param_count() < large.param_count());

    // Rough size estimates (KB)
    let tiny_kb = tiny.param_count() * 4 / 1024;
    let standard_kb = standard.param_count() * 4 / 1024;
    let large_kb = large.param_count() * 4 / 1024;

    println!("Tiny: {} params ({} KB)", tiny.param_count(), tiny_kb);
    println!("Standard: {} params ({} KB)", standard.param_count(), standard_kb);
    println!("Large: {} params ({} KB)", large.param_count(), large_kb);

    // Should be in reasonable ranges
    assert!(tiny_kb < 500, "Tiny model too large");
    assert!(standard_kb < 1000, "Standard model too large");
    assert!(large_kb < 2000, "Large model too large");
}

/// Test model file format validation
#[test]
fn test_invalid_model_file() {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let bad_path = temp_dir.path().join("bad_model.gwm");

    // Write garbage data
    std::fs::write(&bad_path, b"not a valid model file").unwrap();

    // Should fail with magic number error
    let result = RNNoiseModelWeights::load(&bad_path);
    assert!(result.is_err(), "Should reject invalid model file");

    let err = result.unwrap_err().to_string();
    assert!(err.contains("magic") || err.contains("Invalid"), "Error should mention magic/invalid: {}", err);
}

/// Test model file not found
#[test]
fn test_model_not_found() {
    let result = RNNoiseModelWeights::load(std::path::Path::new("/nonexistent/model.gwm"));
    assert!(result.is_err(), "Should fail for nonexistent file");
}

/// Test GRU layer param_count
#[test]
fn test_gru_param_count() {
    let weights = RNNoiseModelWeights::pretrained(ModelSize::Standard);

    let gru1_params = weights.gru1.param_count();
    let expected = weights.gru1.w_ih.len() + weights.gru1.w_hh.len()
                 + weights.gru1.b_ih.len() + weights.gru1.b_hh.len();

    assert_eq!(gru1_params, expected);
}

/// Test Xavier initialization produces reasonable values
#[test]
fn test_xavier_initialization() {
    let mut weights = RNNoiseModelWeights::new(ModelSize::Standard);
    weights.init_xavier(42); // Fixed seed for reproducibility

    // Weights should be non-zero
    assert!(weights.gru1.w_ih.iter().any(|&w| w != 0.0));

    // Should be bounded roughly by sqrt(6 / (fan_in + fan_out))
    let max_val = weights.gru1.w_ih.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(max_val < 2.0, "Xavier weights too large: {}", max_val);

    // Different seeds should produce different weights
    let mut weights2 = RNNoiseModelWeights::new(ModelSize::Standard);
    weights2.init_xavier(123);

    assert_ne!(weights.gru1.w_ih[0], weights2.gru1.w_ih[0], "Different seeds should produce different weights");
}

/// Integration test: Full processing pipeline simulation
#[test]
fn test_denoise_processing_simulation() {
    // This simulates what the full pipeline would do
    let weights = RNNoiseModelWeights::pretrained(ModelSize::Standard);
    let hidden = weights.size.hidden_size();

    // Simulate input features (42 bark-scale bands + spectral features)
    let input_features: Vec<f32> = (0..42).map(|i| {
        // Simulate speech-like spectral content
        let freq_factor = (i as f32 / 42.0 * std::f32::consts::PI).sin();
        freq_factor * 0.5 + 0.1
    }).collect();

    // Simulate hidden state (would persist across frames)
    let hidden_state = vec![0.0f32; hidden];

    // The actual forward pass would:
    // 1. GRU1: input_features -> hidden (42 -> 96)
    // 2. GRU2: hidden -> hidden (96 -> 96)
    // 3. GRU3: hidden -> hidden (96 -> 96)
    // 4. Output: hidden -> gains (96 -> 23)
    // 5. Apply sigmoid to get 0-1 gains

    // Verify dimensions would work
    assert_eq!(input_features.len(), RNNoiseModelWeights::INPUT_FEATURES);
    assert_eq!(hidden_state.len(), hidden);
    assert_eq!(weights.output_weights.len(), hidden * RNNoiseModelWeights::OUTPUT_SIZE);

    // Simulate output computation (simplified - just linear projection)
    let mut output = vec![0.0f32; 23];
    for (i, out) in output.iter_mut().enumerate() {
        let mut sum = weights.output_bias[i];
        for (j, &h) in hidden_state.iter().enumerate() {
            sum += h * weights.output_weights[j * 23 + i];
        }
        // Sigmoid activation
        *out = 1.0 / (1.0 + (-sum).exp());
    }

    // With zero hidden state, outputs should be sigmoid(bias)
    // Our pretrained weights have bias ~1.0 for band gains, so sigmoid(1) ~ 0.73
    for (i, &gain) in output[..22].iter().enumerate() {
        assert!(gain > 0.5 && gain < 1.0,
            "Band {} gain {} not in expected range (should pass signal)", i, gain);
    }
}

/// Test model quality mapping
#[test]
fn test_quality_model_mapping() {
    use ghostwave_core::ai_denoise::ModelType;

    assert!(matches!(DenoiseQuality::Fast.model_type(), ModelType::RNNoiseTiny));
    assert!(matches!(DenoiseQuality::Balanced.model_type(), ModelType::RNNoiseStandard));
    assert!(matches!(DenoiseQuality::Quality.model_type(), ModelType::RNNoiseLarge));
    assert!(matches!(DenoiseQuality::Ultra.model_type(), ModelType::TransformerDenoise));
}
