//! Integration tests for nnnoiseless-based noise reduction.

use ghostwave_core::{NnnoiseDenoiser, DspPipeline};
use ghostwave_core::frame_format::FrameFormat;
use ghostwave_core::processor::{ProcessingProfile, ParamValue};
use ghostwave_core::dsp_pipeline::DenoiserBackend;

/// Generate white noise in [-amplitude, amplitude].
fn white_noise(len: usize, amplitude: f32) -> Vec<f32> {
    // Deterministic pseudo-random for reproducibility
    let mut state: u32 = 0xDEAD_BEEF;
    (0..len)
        .map(|_| {
            // xorshift32
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let normalized = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            normalized * amplitude
        })
        .collect()
}

/// Generate a sine wave.
fn sine_wave(len: usize, frequency_hz: f32, sample_rate: f32, amplitude: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * std::f32::consts::PI * frequency_hz * t).sin() * amplitude
        })
        .collect()
}

/// RMS energy of a buffer.
fn rms(buffer: &[f32]) -> f32 {
    if buffer.is_empty() {
        return 0.0;
    }
    (buffer.iter().map(|&x| x * x).sum::<f32>() / buffer.len() as f32).sqrt()
}

// ── NnnoiseDenoiser adapter tests ───────────────────────────────────────

#[test]
fn test_nnnoiseless_denoises_noise() {
    let mut denoiser = NnnoiseDenoiser::new();

    // Warmup: process frames to fill internal state
    let mut warmup = white_noise(480 * 5, 0.3);
    denoiser.process(&mut warmup, true).unwrap();

    // Process many frames of pure noise and verify some reduction occurs.
    // RNNoise is designed for speech+noise separation; on pure broadband noise
    // it applies partial attenuation (not full suppression), so we use a
    // generous threshold here.
    let noise = white_noise(480 * 40, 0.3);
    let input_rms = rms(&noise);

    let mut buffer = noise;
    denoiser.process(&mut buffer, true).unwrap();

    let output_rms = rms(&buffer);

    assert!(
        output_rms < input_rms,
        "Expected some noise reduction: input_rms={:.4}, output_rms={:.4}",
        input_rms,
        output_rms
    );

    // Verify output is finite and not all zeros
    assert!(buffer.iter().all(|&x| x.is_finite()));
    assert!(buffer.iter().any(|&x| x.abs() > 1e-8));
}

#[test]
fn test_nnnoiseless_preserves_silence() {
    let mut denoiser = NnnoiseDenoiser::new();

    // Warmup with a frame of silence
    let mut warmup = vec![0.0; 480];
    denoiser.process(&mut warmup, false).unwrap();

    // Process more silence
    let mut silence = vec![0.0; 480 * 5];
    denoiser.process(&mut silence, false).unwrap();

    let output_rms = rms(&silence);
    assert!(
        output_rms < 0.001,
        "Silence should remain near-silent, got rms={:.6}",
        output_rms
    );
}

#[test]
fn test_nnnoiseless_various_buffer_sizes() {
    let mut denoiser = NnnoiseDenoiser::new();

    // Process with multiple different buffer sizes — all should work
    for &size in &[32, 64, 128, 256, 480, 512, 1024] {
        let mut buf = white_noise(size, 0.1);
        let result = denoiser.process(&mut buf, true);
        assert!(result.is_ok(), "Failed with buffer size {}", size);
    }
}

#[test]
fn test_nnnoiseless_strength_bypass() {
    let mut denoiser = NnnoiseDenoiser::new();
    denoiser.set_strength(0.0);

    // Warmup
    let mut warmup = vec![0.1; 480 * 3];
    denoiser.process(&mut warmup, true).unwrap();

    // With strength=0.0, output should be very close to input (dry signal)
    let input = white_noise(480, 0.2);
    let mut buf = input.clone();
    denoiser.process(&mut buf, true).unwrap();

    // Check that output closely matches input
    let diff_rms = rms(
        &buf.iter()
            .zip(input.iter())
            .map(|(&a, &b)| a - b)
            .collect::<Vec<_>>(),
    );
    let input_rms = rms(&input);

    assert!(
        diff_rms < input_rms * 0.15,
        "Bypass mode should pass through: diff_rms={:.4}, input_rms={:.4}",
        diff_rms,
        input_rms
    );
}

#[test]
fn test_nnnoiseless_strength_full() {
    let mut denoiser = NnnoiseDenoiser::new();
    denoiser.set_strength(1.0);

    // Process noise at full strength
    let mut buf = white_noise(480 * 10, 0.3);
    denoiser.process(&mut buf, true).unwrap();

    // Just verify it runs without error and produces output
    assert!(buf.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_nnnoiseless_vad_updates() {
    let mut denoiser = NnnoiseDenoiser::new();

    // Process enough data to get at least one VAD reading
    let mut buf = sine_wave(480 * 3, 440.0, 48000.0, 0.5);
    denoiser.process(&mut buf, true).unwrap();

    // VAD should have been updated
    assert!(denoiser.frames_processed() >= 2);
    // VAD probability should be a valid float
    assert!(denoiser.vad_probability().is_finite());
}

// ── DspPipeline integration tests ───────────────────────────────────────

#[test]
fn test_pipeline_uses_nnnoiseless_at_48khz() {
    let format = FrameFormat::balanced(); // 48kHz, 128 frames
    let pipeline = DspPipeline::new(format, ProcessingProfile::Balanced);

    assert_eq!(
        pipeline.denoiser_backend(),
        DenoiserBackend::Nnnoiseless,
        "48kHz pipeline should use nnnoiseless backend"
    );
}

#[test]
fn test_pipeline_uses_spectral_at_96khz() {
    let format = FrameFormat::new(1, 96000, 256).unwrap();
    let pipeline = DspPipeline::new(format, ProcessingProfile::Studio);

    assert_eq!(
        pipeline.denoiser_backend(),
        DenoiserBackend::Spectral,
        "96kHz pipeline should fall back to spectral backend"
    );
}

#[test]
fn test_pipeline_nnnoiseless_processes_audio() {
    let format = FrameFormat::balanced();
    let mut pipeline = DspPipeline::new(format, ProcessingProfile::Balanced);

    let mut buffer = white_noise(format.samples_per_buffer(), 0.2);
    let result = pipeline.process(&mut buffer);
    assert!(result.is_ok());

    // Output should be finite
    assert!(buffer.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_pipeline_denoiser_backend_param() {
    let format = FrameFormat::balanced();
    let mut pipeline = DspPipeline::new(format, ProcessingProfile::Balanced);

    // Should start as nnnoiseless
    let backend = pipeline.get_param("denoiser_backend").unwrap();
    assert_eq!(backend, ParamValue::String("nnnoiseless".to_string()));

    // Switch to spectral
    pipeline
        .set_param("denoiser_backend", ParamValue::String("spectral".to_string()))
        .unwrap();
    let backend = pipeline.get_param("denoiser_backend").unwrap();
    assert_eq!(backend, ParamValue::String("spectral".to_string()));

    // Switch back
    pipeline
        .set_param("denoiser_backend", ParamValue::String("nnnoiseless".to_string()))
        .unwrap();
    let backend = pipeline.get_param("denoiser_backend").unwrap();
    assert_eq!(backend, ParamValue::String("nnnoiseless".to_string()));
}

#[test]
fn test_pipeline_latency_includes_nnnoiseless() {
    let format = FrameFormat::balanced();
    let pipeline = DspPipeline::new(format, ProcessingProfile::Balanced);

    let latency = pipeline.latency_frames();
    // Should include nnnoiseless 480 samples + limiter lookahead
    assert!(
        latency >= 480,
        "Latency should include nnnoiseless frame size, got {}",
        latency
    );
}

#[test]
fn test_pipeline_noise_reduction_strength_propagates() {
    let format = FrameFormat::balanced();
    let mut pipeline = DspPipeline::new(format, ProcessingProfile::Balanced);

    // Set strength via param
    pipeline
        .set_param("noise_reduction_strength", ParamValue::Float(0.5))
        .unwrap();

    let strength = pipeline.get_param("noise_reduction_strength").unwrap();
    assert_eq!(strength, ParamValue::Float(0.5));
}
