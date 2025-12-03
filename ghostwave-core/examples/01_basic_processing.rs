//! Basic GhostWave audio processing example
//!
//! Run with: cargo run --example 01_basic_processing

use ghostwave_core::{Config, GhostWaveProcessor};

fn main() -> anyhow::Result<()> {
    // Initialize with default balanced profile
    let config = Config::load("balanced")?;
    let processor = GhostWaveProcessor::new(config)?;

    println!("GhostWave initialized");
    println!("  Processing mode: {}", processor.get_processing_mode());
    println!("  RTX available: {}", processor.has_rtx_acceleration());

    // Process some audio
    let input = vec![0.1f32; 256]; // Simulated input
    let mut output = vec![0.0f32; 256];

    processor.process(&input, &mut output)?;

    println!("Processed 256 samples");
    println!("  Input RMS: {:.4}", rms(&input));
    println!("  Output RMS: {:.4}", rms(&output));

    Ok(())
}

fn rms(samples: &[f32]) -> f32 {
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}
