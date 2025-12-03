//! GPU acceleration example
//!
//! Run with: cargo run --example 02_gpu_acceleration --features nvidia-rtx

use ghostwave_core::{Config, GhostWaveProcessor};

fn main() -> anyhow::Result<()> {
    let config = Config::load("studio")?;
    let processor = GhostWaveProcessor::new(config)?;

    println!("GhostWave GPU Acceleration Status");
    println!("==================================");
    println!("RTX Available: {}", processor.has_rtx_acceleration());
    println!("Processing Mode: {}", processor.get_processing_mode());

    #[cfg(feature = "nvidia-rtx")]
    if let Some(caps) = processor.get_rtx_capabilities() {
        println!("\nRTX Capabilities:");
        println!("  GPU Generation: {:?}", caps.gpu_generation);
        println!("  Compute: {}.{}", caps.compute_capability.0, caps.compute_capability.1);
        println!("  Memory: {:.1} GB", caps.memory_gb);
        println!("  Tensor Cores: Gen {}", caps.tensor_core_gen);
        println!("  FP4 Support: {}", caps.supports_fp4);
        println!("  RTX Voice Compatible: {}", caps.supports_rtx_voice);
    }

    // Benchmark GPU vs CPU
    let input = vec![0.1f32; 1024];
    let mut output = vec![0.0f32; 1024];

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        processor.process(&input, &mut output)?;
    }
    let elapsed = start.elapsed();

    println!("\nPerformance (1000 iterations, 1024 samples):");
    println!("  Total: {:?}", elapsed);
    println!("  Per frame: {:?}", elapsed / 1000);
    println!("  Throughput: {:.2} frames/sec", 1000.0 / elapsed.as_secs_f64());

    Ok(())
}
