//! Real-time scheduling example
//!
//! Run with: cargo run --example 04_realtime_scheduling

use ghostwave_core::{
    Config, GhostWaveProcessor,
    RealTimeScheduler, AudioBenchmark, LockFreeAudioBuffer, TARGET_LATENCY_MS,
};
use std::time::{Duration, Instant};

fn main() -> anyhow::Result<()> {
    println!("GhostWave Real-Time Scheduling Example");
    println!("======================================");

    let config = Config::load("studio")?;
    let sample_rate = config.audio.sample_rate;
    let buffer_size = config.audio.buffer_size as usize;

    let processor = GhostWaveProcessor::new(config)?;

    // Optimize thread for real-time audio
    match RealTimeScheduler::optimize_thread_for_audio() {
        Ok(()) => println!("RT scheduling enabled (SCHED_FIFO)"),
        Err(e) => println!("RT scheduling unavailable: {} (using normal priority)", e),
    }

    // Create scheduler and benchmark
    let scheduler = RealTimeScheduler::new(sample_rate, buffer_size);
    let benchmark = AudioBenchmark::new(sample_rate, buffer_size);

    // Create lock-free buffer
    let ring_buffer = LockFreeAudioBuffer::new(buffer_size * 4, sample_rate);

    println!("\nConfiguration:");
    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Buffer size: {} frames", buffer_size);
    println!("  Target latency: {} ms", TARGET_LATENCY_MS);

    // Calculate optimal buffer
    let optimal = RealTimeScheduler::get_optimal_buffer_size(sample_rate, TARGET_LATENCY_MS);
    println!("  Optimal buffer for {}ms: {} frames", TARGET_LATENCY_MS, optimal);

    // Simulate real-time processing
    println!("\nSimulating 2 seconds of real-time processing...");

    let input = vec![0.1f32; buffer_size];
    let mut output = vec![0.0f32; buffer_size];

    let start = Instant::now();
    let duration = Duration::from_secs(2);
    let mut frame_count = 0u64;

    while start.elapsed() < duration {
        let frame_start = Instant::now();

        // Process audio
        processor.process(&input, &mut output)?;

        // Write to ring buffer
        ring_buffer.write(&output)?;

        // Record performance
        benchmark.record_frame_processing(frame_start.elapsed());
        frame_count += 1;

        // Maintain frame timing
        scheduler.sleep_until_next_frame(frame_start);
    }

    // Report results
    let stats = benchmark.get_stats();
    println!("\nResults:");
    println!("  Frames processed: {}", frame_count);
    println!("  Total frames: {}", stats.total_frames);
    println!("  Max latency: {} us", stats.max_processing_time.as_micros());
    println!("  Target frame time: {} us", stats.target_frame_time.as_micros());
    println!("  XRuns: {} ({:.3}%)", stats.xrun_count,
             (stats.xrun_count as f64 / frame_count as f64) * 100.0);

    if stats.xrun_count == 0 {
        println!("\nPerfect real-time performance achieved!");
    }

    Ok(())
}
