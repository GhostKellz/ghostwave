//! Latency benchmarks for GhostWave audio processing
//!
//! Run with: cargo bench --bench latency

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ghostwave_core::{Config, GhostWaveProcessor};

fn process_frames_benchmark(c: &mut Criterion) {
    let config = Config::default();
    let processor = GhostWaveProcessor::new(config).unwrap();

    let mut group = c.benchmark_group("frame_processing");

    for buffer_size in [64, 128, 256, 512, 1024].iter() {
        group.bench_with_input(
            BenchmarkId::new("process", buffer_size),
            buffer_size,
            |b, &size| {
                let input = vec![0.1f32; size];
                let mut output = vec![0.0f32; size];
                b.iter(|| {
                    processor.process(black_box(&input), black_box(&mut output)).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn profile_comparison_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiles");

    for profile in ["balanced", "streaming", "studio"].iter() {
        let config = Config::load(profile).unwrap();
        let processor = GhostWaveProcessor::new(config).unwrap();

        let input = vec![0.1f32; 256];
        let mut output = vec![0.0f32; 256];

        group.bench_with_input(
            BenchmarkId::new("process_256", profile),
            profile,
            |b, _| {
                b.iter(|| {
                    processor.process(black_box(&input), black_box(&mut output)).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, process_frames_benchmark, profile_comparison_benchmark);
criterion_main!(benches);
