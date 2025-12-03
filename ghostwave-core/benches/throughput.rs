//! Throughput benchmarks for GhostWave
//!
//! Run with: cargo bench --bench throughput

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use ghostwave_core::{Config, GhostWaveProcessor, LockFreeAudioBuffer};

fn throughput_benchmark(c: &mut Criterion) {
    let config = Config::default();
    let processor = GhostWaveProcessor::new(config).unwrap();

    let mut group = c.benchmark_group("throughput");

    // Measure samples per second
    let buffer_size = 1024;
    group.throughput(Throughput::Elements(buffer_size as u64));

    let input = vec![0.1f32; buffer_size];
    let mut output = vec![0.0f32; buffer_size];

    group.bench_function("process_1024_samples", |b| {
        b.iter(|| {
            processor.process(black_box(&input), black_box(&mut output)).unwrap();
        });
    });

    group.finish();
}

fn ring_buffer_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer");

    let buffer = LockFreeAudioBuffer::new(8192, 48000);
    let test_data = vec![0.5f32; 1024];
    let mut read_buffer = vec![0.0f32; 1024];

    group.throughput(Throughput::Elements(1024));

    group.bench_function("write_1024", |b| {
        b.iter(|| {
            buffer.write(black_box(&test_data)).unwrap();
        });
    });

    // Pre-fill for read test
    buffer.write(&test_data).unwrap();

    group.bench_function("read_1024", |b| {
        b.iter(|| {
            buffer.read(black_box(&mut read_buffer)).unwrap();
            buffer.write(&test_data).unwrap(); // Refill for next iteration
        });
    });

    group.finish();
}

criterion_group!(benches, throughput_benchmark, ring_buffer_benchmark);
criterion_main!(benches);
