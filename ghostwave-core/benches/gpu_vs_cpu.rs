//! GPU vs CPU performance benchmarks for GhostWave AI denoising
//!
//! Run with: cargo bench --bench gpu_vs_cpu --features "nvidia-rtx"
//!
//! This benchmark compares:
//! - CPU inference performance
//! - GPU inference performance (if CUDA available)
//! - Model weight operations
//! - Different model sizes

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::hint::black_box;
use ghostwave_core::ai_denoise::{RNNoiseModelWeights, ModelSize};

/// Benchmark model weight creation for different sizes
fn bench_model_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_creation");

    for size in [ModelSize::Tiny, ModelSize::Standard, ModelSize::Large] {
        let name = match size {
            ModelSize::Tiny => "tiny_64h",
            ModelSize::Standard => "standard_96h",
            ModelSize::Large => "large_128h",
        };

        group.bench_with_input(
            BenchmarkId::new("new", name),
            &size,
            |b, &size| {
                b.iter(|| {
                    RNNoiseModelWeights::new(black_box(size))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pretrained", name),
            &size,
            |b, &size| {
                b.iter(|| {
                    RNNoiseModelWeights::pretrained(black_box(size))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark model save/load operations
fn bench_model_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_io");

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");

    for size in [ModelSize::Tiny, ModelSize::Standard, ModelSize::Large] {
        let name = match size {
            ModelSize::Tiny => "tiny_64h",
            ModelSize::Standard => "standard_96h",
            ModelSize::Large => "large_128h",
        };

        let weights = RNNoiseModelWeights::pretrained(size);
        let path = temp_dir.path().join(format!("{}.gwm", name));

        // Save once for load benchmark
        weights.save(&path).unwrap();

        let param_count = weights.param_count();
        group.throughput(Throughput::Bytes((param_count * 4) as u64));

        group.bench_with_input(
            BenchmarkId::new("save", name),
            &(weights.clone(), path.clone()),
            |b, (w, p)| {
                b.iter(|| {
                    w.save(black_box(p)).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("load", name),
            &path,
            |b, p| {
                b.iter(|| {
                    RNNoiseModelWeights::load(black_box(p)).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark CPU inference simulation (GRU forward pass components)
fn bench_cpu_inference_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_inference");

    for size in [ModelSize::Tiny, ModelSize::Standard, ModelSize::Large] {
        let name = match size {
            ModelSize::Tiny => "tiny_64h",
            ModelSize::Standard => "standard_96h",
            ModelSize::Large => "large_128h",
        };

        let weights = RNNoiseModelWeights::pretrained(size);
        let hidden = size.hidden_size();

        // Simulate input features (42 bark bands)
        let input_features: Vec<f32> = (0..42).map(|i| (i as f32 * 0.01).sin()).collect();
        // Throughput in audio frames (10ms at 48kHz = 480 samples)
        group.throughput(Throughput::Elements(480));

        group.bench_with_input(
            BenchmarkId::new("gru_forward", name),
            &(&weights, &input_features),
            |b, &(weights, input)| {
                let mut hidden_state = vec![0.0f32; hidden];
                let mut output = vec![0.0f32; 23];
                b.iter(|| {
                    // Simulate GRU forward pass (simplified - real impl uses SIMD)
                    cpu_gru_forward_simulation(
                        black_box(weights),
                        black_box(input),
                        black_box(&mut hidden_state),
                        black_box(&mut output),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Simulate CPU GRU forward pass (simplified for benchmarking)
fn cpu_gru_forward_simulation(
    weights: &RNNoiseModelWeights,
    input: &[f32],
    hidden: &mut [f32],
    output: &mut [f32],
) {
    let hidden_size = hidden.len();

    // Simulate GRU1: input -> hidden
    let mut temp_hidden = vec![0.0f32; hidden_size];
    for (i, h) in temp_hidden.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (j, &x) in input.iter().enumerate() {
            if let Some(&w) = weights.gru1.w_ih.get(j * hidden_size + i) {
                sum += x * w;
            }
        }
        *h = (sum).tanh(); // Simplified activation
    }

    // Simulate GRU2: hidden -> hidden
    for (i, h) in hidden.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (j, &x) in temp_hidden.iter().enumerate() {
            if let Some(&w) = weights.gru2.w_ih.get(j * hidden_size + i) {
                sum += x * w;
            }
        }
        *h = (sum).tanh();
    }

    // Simulate output: hidden -> 23
    for (i, o) in output.iter_mut().enumerate() {
        let mut sum = weights.output_bias[i];
        for (j, &h) in hidden.iter().enumerate() {
            sum += h * weights.output_weights[j * 23 + i];
        }
        *o = 1.0 / (1.0 + (-sum).exp()); // Sigmoid
    }
}

/// Benchmark matrix operations (core of GRU)
fn bench_matrix_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_ops");

    for size in [64, 96, 128] {
        let name = format!("{}x{}", size, size);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("matmul_naive", &name),
            &size,
            |bench, &n| {
                let a: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.001).sin()).collect();
                let b: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.002).cos()).collect();
                let mut c_out = vec![0.0f32; n * n];
                bench.iter(|| {
                    naive_matmul(black_box(n), black_box(&a), black_box(&b), black_box(&mut c_out));
                });
            },
        );
    }

    group.finish();
}

/// Naive matrix multiplication (baseline)
fn naive_matmul(n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Benchmark SIMD operations (if available)
fn bench_simd_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_ops");

    let sizes = [256, 512, 1024, 4096];

    for &size in &sizes {
        let a: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).cos()).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("dot_product", size),
            &(&a, &b),
            |bench, &(a, b)| {
                bench.iter(|| {
                    dot_product_scalar(black_box(a), black_box(b))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("vector_add", size),
            &(&a, &b),
            |bench, &(a, b)| {
                let mut c = vec![0.0f32; size];
                bench.iter(|| {
                    vector_add_scalar(black_box(a), black_box(b), black_box(&mut c));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sigmoid", size),
            &(&a,),
            |bench, &(a,)| {
                let mut c = vec![0.0f32; size];
                bench.iter(|| {
                    sigmoid_scalar(black_box(a), black_box(&mut c));
                });
            },
        );
    }

    group.finish();
}

fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vector_add_scalar(a: &[f32], b: &[f32], c: &mut [f32]) {
    for ((c, a), b) in c.iter_mut().zip(a.iter()).zip(b.iter()) {
        *c = a + b;
    }
}

fn sigmoid_scalar(input: &[f32], output: &mut [f32]) {
    for (o, &i) in output.iter_mut().zip(input.iter()) {
        *o = 1.0 / (1.0 + (-i).exp());
    }
}

/// Benchmark full audio frame processing simulation
fn bench_audio_frame_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_frame");

    // Common audio frame sizes at 48kHz
    let frame_configs = [
        (32, "0.67ms"),   // Ultra-low latency
        (64, "1.33ms"),   // Low latency
        (128, "2.67ms"),  // Gaming
        (256, "5.33ms"),  // Streaming
        (480, "10ms"),    // Standard (RNNoise frame)
        (1024, "21.3ms"), // Recording
    ];

    let weights = RNNoiseModelWeights::pretrained(ModelSize::Standard);
    let hidden = ModelSize::Standard.hidden_size();

    for (frame_size, label) in frame_configs {
        group.throughput(Throughput::Elements(frame_size as u64));

        let audio_input: Vec<f32> = (0..frame_size)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 48000.0).sin() * 0.5)
            .collect();
        // Simulate feature extraction + inference + output
        let input_features: Vec<f32> = (0..42).map(|i| (i as f32 * 0.01).sin()).collect();

        group.bench_with_input(
            BenchmarkId::new("full_pipeline", label),
            &(&weights, &audio_input, &input_features),
            |bench, &(weights, audio_in, features)| {
                let mut audio_out = vec![0.0f32; frame_size];
                let mut hidden_state = vec![0.0f32; hidden];
                let mut gains = vec![0.0f32; 23];
                bench.iter(|| {
                    // 1. Feature extraction (simulated)
                    // 2. Neural network inference
                    cpu_gru_forward_simulation(
                        black_box(weights),
                        black_box(features),
                        black_box(&mut hidden_state),
                        black_box(&mut gains),
                    );
                    // 3. Apply gains to audio (simulated)
                    for (out, &inp) in audio_out.iter_mut().zip(audio_in.iter()) {
                        *out = inp * black_box(gains[0]); // Simplified
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_model_creation,
    bench_model_io,
    bench_cpu_inference_simulation,
    bench_matrix_ops,
    bench_simd_ops,
    bench_audio_frame_processing,
);
criterion_main!(benches);
