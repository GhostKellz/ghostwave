# GhostWave Development Guide

Complete guide for contributing to and developing GhostWave.

## Table of Contents

- [Building](#building)
- [Testing](#testing)
- [Debugging](#debugging)
- [Profiling](#profiling)
- [Adding Features](#adding-features)
- [Code Style](#code-style)
- [Performance Requirements](#performance-requirements)

---

## Building

### Development Build

```bash
# Standard debug build
cargo build

# With all features
cargo build --features "pipewire-backend,alsa-backend,jack-backend,nvidia-rtx"

# Specific backend only
cargo build --features "pipewire-backend,nvidia-rtx"
```

### Release Build

```bash
# Standard release build
cargo build --release

# Optimized for native CPU
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Full optimization with all features
RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
  cargo build --release --features full
```

### Minimal Build

For embedded or constrained environments:

```bash
# CPAL-only, no RTX acceleration
cargo build --no-default-features --features "cpal-backend"

# Specific size optimizations
RUSTFLAGS="-C opt-level=z -C lto=fat -C codegen-units=1" \
  cargo build --release --no-default-features
```

### Feature Flags

Available features in `Cargo.toml`:

```toml
[features]
default = ["cpal-backend"]

# Audio backends
pipewire-backend = ["pipewire"]
alsa-backend = ["alsa"]
jack-backend = ["jack"]
cpal-backend = ["cpal"]

# GPU acceleration
nvidia-rtx = ["cudarc"]
cuda-tensorrt = ["cudarc"]
vulkan-compute = []
opencl = []

# All features
full = [
    "pipewire-backend",
    "alsa-backend",
    "jack-backend",
    "cpal-backend",
    "nvidia-rtx",
    "cuda-tensorrt",
]
```

---

## Testing

### Unit Tests

```bash
# Run all unit tests
cargo test

# Run specific module tests
cargo test noise_suppression

# Run with logging
RUST_LOG=debug cargo test -- --nocapture
```

### Integration Tests

```bash
# Integration tests (require audio hardware)
cargo test --test integration -- --ignored

# Test specific backend
cargo test --test integration_pipewire -- --ignored

# Test RTX acceleration
cargo test --features nvidia-rtx rtx_acceleration -- --ignored
```

### Benchmark Tests

```bash
# Run performance benchmarks
cargo test --release --test benchmarks

# Extended benchmark (10 minutes)
cargo test --release --test benchmarks -- --ignored
```

### Test Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html --output-dir ./coverage
```

---

## Debugging

### Verbose Logging

```bash
# Enable all debug logs
RUST_LOG=ghostwave=debug cargo run -- --verbose

# Module-specific logging
RUST_LOG=ghostwave::low_latency=trace cargo run

# Audio-specific debugging
RUST_LOG=ghostwave::noise_suppression=debug,ghostwave::rtx_acceleration=trace \
  cargo run -- --bench
```

### Log Levels

- `error`: Critical errors only
- `warn`: Warnings and errors
- `info`: General information (default)
- `debug`: Detailed debugging
- `trace`: Very verbose (audio path debugging)

### GDB Debugging

```bash
# Build with debug symbols
cargo build

# Run with GDB
rust-gdb ./target/debug/ghostwave

# Set breakpoints
(gdb) break noise_suppression.rs:123
(gdb) run --bench
```

### LLDB Debugging (macOS/BSD)

```bash
rust-lldb ./target/debug/ghostwave
```

---

## Profiling

### CPU Profiling with `perf`

```bash
# Record performance data
perf record --call-graph=dwarf ./target/release/ghostwave --bench

# Analyze results
perf report

# Generate flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

### Memory Profiling

**Valgrind Massif:**
```bash
valgrind --tool=massif ./target/release/ghostwave --bench
ms_print massif.out.XXXXX
```

**Heaptrack:**
```bash
heaptrack ./target/release/ghostwave --bench
heaptrack_gui heaptrack.ghostwave.XXXXX.gz
```

### GPU Profiling

**NVIDIA Nsight:**
```bash
# Profile GPU operations
nsys profile --trace=cuda,nvtx ./target/release/ghostwave --bench

# View results
nsys-ui report.qdrep
```

**nvidia-smi Monitoring:**
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Log GPU stats
nvidia-smi dmon -s pucvmet
```

---

## Adding Features

### Adding a New Audio Backend

1. **Create backend module** in `ghostwave-core/src/backends/`

```rust
// my_backend.rs
#[cfg(feature = "my-backend")]
pub fn check_my_backend_availability() -> bool {
    // Check if backend is available
    true
}

#[cfg(not(feature = "my-backend"))]
pub fn check_my_backend_availability() -> bool {
    false
}

#[cfg(feature = "my-backend")]
pub struct MyBackend {
    // Backend state
}

#[cfg(feature = "my-backend")]
impl MyBackend {
    pub fn new() -> Result<Self> {
        // Initialize backend
    }

    pub fn start_audio(&mut self) -> Result<()> {
        // Start audio processing
    }
}
```

2. **Add feature flag** to `Cargo.toml`

```toml
[dependencies]
my-backend-sys = { version = "1.0", optional = true }

[features]
my-backend = ["my-backend-sys"]
```

3. **Update `AudioBackend` enum** in `lib.rs`

```rust
pub enum AudioBackend {
    PipeWire,
    ALSA,
    JACK,
    CPAL,
    #[cfg(feature = "my-backend")]
    MyBackend,
}
```

4. **Add integration tests**

```rust
// tests/integration_my_backend.rs
#[cfg(feature = "my-backend")]
#[ignore] // Requires hardware
#[test]
fn test_my_backend_initialization() {
    use ghostwave_core::MyBackend;
    let backend = MyBackend::new().expect("Failed to initialize backend");
}
```

### Adding GPU Acceleration Support

See `src/rtx_acceleration.rs` and `src/gpu_acceleration.rs` for examples.

Key components:
- GPU detection and capability checking
- Memory transfer (host ↔ device)
- Kernel/shader implementation
- CPU fallback path
- Performance benchmarking

---

## Code Style

### Rust Style Guidelines

**Use `rustfmt`:**
```bash
cargo fmt

# Check formatting without changes
cargo fmt -- --check
```

**Use `clippy`:**
```bash
cargo clippy

# Fail on warnings
cargo clippy -- -D warnings
```

### Code Conventions

1. **Naming:**
   - `snake_case` for functions and variables
   - `PascalCase` for types and enums
   - `SCREAMING_SNAKE_CASE` for constants

2. **Documentation:**
   - Document all public APIs
   - Include examples in doc comments
   - Run `cargo doc --open` to verify

3. **Error Handling:**
   - Use `Result<T, NvControlError>` for recoverable errors
   - Use `panic!` only for unrecoverable errors
   - Provide context with `.context()` or `.with_context()`

4. **Performance:**
   - No allocations in audio processing path
   - Use `#[inline]` for hot functions
   - Profile before optimizing

### Example Function

```rust
/// Processes audio buffer with noise suppression
///
/// # Arguments
///
/// * `input` - Input audio samples (interleaved if multi-channel)
/// * `output` - Output buffer (must be same size as input)
///
/// # Returns
///
/// * `Ok(())` - Processing succeeded
/// * `Err(NvControlError)` - Processing failed
///
/// # Example
///
/// ```
/// use ghostwave_core::NoiseProcessor;
///
/// let mut processor = NoiseProcessor::new(&config)?;
/// let input = vec![0.1f32; 1024];
/// let mut output = vec![0.0f32; 1024];
/// processor.process(&input, &mut output)?;
/// ```
#[inline]
pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
    // Implementation
}
```

---

## Performance Requirements

### Audio Thread Requirements

**Critical metrics:**
- Latency: < 15ms (99th percentile)
- Memory allocations in audio path: **Zero**
- CPU usage: < 20% on target hardware
- XRun rate: < 0.1% under normal operation

### Testing Performance

```bash
# Run performance benchmark
cargo test --release --test benchmarks

# Check for allocations in audio path
RUST_LOG=trace cargo test --release test_zero_allocations

# Measure XRun rate
cargo test --release test_xrun_rate -- --ignored
```

### Optimization Checklist

Before merging performance-critical code:

- [ ] Profiled with `perf` or similar
- [ ] Zero allocations in audio path verified
- [ ] Benchmarks show improvement
- [ ] No performance regressions
- [ ] Documentation updated

---

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - run: cargo test
      - run: cargo clippy -- -D warnings
      - run: cargo fmt -- --check
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
cargo fmt -- --check
cargo clippy -- -D warnings
cargo test
EOF

chmod +x .git/hooks/pre-commit
```

---

## See Also

- [Architecture](ARCHITECTURE.md) - System architecture
- [API Reference](API_REFERENCE.md) - API documentation
- [Performance](PERFORMANCE.md) - Optimization guide
- [Contributing Guidelines](../CONTRIBUTING.md) - Contribution process
