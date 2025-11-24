# GhostWave Technical Documentation

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Audio Processing Pipeline](#audio-processing-pipeline)
- [Performance Optimization](#performance-optimization)
- [Configuration System](#configuration-system)
- [API Reference](#api-reference)
- [Development Guide](#development-guide)

---

## Architecture Overview

GhostWave is built with a modular, performance-first architecture designed for real-time audio processing:

```
┌─────────────────────────────────────────────────────────────┐
│                    GhostWave CLI                           │
├─────────────────────────────────────────────────────────────┤
│  Audio Backends  │   Integration   │    Management         │
│  ├─ PipeWire     │   ├─ PhantomLink│    ├─ SystemD         │
│  ├─ ALSA         │   ├─ IPC/RPC   │    ├─ Service         │
│  ├─ JACK         │   └─ JSON API   │    └─ Auto-Start     │
│  └─ CPAL         │                 │                       │
├─────────────────────────────────────────────────────────────┤
│                   GhostWave Core                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │    Audio    │ │ Real-Time   │ │   Device            │   │
│  │ Processing  │ │ Scheduling  │ │   Detection         │   │
│  │             │ │             │ │                     │   │
│  │• Noise      │ │• Lock-free  │ │• Hardware           │   │
│  │  Suppression│ │  Buffers    │ │  Auto-config        │   │
│  │• RTX Accel  │ │• Memory     │ │• XLR Interface      │   │
│  │• CPU Fallback│ │  Pools      │ │  Detection         │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
    ┌──────▼──────┐      ┌──────▼──────┐      ┌──────▼──────┐
    │   PipeWire  │      │    CUDA     │      │   Hardware  │
    │   Audio     │      │   Runtime   │      │  Interfaces │
    │   Server    │      │             │      │             │
    └─────────────┘      └─────────────┘      └─────────────┘
```

### Design Principles

1. **Zero-Copy Audio Processing** - Direct buffer manipulation without unnecessary allocations
2. **Lock-Free Data Structures** - Atomic operations for real-time thread safety
3. **Modular Backend System** - Pluggable audio system support
4. **Feature-Gated Compilation** - Optional components reduce binary size
5. **Hardware-Specific Optimization** - Auto-detection and configuration

---

## Core Components

### 1. Noise Suppression Engine (`noise_suppression.rs`)

The heart of GhostWave's audio processing:

```rust
pub struct NoiseProcessor {
    config: NoiseSuppressionConfig,
    gate: NoiseGate,
    spectral_filter: SpectralFilter,
    #[cfg(feature = "nvidia-rtx")]
    rtx_accelerator: Option<RtxAccelerator>,
}
```

**Features:**
- **RTX Acceleration**: GPU-powered noise reduction using CUDA
- **CPU Fallback**: High-quality spectral filtering when GPU unavailable
- **Adaptive Noise Gate**: Dynamic threshold adjustment
- **Real-time Processing**: <1ms processing time per frame

**Processing Pipeline:**
1. Input audio frame received
2. RTX accelerator processes if available, else CPU spectral filter
3. Noise gate applies post-processing
4. Output frame delivered

### 2. Real-Time Scheduler (`low_latency.rs`)

Ensures consistent, low-latency audio processing:

```rust
pub struct RealTimeScheduler {
    target_latency: Duration,
    buffer_size: usize,
    sample_rate: u32,
    frame_duration: Duration,
}
```

**Optimizations:**
- **FIFO Thread Scheduling**: Real-time priority for audio threads
- **CPU Affinity**: Pin threads to specific cores
- **Frame-Accurate Timing**: Precise sleep/wake cycles
- **XRun Detection**: Monitor and report audio dropouts

### 3. Device Detection (`device_detection.rs`)

Automatically configures optimal settings for audio hardware:

```rust
pub struct DeviceDetector {
    known_devices: HashMap<String, AudioDevice>,
}
```

**Supported Hardware:**
- **Focusrite Scarlett Solo 4th Gen**: Optimized XLR configuration
- **Generic USB Audio**: Universal compatibility
- **Future**: Extensible for additional interfaces

**Auto-Configuration:**
- Sample rate optimization (44.1kHz → 192kHz)
- Buffer size recommendations
- Channel mapping
- Profile suggestions

---

## Audio Processing Pipeline

### Buffer Flow Architecture

```
Input → [Lock-Free Ring Buffer] → [Noise Processor] → [Lock-Free Ring Buffer] → Output
         ↑                                                                      ↓
    [Memory Pool]                                                         [Audio Backend]
         ↓                                                                      ↑
    [Recycled Buffers] ←──────────────────────────────────────────────────────┘
```

### Processing Stages

1. **Input Capture**
   - Audio backend captures from hardware
   - Frames written to lock-free input buffer
   - Zero-copy transfer to processing thread

2. **Noise Suppression**
   - RTX accelerator processes frame (if available)
   - CPU spectral filter as fallback
   - Adaptive noise gate post-processing

3. **Output Delivery**
   - Processed frames written to output buffer
   - Audio backend consumes for playback
   - Frame timing maintained for real-time operation

### Memory Management

**Lock-Free Audio Buffers:**
```rust
pub struct LockFreeAudioBuffer {
    buffer: Vec<f32>,
    capacity: usize,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
    sample_rate: u32,
}
```

**Memory Pool System:**
```rust
pub struct AudioMemoryPool {
    buffers: crossbeam::channel::Receiver<Vec<f32>>,
    buffer_sender: crossbeam::channel::Sender<Vec<f32>>,
    buffer_size: usize,
}
```

**Benefits:**
- No allocations in audio processing path
- Predictable memory usage
- Cache-friendly buffer reuse
- Zero-copy operations where possible

---

## Performance Optimization

### Real-Time Threading

**Thread Priority Setup:**
```rust
const SCHED_FIFO: c_int = 1;
const RT_PRIORITY: c_int = 80;

unsafe {
    sched_setscheduler(0, SCHED_FIFO, &RT_PRIORITY as *const c_int);
}
```

**CPU Affinity (Optional):**
- Pin audio threads to performance cores
- Isolate from system interrupts
- Reduce cache misses and context switches

### Latency Targets

| Profile | Buffer Size | Target Latency | Use Case |
|---------|-------------|----------------|----------|
| Studio | 64-128 frames | <3ms | Professional recording |
| Balanced | 256-512 frames | 5-10ms | Gaming, streaming |
| Streaming | 1024+ frames | 10-20ms | Content creation |

### Benchmarking System

Built-in performance analysis:
```rust
pub struct AudioBenchmark {
    processing_times: Vec<Duration>,
    frame_count: AtomicU64,
    xrun_count: AtomicU64,
    max_processing_time: AtomicU64,
}
```

**Metrics Tracked:**
- Frame processing time distribution
- XRun (audio dropout) detection
- CPU usage patterns
- Memory allocation tracking

---

## Configuration System

### Profile-Based Configuration

**Built-in Profiles:**
- `balanced`: General-purpose settings
- `streaming`: Optimized for content creation
- `studio`: Professional audio production

**Configuration Structure:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub profile: ProfileConfig,
    pub audio: AudioConfig,
    pub noise_suppression: NoiseSuppressionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub buffer_size: u32,
    pub channels: u8,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
}
```

### Dynamic Configuration

**Runtime Parameter Adjustment:**
- Sample rate switching
- Buffer size optimization
- Noise suppression strength
- Audio routing changes

**Configuration Sources:**
1. Built-in profiles
2. JSON configuration files
3. Environment variables
4. Command-line overrides
5. IPC API calls

---

## API Reference

### Core Audio Processing

```rust
// Initialize noise processor
let config = Config::load("studio")?;
let mut processor = NoiseProcessor::new(&config.noise_suppression)?;

// Process audio frame
let input = vec![0.1f32; 1024];
let mut output = vec![0.0f32; 1024];
processor.process(&input, &mut output)?;

// Check processing mode
let mode = processor.get_processing_mode(); // "RTX GPU + CPU Gate" or "CPU Only"
```

### Device Detection

```rust
// Auto-detect audio devices
let detector = DeviceDetector::new();
let devices = detector.detect_devices().await?;

// Find specific device
let scarlett = detector.find_scarlett_solo_4th_gen().await?;
if let Some(device) = scarlett {
    let config = detector.get_optimal_config_for_device(&device).await?;
}
```

### Real-Time Optimization

```rust
// Set up real-time scheduling
RealTimeScheduler::optimize_thread_for_audio()?;

// Create scheduler for specific configuration
let scheduler = RealTimeScheduler::new(48000, 512);
let optimal_buffer = RealTimeScheduler::get_optimal_buffer_size(48000, 15);
```

### Backend Selection

```rust
// Enumerate available backends
let backends = AudioBackend::available_backends();

// Check backend availability
if AudioBackend::PipeWire.is_available() {
    // Use PipeWire
}

// Get recommended backend
let recommended = AudioBackend::recommended();
```

---

## Development Guide

### Building with Features

```bash
# All features enabled
cargo build --features "pipewire-backend,alsa-backend,jack-backend,nvidia-rtx"

# Minimal build (CPAL only)
cargo build --no-default-features --features "cpal-backend"

# Release with optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Testing

```bash
# Unit tests
cargo test

# Integration tests with hardware
cargo test --test integration -- --ignored

# Benchmark tests
cargo test --release --test benchmarks
```

### Debugging

**Enable verbose logging:**
```bash
RUST_LOG=ghostwave=debug cargo run -- --verbose
```

**Audio-specific debugging:**
```bash
RUST_LOG=ghostwave::low_latency=trace cargo run -- --bench
```

### Profiling

**CPU profiling with perf:**
```bash
perf record --call-graph=dwarf ./ghostwave --bench
perf report
```

**Memory profiling:**
```bash
valgrind --tool=massif ./ghostwave --bench
```

### Adding New Audio Backends

1. **Create backend module** in `ghostwave-core/src/`
2. **Implement availability check** function
3. **Add feature flag** to `Cargo.toml`
4. **Update AudioBackend enum** in `lib.rs`
5. **Add integration tests**

Example structure:
```rust
// my_backend.rs
#[cfg(feature = "my-backend")]
pub fn check_my_backend_availability() -> bool {
    // Implementation
}

#[cfg(not(feature = "my-backend"))]
pub fn check_my_backend_availability() -> bool {
    false
}
```

---

## Troubleshooting

### Common Issues

**Audio dropouts/XRuns:**
- Increase buffer size
- Enable real-time scheduling
- Check CPU governor settings
- Reduce system load

**High latency:**
- Decrease buffer size
- Use ALSA direct mode
- Optimize thread affinity
- Disable CPU frequency scaling

**RTX acceleration not working:**
- Verify CUDA installation
- Check GPU compute capability
- Ensure proper driver version
- Monitor GPU memory usage

### Performance Tuning

**System-level optimizations:**
```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase audio thread priority
sudo setcap cap_sys_nice+ep ./ghostwave

# Disable audio power management
echo 0 | sudo tee /sys/module/snd_hda_intel/parameters/power_save
```

**Audio system configuration:**
```bash
# PipeWire optimization
mkdir -p ~/.config/pipewire
cp /usr/share/pipewire/pipewire.conf ~/.config/pipewire/
# Edit quantum settings for lower latency
```

---

## Contributing

### Code Style

- **Rust 2024 Edition** features preferred
- **`rustfmt`** for formatting
- **`clippy`** for linting
- **Documentation** for public APIs

### Performance Requirements

- **Audio thread latency** < 15ms (99th percentile)
- **Memory allocations** in audio path: Zero
- **CPU usage** < 20% on target hardware
- **XRun rate** < 0.1% under normal operation

### Testing Standards

- **Unit tests** for all core functions
- **Integration tests** with mock hardware
- **Performance benchmarks** for regressions
- **Documentation tests** for examples

---

This documentation provides a comprehensive technical overview of GhostWave's architecture and implementation. For specific integration examples, see [INTEGRATION.md](INTEGRATION.md).