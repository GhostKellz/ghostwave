# GhostWave Architecture

Complete technical architecture and design documentation for GhostWave.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Design Principles](#design-principles)
- [Core Components](#core-components)
- [Audio Processing Pipeline](#audio-processing-pipeline)
- [Memory Management](#memory-management)

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

## Design Principles

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

---

## Memory Management

### Lock-Free Audio Buffers

```rust
pub struct LockFreeAudioBuffer {
    buffer: Vec<f32>,
    capacity: usize,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
    sample_rate: u32,
}
```

### Memory Pool System

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

## See Also

- [Performance Optimization](PERFORMANCE.md) - Detailed performance tuning guide
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Development Guide](DEVELOPMENT.md) - Contributing and development workflow
