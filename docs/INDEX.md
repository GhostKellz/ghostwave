# GhostWave Documentation Hub

Complete documentation for GhostWave - NVIDIA RTX Voice for Linux.

## Quick Start

**New Users:**
1. [README](../README.md) - Project overview and installation
2. [NVIDIA Setup](NVIDIA.md) - RTX GPU acceleration guide
3. [PipeWire Integration](PIPEWIRE.md) - Modern Linux audio setup

**RTX 5090 Users:**
- [RTX 5090 Optimizations](RTX_5090_OPTIMIZATIONS.md) - Blackwell-specific features and ASUS ROG Astral setup

---

## Core Documentation

### Architecture & Design

- [**Architecture**](ARCHITECTURE.md) - System architecture and core components
  - Design principles
  - Audio processing pipeline
  - Memory management
  - Component overview

- [**Performance**](PERFORMANCE.md) - Optimization and benchmarking
  - Real-time threading
  - Latency targets
  - System-level optimization
  - Hardware requirements

### Development

- [**API Reference**](API_REFERENCE.md) - Complete API documentation
  - Noise suppression API
  - Device detection
  - Real-time optimization
  - Configuration system

- [**Development Guide**](DEVELOPMENT.md) - Contributing and development
  - Building and testing
  - Debugging and profiling
  - Adding features
  - Code style guidelines

---

## Audio Integration

### Audio Backends

- [**PipeWire**](PIPEWIRE.md) - Modern Linux audio (recommended)
  - Setup and configuration
  - Low-latency optimization
  - Node routing

- [**ALSA**](ALSA.md) - Direct hardware access
  - Minimal latency setup
  - Hardware configuration
  - Troubleshooting

### Library Integration

- [**Integration Guide**](INTEGRATION.md) - Embedding ghostwave-core in your projects
  - Quick start examples
  - Real-time audio processing
  - PhantomLink JSON-RPC API
  - Performance optimization patterns

### External Integration

- [**PhantomLink Integration**](phantomlink-integration.md) - Professional audio mixer
  - Setup guide
  - Routing configuration
  - Performance optimization

- [**nvcontrol Integration**](nvcontrol-integration.md) - GPU management
  - Power profiles
  - Clock speed optimization
  - Thermal management

---

## GPU Acceleration

### NVIDIA RTX

- [**NVIDIA RTX Guide**](NVIDIA.md) - RTX GPU acceleration
  - Hardware requirements (RTX 20-50 series)
  - Driver installation + diagnostics (`ghostwave --doctor`)
  - CUDA setup
  - Performance optimization

- [**RTX 5090 Optimizations**](RTX_5090_OPTIMIZATIONS.md) - Blackwell architecture
  - FP4 Tensor Core acceleration
  - ASUS ROG Astral setup
  - Power management (630W)
  - Thermal profiles

### Performance Comparisons

| GPU | Latency | Architecture | Tensor Cores |
|-----|---------|--------------|--------------|
| RTX 2060 | ~15ms | Turing | 2nd Gen |
| RTX 3070 | ~10ms | Ampere | 3rd Gen |
| RTX 4090 | ~7ms | Ada Lovelace | 4th Gen |
| **RTX 5090** | **<5ms** | **Blackwell** | **5th Gen (FP4)** |
| CPU Only | ~25ms | N/A | N/A |

---

## Troubleshooting

- [**Troubleshooting Guide**](troubleshooting.md) - Common issues and solutions
  - Audio dropouts / XRuns
  - High latency
  - GPU acceleration issues
  - Driver problems

---

## Use Cases

### By Application

**Gaming & Streaming:**
- Profile: `balanced` (512 frames, ~10ms)
- RTX recommended: RTX 3060+
- Use PipeWire or ALSA backend

**Professional Recording:**
- Profile: `studio` (128 frames, <3ms)
- RTX recommended: RTX 4070+
- Use ALSA direct mode or PipeWire optimized

**Conference Calls:**
- Profile: `streaming` (1024 frames, ~20ms)
- RTX minimum: RTX 2060
- Use PipeWire backend

**Content Creation (Elite):**
- Profile: `studio` with RTX 5090
- GPU: RTX 5090 or ASUS ROG Astral
- Latency: <5ms with FP4 acceleration
- Use ALSA direct or optimized PipeWire

### By Hardware

**Entry Level (RTX 2060-3060):**
- Latency: 10-15ms
- Profile: Balanced or Streaming
- Good for: Gaming, casual streaming

**Mid-Range (RTX 3070-4070):**
- Latency: 7-10ms
- Profile: Balanced or Studio
- Good for: Professional streaming, recording

**High-End (RTX 4080-4090):**
- Latency: 3-7ms
- Profile: Studio
- Good for: Professional audio production

**Elite (RTX 5090):**
- Latency: <3ms (with FP4)
- Profile: Studio with optimizations
- Good for: Ultra-low latency professional work

---

## Hardware-Specific Guides

### Audio Interfaces

**Focusrite Scarlett Solo 4th Gen:**
- Optimized XLR configuration
- Auto-detection support
- Sample rate: 44.1kHz - 192kHz
- Buffer: 64-2048 frames

**Generic USB Audio:**
- Universal compatibility
- Auto-configuration
- Multiple backend support

### GPU Variants

**ASUS ROG Astral RTX 5090:**
- Factory OC: 2610MHz boost
- Quad-fan cooling
- 630W max power
- Silent operation for audio work
- See [RTX 5090 Optimizations](RTX_5090_OPTIMIZATIONS.md)

**Reference RTX Cards:**
- Stock clocks
- Standard cooling
- Lower power limits
- Still excellent performance

---

## Feature Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| **Audio Backends** |
| PipeWire | ✅ | Recommended for modern systems |
| ALSA | ✅ | Lowest latency |
| JACK | ✅ | Professional workflows |
| CPAL | ✅ | Cross-platform fallback |
| **GPU Acceleration** |
| RTX 20 series | ✅ | Compute 7.5 (Turing) |
| RTX 30 series | ✅ | Compute 8.6 (Ampere) |
| RTX 40 series | ✅ | Compute 8.9 (Ada) |
| RTX 50 series | ✅ | Compute 10.0 (Blackwell + FP4) |
| AMD GPU | ⚠️  | Planned (Vulkan compute) |
| Intel GPU | ⚠️  | Planned (oneAPI) |
| **Features** |
| Noise suppression | ✅ | AI-powered with GPU acceleration |
| Adaptive noise gate | ✅ | Dynamic threshold adjustment |
| Real-time processing | ✅ | <15ms latency target |
| Multi-profile support | ✅ | Studio, Balanced, Streaming |
| Device auto-detection | ✅ | XLR interfaces, USB audio |
| SystemD integration | ✅ | Auto-start, service management |
| IPC/JSON-RPC API | ✅ | External control |
| PhantomLink integration | ✅ | Professional mixer |
| nvcontrol integration | ✅ | GPU power management |

---

## External Resources

### NVIDIA

- [NVIDIA Open GPU Kernel Modules](https://github.com/NVIDIA/open-gpu-kernel-modules) - Open source drivers
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/) - CUDA programming guide
- [RTX Voice](https://www.nvidia.com/en-us/geforce/guides/nvidia-rtx-voice-setup-guide/) - Windows RTX Voice (inspiration)

### Audio

- [PipeWire Documentation](https://docs.pipewire.org/) - Modern Linux audio
- [ALSA Project](https://www.alsa-project.org/) - Advanced Linux Sound Architecture
- [JACK Audio](https://jackaudio.org/) - Professional audio connection kit

### Hardware

- [Focusrite Scarlett Solo](https://focusrite.com/en/scarlett) - XLR audio interface
- [ASUS ROG Graphics Cards](https://rog.asus.com/graphics-cards/) - ROG Astral series

---

## Contributing

See [Development Guide](DEVELOPMENT.md) for:
- Building and testing
- Code style guidelines
- Adding new features
- Performance requirements

---

## License

GhostWave is licensed under MIT OR Apache-2.0. See [LICENSE](../LICENSE) for details.

---

**Last Updated**: December 2025 (v0.2.0 - RTX 5090 Blackwell support)
