<p align="center">
  <img src="assets/ghostwave-logo.png" alt="GhostWave Logo" width="256" height="256">
</p>

<h1 align="center">GhostWave</h1>

<p align="center">
  <strong>NVIDIA RTX Voice-Powered Noise Suppression for Linux</strong><br>
  <em>Wayland-ready | Low-latency | Built for creators & gamers</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/Rust_1.95+-F74C00?style=for-the-badge&logo=rust&logoColor=white" alt="Rust 1.95+">
  <img src="https://img.shields.io/badge/NVIDIA_RTX-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA RTX Optimized">
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/Linux_Native-FCC624?style=for-the-badge&logo=linux&logoColor=black" alt="Linux Native">
  <img src="https://img.shields.io/badge/PipeWire-00A8E8?style=for-the-badge&logo=linux&logoColor=white" alt="PipeWire">
  <img src="https://img.shields.io/badge/Wayland-FFB800?style=for-the-badge&logo=wayland&logoColor=black" alt="Wayland">
  <img src="https://img.shields.io/badge/ONNX_Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX Runtime">
</p>

---

> **Status: Open Beta**
>
> GhostWave provides a working real-time DSP pipeline with PipeWire integration.
> GPU-accelerated neural inference is under development. The current release uses
> CPU-based spectral processing for noise reduction.

---

## What It Does

GhostWave creates a virtual audio source ("GhostWave Clean") in PipeWire that applies
real-time noise reduction to your microphone input. Applications see it as a regular
audio device.

### Working Features
- **DSP Processing Pipeline** — High-pass filter, voice activity detection, spectral
  denoising, noise gate/expander, soft limiter with lookahead
- **PipeWire Filter Node** — Appears as a virtual audio source in any application
- **Multiple Audio Backends** — PipeWire (recommended), ALSA, JACK, CPAL
- **Processing Profiles** — Balanced, Streaming, Studio configurations
- **NVIDIA GPU Detection** — Detects RTX capabilities, compute architecture, driver version
- **Hardware Auto-Detection** — XLR interface identification (Scarlett Solo, etc.)
- **JSON-RPC IPC** — Remote control over UNIX sockets
- **VST3/CLAP Plugin** — Plugin shell with GUI (via nih-plug + egui)
- **SystemD Integration** — User service for auto-startup
- **CLI Tools** — System diagnostics (`--doctor`), benchmarking (`--bench`)

### In Development
- Neural network inference (RNNoise-style model loading and execution)
- CUDA kernel integration for GPU-accelerated denoising
- TensorRT engine support
- ONNX Runtime model inference
- Full VST plugin audio processing pipeline

---

## Installation

### Prerequisites
- **Linux** with PipeWire (PulseAudio compatibility layer works too)
- **Rust 1.85+** (2024 edition)
- **Optional**: NVIDIA GPU with CUDA for future GPU acceleration

### System Dependencies
```bash
# Arch Linux
sudo pacman -S pipewire pipewire-pulse wireplumber alsa-lib

# Ubuntu/Debian
sudo apt install pipewire pipewire-pulse wireplumber libasound2-dev

# Fedora
sudo dnf install pipewire pipewire-pulseaudio wireplumber alsa-lib-devel
```

### Build from Source
```bash
git clone https://github.com/ghostkellz/ghostwave
cd ghostwave
cargo build --release

# Optional: enable real-time audio privileges
sudo setcap cap_sys_nice+ep ./target/release/ghostwave
```

### Arch Linux (PKGBUILD)
```bash
cd release/arch
makepkg -si
```

---

## Usage

### Quick Start
```bash
# Run system diagnostics
ghostwave --doctor

# Start PipeWire filter node (recommended)
ghostwave --pipewire-module --profile balanced

# Performance benchmark
ghostwave --bench --profile studio
```

### Profiles
```bash
# Balanced: 48kHz, 128 frames, moderate noise reduction
ghostwave --profile balanced

# Streaming: 48kHz, 128 frames, aggressive noise reduction
ghostwave --profile streaming

# Studio: 96kHz, 256 frames, gentle processing
ghostwave --profile studio
```

### Backend Selection
```bash
# ALSA direct mode
ghostwave --alsa --frames 64 --samplerate 48000

# JACK integration
ghostwave --jack --profile studio

# PipeWire module mode (creates virtual device)
ghostwave --pipewire-module
```

### Integration
```bash
# IPC server for external control
ghostwave --ipc-server --profile balanced

# PhantomLink integration
ghostwave --phantomlink --profile streaming

# Install as systemd user service
ghostwave --install-systemd
systemctl --user enable --now ghostwave
```

---

## Configuration

Default config location: `~/.config/ghostwave/`

See [`examples/config.toml`](examples/config.toml) for all available options including:
- Audio device and backend selection
- DSP pipeline parameters (VAD, gate, limiter, high-pass)
- Noise suppression strength and thresholds
- IPC server settings
- Performance tuning (real-time priority, CPU affinity, buffer sizes)

---

## Architecture

```
ghostwave/
├── ghostwave-core/          # Core DSP library (rlib + cdylib + staticlib)
│   ├── processor.rs         # Main audio processing pipeline
│   ├── dsp_pipeline.rs      # HPF → VAD → Denoise → Gate → Limiter
│   ├── noise_suppression.rs # Spectral noise reduction
│   ├── pipewire_integration.rs  # PipeWire filter node
│   ├── device_detection.rs  # Hardware auto-configuration
│   ├── ai_denoise/          # Neural inference framework (in development)
│   └── rtx_acceleration.rs  # GPU capability detection
├── ghostwave-vst/           # VST3/CLAP plugin (nih-plug + egui)
└── src/                     # CLI application
    ├── main.rs              # Argument parsing and run modes
    ├── ipc.rs               # JSON-RPC server
    ├── pipewire_module.rs   # PipeWire integration entry point
    └── phantomlink.rs       # PhantomLink bridge
```

### As a Rust Crate
```toml
[dependencies]
ghostwave-core = { git = "https://github.com/ghostkellz/ghostwave" }
```

### Feature Flags
| Feature | Description |
|---------|-------------|
| `cpal-backend` (default) | Cross-platform audio via CPAL |
| `pipewire-backend` | PipeWire native integration |
| `alsa-backend` | Direct ALSA hardware access |
| `jack-backend` | JACK professional audio |
| `nvidia-rtx` | CUDA GPU detection and acceleration |
| `onnx-inference` | ONNX Runtime model inference |
| `full` | All backends and GPU features |

---

## Documentation

See the [`docs/`](docs/README.md) directory:

- [PipeWire](docs/backends/pipewire.md) — PipeWire module setup
- [ALSA](docs/backends/alsa.md) — Direct ALSA integration
- [NVIDIA](docs/gpu/nvidia.md) — GPU acceleration setup
- [Architecture](docs/development/architecture.md) — System design
- [Performance](docs/development/performance.md) — Tuning and benchmarking
- [API Reference](docs/development/api-reference.md) — Library API
- [Known Gaps](docs/known-gaps.md) — Current limitations
- [Troubleshooting](docs/troubleshooting.md) — Common issues

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
