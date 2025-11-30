# GhostWave — Development Roadmap

## ✅ Completed Features

### Core & API
- [x] Processing trait/API (`init`, `process_inplace`, `set_profile`, `set_param`) with docs
- [x] Split crates: `ghostwave-core` (DSP) and `ghostwave` (CLI/daemon)
- [x] Stable frame format (f32, interleaved; 48kHz default; 1–2ch)

### DSP Pipeline
- [x] Complete chain: HPF → VAD → spectral denoise → expander/gate → soft-clip/limiter
- [x] Three presets (Balanced/Streaming/Studio) with saved params
- [x] Runtime-safe guardrails (NaN/Inf scrub, denormal disable, headroom)

### AI-Powered Denoising (NVIDIA Broadcast / Krisp Parity)
- [x] RNNoise-style GRU-based noise suppression
- [x] TensorRT/ONNX inference engine integration
- [x] FP16 Tensor Core acceleration (RTX 20/30/40)
- [x] FP4 Tensor Core acceleration (RTX 50 Blackwell)
- [x] Multi-model support (Tiny/Standard/Large/Transformer)
- [x] Voice Activity Detection (VAD) with adaptive thresholds
- [x] Bark-scale spectral feature extraction

### Echo Cancellation
- [x] Acoustic Echo Canceller (AEC) with NLMS adaptive filter
- [x] Double-talk detection
- [x] Non-linear residual echo processing
- [x] Full-duplex AEC with loopback detection
- [x] Configurable tail length (up to 500ms)

### Voice Isolation
- [x] Primary speaker isolation
- [x] Speaker embedding/fingerprinting
- [x] Speaker enrollment for targeted isolation
- [x] Multi-speaker separation framework
- [x] Frequency-domain voice masking

### RTX GPU Acceleration
- [x] CUDA context management
- [x] TensorRT engine building and caching
- [x] Automatic precision selection (FP32/FP16/FP4)
- [x] Architecture detection (Turing → Blackwell)
- [x] nvidia-open 580+ driver support for RTX 5090
- [x] GPU memory management and buffer pooling

### Model Management
- [x] Model directory scanning (user + system)
- [x] TensorRT engine caching
- [x] Model validation
- [x] Download from model repository (framework)

### PipeWire Integration
- [x] `--pipewire-module` path exposing named node ("GhostWave Clean")
- [x] Auto device select with hotplug debounce
- [x] Node props: `media.class = Audio/Source`

### Packaging
- [x] Systemd user unit (`ghostwave.user.service`)
- [x] PKGBUILD for Arch Linux
- [x] Default configuration (`config/default.toml`)

---

## 🔄 In Progress

### Latency & RT
- [ ] Target <10ms E2E with RTX acceleration
- [ ] Lock-free ring buffer optimization
- [ ] RT priority scheduling

### Config & Persistence
- [x] `~/.config/ghostwave/config.toml` with schema + defaults
- [ ] CLI overrides (`--profile`, `--frames`, `--samplerate`)
- [ ] Safe reload on file change (SIGHUP or inotify)

---

## 📋 Pending

### Control Plane (IPC)
- [ ] JSON-RPC over UNIX socket (`$XDG_RUNTIME_DIR/ghostwave.sock`)
- [ ] Methods: `get_profile`, `set_profile`, `get_params`, `set_param`, `levels`, `stats`, `version`
- [ ] Versioned schema & simple auth

### Diagnostics & Bench
- [ ] `ghostwave doctor`: devices, PipeWire graph, driver, GPU accel availability
- [ ] `ghostwave bench`: per-block processing time (avg/99p), XRuns, latency
- [ ] `--dry-run` validation mode
- [ ] Mic testing/preview functionality

### Logging & Telemetry
- [ ] Structured logs (warn! info!); ring buffer for recent errors
- [ ] `--verbose` and `--quiet`; no network telemetry

### Testing & CI
- [ ] Unit tests for DSP blocks (impulse, sine sweeps, noise)
- [ ] Golden audio tests (A/B fixtures) with tolerance windows
- [ ] CI matrix: stable + nightly, x86_64 + aarch64 (no-GPU path)

### Docs
- [ ] README quick-start; examples for each profile
- [ ] IPC API doc; config reference; troubleshooting
- [ ] "Integrating with PhantomLink" mini-guide

### PhantomLink Integration
- [ ] Add `GhostWaveNode` in PhantomLink's graph (PreFX → GhostWave → PostFX)
- [ ] Live control via JSON-RPC; smooth crossfade on profile change
- [ ] QA pass with Scarlett Solo 4th gen and streaming apps

### Release
- [ ] `v0.1.0` tag, changelog, reproducible build flags
- [ ] Prebuilt binaries (Linux x86_64/aarch64); checksums
- [ ] Open tracking issues: "GUI control panel (egui)" and "Vulkan compute accel"

---

## 🚀 Future Enhancements

### Transformer Models (Premium)
- [ ] Train custom transformer denoiser
- [ ] Multi-head attention for temporal context
- [ ] Export to TensorRT with FP4 quantization

### Advanced Features
- [ ] Room impulse response estimation
- [ ] Automatic gain control (AGC)
- [ ] De-reverb processing
- [ ] Background music separation

### Platform Support
- [ ] Fedora/RPM packaging
- [ ] Debian/Ubuntu .deb packaging
- [ ] Bazzite/Nobara optimized builds
- [ ] Pop!_OS integration

### Hardware Support
- [ ] AMD AMF fallback (future)
- [ ] Intel OneAPI/QuickSync (future)
- [ ] Vulkan compute backend

---

## Hardware Targets

### Primary (Optimized)
- **NVIDIA RTX 50 Series (Blackwell)**: FP4 Tensor Cores, nvidia-open 580+
- **NVIDIA RTX 40 Series (Ada)**: FP16 Tensor Cores
- **NVIDIA RTX 30 Series (Ampere)**: FP16 Tensor Cores

### Secondary (Supported)
- **NVIDIA RTX 20 Series (Turing)**: FP16 Tensor Cores
- **NVIDIA GTX 16 Series**: CUDA compute (no Tensor Cores)
- **CPU-only**: AVX2/SIMD optimized fallback

### Audio Interfaces Tested
- Scarlett Solo 4th Gen (PhantomLink target)
- Rode PodMic
- Various USB audio devices

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Applications                             │
│        (Discord, OBS, Streaming apps, PhantomLink)         │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                    GhostWave Daemon                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  Audio Pipeline                       │  │
│  │  Input → AEC → AI Denoise → Voice Isolation → Output │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   PipeWire  │  │    IPC      │  │   Model Manager     │ │
│  │   Backend   │  │  (JSON-RPC) │  │   (ONNX/TensorRT)   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                   Inference Engine                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                TensorRT Runtime                      │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐        │   │
│  │  │ RNNoise   │  │  Voice    │  │   Echo    │        │   │
│  │  │  Model    │  │ Isolation │  │   AEC     │        │   │
│  │  └───────────┘  └───────────┘  └───────────┘        │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│                     NVIDIA RTX GPU                          │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │  Tensor Cores  │  │   CUDA Cores   │  │  GPU Memory  │  │
│  │  (FP16/FP4)    │  │   (General)    │  │   (VRAM)     │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
└────────────────────────────────────────────────────────────┘
```
