# GhostWave v0.2.0 Known Gaps

This document describes the current limitations and planned improvements for GhostWave v0.2.0.

## PipeWire Integration

### Current Status
- ✅ PipeWire module runs with proper node properties and latency hints
- ✅ Processing modes (low-latency, balanced, high-quality) are wired to CLI
- ✅ Presets (gaming, recording, rtx50) are selectable via `--pipewire-preset`
- ✅ Worker thread cleanup and stats tracking implemented
- ✅ PipeWire context and core connection established
- ✅ Node properties configured for session manager integration
- ⚠️ **Filter Node WIP**: PipeWire filter infrastructure created, full pw_filter callback processing in progress

### What Works
```bash
# These commands work and apply the correct settings
ghostwave --pipewire-module --pipewire-preset gaming      # 10ms latency
ghostwave --pipewire-module --processing-mode low-latency # NVIDIA Maxine style
ghostwave --pipewire-module --processing-mode high-quality # Recording mode
```

### Known Limitations
1. **pw_filter Callback Not Connected**: The PipeWire filter node is created with correct properties, but the audio processing callback isn't yet wired to the pw_filter process function. Audio is processed in a separate thread.

2. ~~**Virtual Devices Not Auto-Linked**~~: **RESOLVED in v0.2.0** - Use `--auto-link` flag to automatically connect to default audio devices. Requires `pw-link` (pipewire-tools package).

3. **Sample Rate Fixed at 48kHz**: While higher sample rates are accepted in config, only 48kHz is fully validated for the NVIDIA Maxine-compatible modes.

## RTX Acceleration

### Current Status
- ✅ RTX 50 series (Blackwell) compute 12.0 detection works
- ✅ Driver version checking for FP4 support (requires 590+)
- ✅ Graceful fallback to FP16 or CPU when FP4 not available
- ✅ Driver validation before enabling FP4 Tensor Core paths
- ✅ **CUDA Auto-Detection**: `build.rs` detects CUDA at build time and sets `has_cuda` cfg flag
- ⚠️ **Feature Flag Still Required**: While CUDA is auto-detected, you still need `--features nvidia-rtx` to enable RTX code paths

### What Works
```bash
# System diagnostics show correct GPU detection
ghostwave --doctor

# Shows (example for RTX 5090):
#   RTX GPU: NVIDIA GeForce RTX 5090
#   Compute: 12.0 (Gen Blackwell, Tensor Core Gen 5)
#   FP4: Available (driver 590+) or "Requires driver 590+"
#   Readiness -> Driver: true, CUDA: true, FP4: true/false
```

### Known Limitations
1. **nvidia-rtx Feature Not Default**: The RTX acceleration code requires explicit compilation with `--features nvidia-rtx`. Without this, CPU fallback is used.

2. **FP4 Requires Driver 590+**: Even on RTX 50 series hardware, FP4 Tensor Core paths require NVIDIA driver version 590 or newer. Drivers <590 will use FP16 paths instead.

3. **GPU Fallback Logged**: When GPU processing fails, the system falls back to CPU and logs a warning. This is now properly reported rather than silent.

## Configuration

### Current Status
- ✅ Profile-based configuration (balanced, streaming, studio)
- ✅ CLI overrides for sample rate, buffer size, latency
- ⚠️ New presets not in JSON config files

### Known Limitations
1. **Processing Modes Not in Config Files**: The new `low-latency`, `balanced`, `high-quality` modes must be specified via CLI flags, not in JSON config.

2. **Preset Defaults Not Persisted**: When using `--pipewire-preset rtx50`, settings are not saved for next run.

## Noise Suppression

### Current Status
- ✅ CPU spectral filtering works for basic noise reduction
- ✅ Noise gate with configurable threshold
- ⚠️ **Not Yet NVIDIA Broadcast Quality**: AI-based denoising uses simplified GRU, not full NVIDIA Maxine models

### Known Limitations
1. **No NVIDIA NIM Integration**: The code references NVIDIA Maxine NIM clients but doesn't yet integrate with their gRPC services.

2. **Voice Isolation Placeholder**: The `--voice-isolation` flag is accepted but the multi-speaker separation is not implemented.

## Recommended Workflow (v0.2.0)

For production use, we recommend:

```bash
# 1. Check system
ghostwave --doctor

# 2. Run with explicit settings
ghostwave --pipewire-module \
    --profile studio \
    --processing-mode balanced \
    --samplerate 48000 \
    --frames 512 \
    --verbose
```

## Roadmap

### v0.2.1 (Next)
- [ ] Real PipeWire filter node implementation
- [ ] Virtual device registration with WirePlumber
- [ ] Persist processing mode preferences

### v0.3.0 (Future)
- [ ] NVIDIA Maxine NIM integration (optional)
- [ ] Full voice isolation model
- [ ] Real-time GPU metrics API

---

*Last updated: v0.2.0*
