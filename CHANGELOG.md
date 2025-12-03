# Changelog

## [0.2.0] - 2025-12-03

### Added
- **Real PipeWire AudioStream Integration**: `--pipewire-module` now uses `ghostwave_core::AudioStream` for live audio processing instead of the simulated loop.
- **PipeWire Auto-Linking**: New `--auto-link` flag automatically connects GhostWave to default audio devices using `pw-link`. No more manual node linking required!
- **Processing Modes**: New `--processing-mode` flag with NVIDIA Maxine-compatible presets:
  - `low-latency`: 10ms (optimal for Discord/gaming)
  - `balanced`: 20ms (general use)
  - `high-quality`: 50ms (recording/production)
- **PipeWire Presets**: New `--pipewire-preset` flag with `gaming`, `recording`, and `rtx50` configurations.
- **CUDA Auto-Detection**: `build.rs` now detects CUDA installation at compile time and sets `has_cuda` cfg flag. Build output shows whether CUDA was detected.
- **GPU Fallback Telemetry**: `ProcessingStats` now tracks `gpu_fallback_active`, `gpu_fallback_count`, and `gpu_fallback_reason` for monitoring.
- **Info-Level Mode Logging**: First-time processing logs whether FP4, FP16, or CPU mode is active at `info` level.
- `check_rtx_system_requirements()` reports driver/CUDA/TensorRT readiness plus FP4 capability and Tensor Core generation.
- Shared `RtxDenoiser` cache with `OnceLock` to reuse CUDA plans across the pipeline.
- CLI `--doctor` integrates the new diagnostics output so users can validate RTX readiness without compiling GPU features.

### Changed
- `ghostwave-core` GPU path auto-selects FP4 Tensor Cores on RTX 50 "Blackwell" hardware with CPU fallback retained for unsupported devices.
- Noise suppression pipeline highlights active GPU/CPU modes, improving log clarity during troubleshooting.
- PipeWire module now properly shuts down worker threads with `AtomicBool` stop flag and `JoinHandle` cleanup.

### Fixed
- Resolved stale GPU buffer reuse by re-checking FFT sizes per block, preventing mismatched allocations when switching profiles.
- Fixed `test_device_scoring` - score calculation now includes all criteria.
- Fixed `test_channel_operations` and `test_format_conversion` - buffer sizes now meet 32-frame minimum.
- Fixed `test_parameter_operations` - `get_param` returns plain JSON values instead of wrapped `ParamValue`.

### Known Limitations
> **RTX Feature Flag**: While CUDA is auto-detected at build time, RTX GPU acceleration still requires `--features nvidia-rtx`. Runtime CUDA loading is planned for v0.3.0.

> **WirePlumber Integration**: The `--auto-link` flag uses `pw-link` for device connection. Native WirePlumber metadata integration is planned for v0.3.0.

See [docs/KNOWN_GAPS.md](docs/KNOWN_GAPS.md) for the complete list of current limitations and planned improvements.
