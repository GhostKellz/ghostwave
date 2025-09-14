# GhostWave — Finish Line TODO

## Core & API
- [ ] Finalize processing trait/API (`init`, `process_inplace`, `set_profile`, `set_param`) with docs.
- [ ] Split crates: `ghostwave-core` (DSP) and `ghostwave` (CLI/daemon); keep `no_std`-friendly core where possible.
- [ ] Define stable frame format (f32, interleaved; 48kHz default; 1–2ch).

## DSP Pipeline
- [ ] Lock the chain: HPF → VAD → spectral denoise → expander/gate → soft-clip/limiter.
- [ ] Tune three presets (Balanced/Streaming/Studio) with saved params.
- [ ] Add runtime-safe guardrails (NaN/Inf scrub, denormal disable, headroom).

## Latency & RT
- [ ] Target <15 ms E2E (48kHz, 128f block); support 96kHz/256f for Studio.
- [ ] Optional lock-free ring buffer; avoid allocations in audio callback.
- [ ] Set RT prio when allowed; graceful fallback when not.

## PipeWire Integration
- [ ] `--pipewire-module` path that exposes a named node (“GhostWave Clean”).
- [ ] Auto device select (`--input auto`) with debounce on hotplug.
- [ ] Node props: `media.class = Audio/Source`, friendly nick, channel map.

## Config & Persistence
- [ ] `~/.config/ghostwave/config.toml` with schema + defaults.
- [ ] CLI overrides (`--profile`, `--frames`, `--samplerate`, `--input`, `--output`).
- [ ] Safe reload on file change (SIGHUP or inotify).

## Control Plane (IPC)
- [ ] JSON-RPC over UNIX socket (`$XDG_RUNTIME_DIR/ghostwave.sock`).
- [ ] Methods: `get_profile`, `set_profile`, `get_params`, `set_param`, `levels`, `stats`, `version`.
- [ ] Versioned schema & simple auth (file perms, abstract socket optional).

## Diagnostics & Bench
- [ ] `ghostwave doctor`: print devices, PipeWire graph, driver, CPU/GPU accel availability.
- [ ] `ghostwave bench`: per-block processing time (avg/99p), XRuns, effective latency.
- [ ] `--dry-run` validation mode.

## Packaging
- [ ] Systemd **user** unit (`/usr/lib/systemd/user/ghostwave.service`) with sane defaults.
- [ ] PKGBUILD(s): `ghostwave` (bin + service) and `ghostwave-core` (lib).
- [ ] Completions for bash/zsh/fish; man page (`--help` → scdoc).

## Acceleration (optional/feature-gated)
- [ ] CPU baseline (SIMD/AVX2 when available).
- [ ] Feature flags for CUDA/TensorRT or Vulkan compute; runtime selection & fallback.

## Logging & Telemetry
- [ ] Structured logs (warn! info!); ring buffer for recent errors.
- [ ] `--verbose` and `--quiet`; no network telemetry.

## Testing & CI
- [ ] Unit tests for DSP blocks (impulse, sine sweeps, noise).
- [ ] Golden audio tests (A/B fixtures) with tolerance windows.
- [ ] CI matrix: stable + nightly, x86_64 + aarch64 (no-GPU path).

## Docs
- [ ] README quick-start; examples for each profile.
- [ ] IPC API doc; config reference; troubleshooting (XRuns, crackle, drift).
- [ ] “Integrating with PhantomLink” mini-guide.

## PhantomLink Integration
- [ ] Add `GhostWaveNode` in PhantomLink’s graph (PreFX → GhostWave → PostFX).
- [ ] Live control via the JSON-RPC; smooth 10–20 ms crossfade on profile/param change.
- [ ] QA pass with your Scarlett Solo 4th gen and streaming apps.

## Release
- [ ] `v0.1.0` tag, changelog, reproducible build flags.
- [ ] Prebuilt binaries (Linux x86_64/aarch64); checksums.
- [ ] Open two tracking issues: “GUI control panel (egui)” and “Vulkan/CUDA accel”.
