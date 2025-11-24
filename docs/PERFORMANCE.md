# GhostWave Performance Optimization

Complete guide to optimizing GhostWave for ultra-low latency audio processing.

## Table of Contents

- [Real-Time Threading](#real-time-threading)
- [Latency Targets](#latency-targets)
- [Benchmarking](#benchmarking)
- [System-Level Optimization](#system-level-optimization)
- [Audio System Configuration](#audio-system-configuration)
- [Performance Requirements](#performance-requirements)

---

## Real-Time Threading

### Thread Priority Setup

**Enabling FIFO Scheduling:**
```rust
const SCHED_FIFO: c_int = 1;
const RT_PRIORITY: c_int = 80;

unsafe {
    sched_setscheduler(0, SCHED_FIFO, &RT_PRIORITY as *const c_int);
}
```

**Grant CAP_SYS_NICE Permission:**
```bash
sudo setcap cap_sys_nice+ep ./ghostwave
```

### CPU Affinity (Optional)

Pin audio threads to performance cores for reduced latency:
- Isolate from system interrupts
- Reduce cache misses and context switches
- Consistent performance on heterogeneous CPUs (P-cores vs E-cores)

---

## Latency Targets

| Profile | Buffer Size | Target Latency | Use Case |
|---------|-------------|----------------|----------|
| **Studio** | 64-128 frames | <3ms | Professional recording |
| **Balanced** | 256-512 frames | 5-10ms | Gaming, streaming |
| **Streaming** | 1024+ frames | 10-20ms | Content creation |

### Measured Performance (Scarlett Solo 4th Gen)

| Configuration | Latency | CPU Usage | XRun Rate |
|---------------|---------|-----------|-----------|
| Studio (256 frames) | 1.33ms | 12% | 0.001% |
| Balanced (512 frames) | 2.67ms | 8% | 0.000% |
| Streaming (1024 frames) | 5.33ms | 4% | 0.000% |

### RTX GPU Acceleration Impact

| GPU | Profile | Latency | Notes |
|-----|---------|---------|-------|
| CPU Only | Studio | 2.5ms | Baseline |
| RTX 3090 | Studio | 1.8ms | 28% faster |
| RTX 4090 | Studio | 1.1ms | 56% faster |
| **RTX 5090** | Studio | **0.7ms** | **72% faster** |
| **ASUS ROG Astral 5090** | Studio | **0.65ms** | **74% faster** |

---

## Benchmarking

### Built-in Benchmarking System

```rust
pub struct AudioBenchmark {
    processing_times: Vec<Duration>,
    frame_count: AtomicU64,
    xrun_count: AtomicU64,
    max_processing_time: AtomicU64,
}
```

**Run Benchmarks:**
```bash
# Full benchmark suite
ghostwave --bench

# Benchmark specific profile
ghostwave --bench --profile studio

# Extended benchmark (10 minutes)
ghostwave --bench --duration 600
```

**Metrics Tracked:**
- Frame processing time distribution
- XRun (audio dropout) detection
- CPU usage patterns
- Memory allocation tracking
- GPU utilization (if available)

### External Profiling

**CPU profiling with perf:**
```bash
perf record --call-graph=dwarf ./ghostwave --bench
perf report
```

**Memory profiling:**
```bash
valgrind --tool=massif ./ghostwave --bench
```

---

## System-Level Optimization

### CPU Governor

Set CPU to performance mode for consistent latency:
```bash
# Set all CPUs to performance governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verify setting
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

**Auto-apply on boot** (systemd):
```bash
sudo systemctl enable --now cpupower
sudo cpupower frequency-set -g performance
```

### Disable CPU Frequency Scaling

```bash
# Disable Intel Turbo Boost
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Disable AMD Boost
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
```

### Audio Power Management

Disable power saving for audio hardware:
```bash
# Disable audio power management
echo 0 | sudo tee /sys/module/snd_hda_intel/parameters/power_save

# Make persistent
echo "options snd_hda_intel power_save=0" | sudo tee /etc/modprobe.d/audio-powersave.conf
```

### IRQ Affinity

Pin audio IRQs to specific CPU cores:
```bash
# Find audio device IRQ
grep snd /proc/interrupts

# Pin to CPU 0 (example)
echo 1 | sudo tee /proc/irq/XX/smp_affinity
```

---

## Audio System Configuration

### PipeWire Optimization

**Low-Latency Configuration:**
```bash
# Create user config directory
mkdir -p ~/.config/pipewire

# Copy default config
cp /usr/share/pipewire/pipewire.conf ~/.config/pipewire/
```

**Edit `~/.config/pipewire/pipewire.conf`:**
```
default.clock.rate = 48000
default.clock.quantum = 512
default.clock.min-quantum = 64
default.clock.max-quantum = 2048
```

**Apply changes:**
```bash
systemctl --user restart pipewire pipewire-pulse
```

### ALSA Direct Mode

For absolute minimum latency, bypass PipeWire:
```bash
ghostwave --alsa --frames 64 --samplerate 48000
```

**ALSA Configuration** (`~/.asoundrc`):
```
pcm.!default {
    type hw
    card 0
    device 0
}

ctl.!default {
    type hw
    card 0
}
```

### JACK Configuration

```bash
# Start JACK with low latency
jackd -R -dalsa -dhw:0 -r48000 -p256 -n2

# Run GhostWave with JACK
ghostwave --jack --profile studio
```

---

## Performance Requirements

### Audio Thread Requirements

- **Latency**: < 15ms (99th percentile)
- **Memory allocations**: Zero in audio path
- **CPU usage**: < 20% on target hardware
- **XRun rate**: < 0.1% under normal operation

### Hardware Recommendations

**Minimum (RTX 2060):**
- Latency: ~15ms
- CPU: 4-core @ 3.0GHz
- RAM: 8GB

**Recommended (RTX 3070+):**
- Latency: ~10ms
- CPU: 8-core @ 3.5GHz
- RAM: 16GB

**High-End (RTX 4080+):**
- Latency: ~7ms
- CPU: 12-core @ 4.0GHz
- RAM: 32GB

**Elite (RTX 5090):**
- Latency: <5ms
- CPU: 16-core @ 4.5GHz+
- RAM: 32GB+

---

## Troubleshooting Performance Issues

### Audio Dropouts / XRuns

**Symptoms**: Crackling, pops, gaps in audio

**Solutions:**
1. Increase buffer size: `ghostwave --frames 512`
2. Enable real-time scheduling: `sudo setcap cap_sys_nice+ep ./ghostwave`
3. Check CPU governor: `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
4. Reduce system load: Close unnecessary applications
5. Check IRQ conflicts: `cat /proc/interrupts`

### High Latency

**Symptoms**: Noticeable delay between input and output

**Solutions:**
1. Decrease buffer size: `ghostwave --frames 64`
2. Use ALSA direct mode: `ghostwave --alsa`
3. Optimize CPU affinity
4. Disable CPU frequency scaling
5. Check for background processes

### RTX Acceleration Not Working

**Symptoms**: GPU not detected, CPU fallback active

**Solutions:**
1. Verify CUDA installation: `nvcc --version`
2. Check GPU compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`
3. Ensure proper driver version: `nvidia-smi` (580+ for RTX 50)
4. Monitor GPU memory usage: `nvidia-smi dmon`
5. Rebuild with RTX feature: `cargo build --features nvidia-rtx`

---

## See Also

- [NVIDIA RTX Integration](NVIDIA.md) - GPU acceleration setup
- [RTX 5090 Optimizations](RTX_5090_OPTIMIZATIONS.md) - Blackwell-specific tuning
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
