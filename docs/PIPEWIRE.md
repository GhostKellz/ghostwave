# PipeWire Integration Guide

This guide covers GhostWave's integration with PipeWire, the modern Linux audio system. PipeWire provides low-latency, professional-grade audio routing with excellent Wayland support.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Modes](#usage-modes)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Why PipeWire?

PipeWire is the recommended audio backend for GhostWave on modern Linux systems:

- **Low Latency**: Sub-5ms round-trip latency achievable
- **Wayland Native**: First-class support for modern desktop environments
- **Professional Routing**: Complex audio graph management
- **Security**: Sandboxed audio processing
- **Compatibility**: Drop-in replacement for PulseAudio and JACK

### GhostWave PipeWire Integration

> **Note (v0.2.0)**: GhostWave now uses `AudioStream` from `ghostwave_core` for real-time
> audio processing. The low-level `pw_filter` callback integration is work-in-progress.
> When `AudioStream` is unavailable, a simulated processing loop is used as fallback.
> See [KNOWN_GAPS.md](KNOWN_GAPS.md) for current limitations.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   GhostWave     │    │   PipeWire      │
│                 │    │   Module        │    │   Server        │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │  Audio    │  │◄──►│  │   Noise   │  │◄──►│  │   Audio   │  │
│  │  Stream   │  │    │  │ Processor │  │    │  │   Graph   │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
│                 │    │                 │    │                 │
│  Discord/OBS    │    │  RTX Accel +    │    │  Hardware I/O   │
│  Games/etc      │    │  Real-time DSP  │    │  Routing/Mixing │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Installation

### System Requirements

**Arch Linux:**
```bash
sudo pacman -S pipewire pipewire-pulse pipewire-jack wireplumber
systemctl --user enable --now pipewire pipewire-pulse wireplumber
```

**Ubuntu 22.04+:**
```bash
sudo apt install pipewire pipewire-pulse pipewire-jack
systemctl --user enable --now pipewire pipewire-pulse
```

**Fedora 35+:**
```bash
sudo dnf install pipewire pipewire-pulseaudio pipewire-jack-audio-connection-kit
systemctl --user enable --now pipewire pipewire-pulse
```

### Verify Installation

```bash
# Check PipeWire is running
systemctl --user status pipewire

# List audio devices
pw-cli list-objects

# Test audio routing
pw-play /usr/share/sounds/alsa/Front_Left.wav
```

### GhostWave PipeWire Module

Install GhostWave's PipeWire module:

```bash
# Build with PipeWire support
cargo build --features "pipewire-backend,nvidia-rtx"

# Install PipeWire module
./ghostwave --install-pipewire

# Start as PipeWire module
./ghostwave --pipewire-module --profile balanced
```

---

## Configuration

### PipeWire Quantum Settings

For optimal latency, configure PipeWire's quantum (buffer size):

```bash
# Create custom PipeWire config
mkdir -p ~/.config/pipewire
cp /usr/share/pipewire/pipewire.conf ~/.config/pipewire/
```

Edit `~/.config/pipewire/pipewire.conf`:

```ini
context.properties = {
    default.clock.rate        = 48000
    default.clock.quantum     = 256      # For balanced latency
    default.clock.min-quantum = 64       # Minimum for low latency
    default.clock.max-quantum = 1024     # Maximum for stability

    # For professional audio interfaces
    default.clock.quantum     = 128      # Studio latency
    default.clock.min-quantum = 32
}
```

### GhostWave PipeWire Configuration

Create GhostWave-specific PipeWire module configuration:

```bash
# ~/.config/pipewire/pipewire.conf.d/ghostwave.conf
context.modules = [
    {   name = libpipewire-module-filter-chain
        args = {
            node.description = "GhostWave Noise Suppressor"
            media.name       = "GhostWave"
            filter.graph = {
                nodes = [
                    {
                        type   = ladspa
                        name   = ghostwave
                        plugin = /usr/local/lib/ladspa/ghostwave.so
                        label  = noise_suppressor
                        control = {
                            "Suppression Strength" = 0.8
                            "Gate Threshold" = -30.0
                        }
                    }
                ]
            }
            capture.props = {
                node.name      = "ghostwave_input"
                media.class    = "Audio/Sink"
                audio.channels = 2
                audio.position = [ FL FR ]
            }
            playback.props = {
                node.name      = "ghostwave_output"
                media.class    = "Audio/Source"
                audio.channels = 2
                audio.position = [ FL FR ]
            }
        }
    }
]
```

### Audio Routing Configuration

Set up automatic routing for applications:

```bash
# Route Discord through GhostWave
pw-cli create-object adapter { factory.name=support.null-audio-sink object.linger=true node.name=discord-input media.class=Audio/Sink }

# Connect Discord output to GhostWave input
pw-link discord-input:monitor_FL ghostwave_input:input_FL
pw-link discord-input:monitor_FR ghostwave_input:input_FR

# Connect GhostWave output to system output
pw-link ghostwave_output:output_FL alsa_output.pci-0000_00_1f.3.analog-stereo:playback_FL
pw-link ghostwave_output:output_FR alsa_output.pci-0000_00_1f.3.analog-stereo:playback_FR
```

---

## Usage Modes

### 1. Native PipeWire Module

Run as a native PipeWire module for seamless integration:

```bash
# Start module with auto-linking to default audio devices (recommended)
./ghostwave --pipewire-module --auto-link --profile balanced

# With specific processing mode (NVIDIA Maxine compatible)
./ghostwave --pipewire-module --auto-link --processing-mode low-latency   # 10ms (gaming/Discord)
./ghostwave --pipewire-module --auto-link --processing-mode balanced      # 20ms (general use)
./ghostwave --pipewire-module --auto-link --processing-mode high-quality  # 50ms (recording)

# With preset configurations
./ghostwave --pipewire-module --auto-link --pipewire-preset gaming     # Optimized for Discord
./ghostwave --pipewire-module --auto-link --pipewire-preset recording  # High-quality audio
./ghostwave --pipewire-module --auto-link --pipewire-preset rtx50      # RTX 50 series optimized

# Without auto-linking (for manual routing via qpwgraph)
./ghostwave --pipewire-module --profile studio --verbose

# Monitor performance
./ghostwave --pipewire-module --bench
```

> **v0.2.0 Auto-Linking**:
> - Use `--auto-link` to automatically connect GhostWave to your default audio devices
> - Requires `pw-link` (part of `pipewire-tools` package)
> - If auto-linking fails, you can manually link using `pw-link` or `qpwgraph`
> - Virtual devices `ghostwave_input` and `ghostwave_output` are created
> - Use `pw-cli ls Node` to verify GhostWave appears in the node list

**Benefits:**
- Automatic audio routing
- Low CPU overhead
- Persistent across applications
- System-wide noise suppression

### 2. Application-Specific Processing

Target specific applications:

```bash
# Discord/gaming setup
./ghostwave --pipewire-module --profile streaming --frames 512

# Professional streaming
./ghostwave --pipewire-module --profile studio --frames 256 --samplerate 96000

# Content creation
./ghostwave --pipewire-module --profile balanced --verbose
```

### 3. Real-time Monitoring

Monitor audio processing in real-time:

```rust
use ghostwave_core::pipewire::PipeWireModule;

let mut module = PipeWireModule::new(config)?;

// Start processing with latency monitoring
module.start_processing().await?;

// Get real-time statistics
loop {
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    if let Some(stats) = module.get_latency_stats() {
        println!("Latency: {}", stats);
    }
}
```

---

## Performance Optimization

### 1. Real-time Scheduling

Enable real-time priority for optimal performance:

```bash
# Set capabilities for real-time scheduling
sudo setcap cap_sys_nice+ep ./ghostwave

# Add user to audio group
sudo usermod -a -G audio $USER

# Configure system limits
echo '@audio - rtprio 95' | sudo tee -a /etc/security/limits.conf
echo '@audio - memlock unlimited' | sudo tee -a /etc/security/limits.conf
```

### 2. CPU Governor and IRQ Tuning

```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable audio power management
echo 0 | sudo tee /sys/module/snd_hda_intel/parameters/power_save

# IRQ affinity for audio (optional)
echo 2 | sudo tee /proc/irq/$(cat /proc/interrupts | grep audio | cut -d: -f1)/smp_affinity
```

### 3. Memory and Thread Optimization

```rust
use ghostwave_core::pipewire::LatencyMonitor;

struct OptimizedPipeWireProcessor {
    module: PipeWireModule,
    monitor: LatencyMonitor,
}

impl OptimizedPipeWireProcessor {
    pub fn new(config: Config) -> anyhow::Result<Self> {
        // Pin to specific CPU cores for consistency
        let core_mask = 0b1100; // Cores 2 and 3
        unsafe {
            libc::sched_setaffinity(0, std::mem::size_of_val(&core_mask), &core_mask);
        }

        let module = PipeWireModule::new(config)?;
        let monitor = LatencyMonitor::new();

        Ok(Self { module, monitor })
    }

    pub async fn run_optimized(&mut self) -> anyhow::Result<()> {
        // Pre-allocate audio buffers
        let mut input_buffer = vec![0.0f32; 2048];
        let mut output_buffer = vec![0.0f32; 2048];

        loop {
            let start = std::time::Instant::now();

            // Process audio frame
            self.module.process_audio_block(&input_buffer, &mut output_buffer)?;

            // Monitor latency
            let processing_time = start.elapsed();
            self.monitor.record_frame(processing_time);

            // Report every 1000 frames
            if self.monitor.frame_count % 1000 == 0 {
                self.monitor.report_average_latency();
            }
        }
    }
}
```

### 4. Quantum and Rate Optimization

```bash
# For gaming (balanced latency/CPU)
pw-metadata -n settings 0 clock.force-quantum 512
pw-metadata -n settings 0 clock.force-rate 48000

# For streaming (lower latency)
pw-metadata -n settings 0 clock.force-quantum 256
pw-metadata -n settings 0 clock.force-rate 48000

# For professional audio (minimal latency)
pw-metadata -n settings 0 clock.force-quantum 128
pw-metadata -n settings 0 clock.force-rate 96000
```

---

## Integration Examples

### 1. Discord/Gaming Setup

```bash
# Create virtual microphone for Discord
pactl load-module module-null-sink sink_name=ghostwave_input sink_properties=device.description="GhostWave_Input"
pactl load-module module-null-sink sink_name=ghostwave_output sink_properties=device.description="GhostWave_Output"

# Start GhostWave processor
./ghostwave --pipewire-module --profile gaming

# Configure Discord to use GhostWave virtual mic
# Input: GhostWave_Output (Monitor)
# Output: Your regular speakers/headphones
```

### 2. OBS Studio Integration

```bash
# Create OBS-specific audio routing
pw-cli create-object adapter {
    factory.name=support.null-audio-sink
    object.linger=true
    node.name=obs-ghostwave
    media.class=Audio/Sink
    audio.channels=2
}

# Start GhostWave for streaming
./ghostwave --pipewire-module --profile streaming --frames 256

# In OBS: Add Audio Input Capture source
# Device: obs-ghostwave (Monitor)
```

### 3. Professional Recording Setup

```bash
# Set up for Scarlett Solo 4th Gen
./ghostwave --doctor  # Verify Scarlett detection

# Start with optimized studio settings
./ghostwave --pipewire-module --profile studio --samplerate 192000 --frames 128

# Monitor real-time performance
./ghostwave --bench --profile studio --verbose
```

---

## Troubleshooting

### Common Issues

**1. High Latency**
```bash
# Check current quantum settings
pw-metadata -n settings

# Reduce quantum for lower latency
pw-metadata -n settings 0 clock.force-quantum 128

# Verify audio interface buffer size
cat /proc/asound/*/pcm*/sub*/hw_params
```

**2. Audio Dropouts (XRuns)**
```bash
# Increase quantum for stability
pw-metadata -n settings 0 clock.force-quantum 512

# Check system load
top -p $(pgrep pipewire)

# Monitor XRuns
./ghostwave --bench --verbose
```

**3. PipeWire Service Issues**
```bash
# Restart PipeWire services
systemctl --user restart pipewire pipewire-pulse wireplumber

# Check service status
systemctl --user status pipewire

# Debug with verbose logging
PIPEWIRE_DEBUG=3 pipewire &
```

**4. Module Loading Failures**
```bash
# Check module dependencies
ldd ./ghostwave

# Verify PipeWire API version
pkg-config --modversion libpipewire-0.3

# Test module loading
./ghostwave --pipewire-module --verbose
```

### Performance Debugging

**Monitor Audio Processing:**
```bash
# Real-time latency monitoring
./ghostwave --pipewire-module --bench --verbose

# PipeWire graph analysis
pw-top

# System-wide audio analysis
pw-cli info all | grep -E "(latency|quantum|rate)"
```

**Profile CPU Usage:**
```bash
# Profile GhostWave performance
perf record -g ./ghostwave --pipewire-module --bench
perf report

# Monitor context switches
perf stat -e context-switches ./ghostwave --pipewire-module
```

### Advanced Debugging

**PipeWire Graph Inspection:**
```bash
# Dump complete audio graph
pw-dump

# Monitor node creation/destruction
pw-mon

# Check specific node properties
pw-cli info <node-id>
```

**Audio Buffer Analysis:**
```rust
use ghostwave_core::pipewire::LatencyMonitor;

// Custom latency monitoring
let monitor = LatencyMonitor::new();

// In your audio callback
let start = std::time::Instant::now();
processor.process(&input, &mut output)?;
let latency = start.elapsed();

monitor.record_frame(latency);

// Detect problematic frames
if latency > std::time::Duration::from_millis(5) {
    eprintln!("High latency detected: {:?}", latency);
}
```

---

## Best Practices

### 1. System Configuration

- Use PipeWire 0.3.65+ for best compatibility
- Configure real-time limits properly
- Set appropriate CPU governor
- Disable unnecessary system services during audio work

### 2. Application Integration

- Start GhostWave before audio applications
- Use appropriate profiles for your use case
- Monitor performance regularly with `--bench`
- Configure automatic startup for production use

### 3. Troubleshooting Workflow

1. **Verify PipeWire is running**: `systemctl --user status pipewire`
2. **Check audio devices**: `pw-cli list-objects`
3. **Test basic routing**: `pw-play test.wav`
4. **Start GhostWave**: `./ghostwave --pipewire-module --verbose`
5. **Monitor performance**: `./ghostwave --bench`

---

This PipeWire integration guide ensures optimal performance and compatibility with modern Linux audio systems. For ALSA direct integration, see [ALSA.md](ALSA.md). For NVIDIA RTX setup, see [NVIDIA.md](NVIDIA.md).