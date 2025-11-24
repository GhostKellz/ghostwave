# ALSA Direct Integration Guide

This guide covers GhostWave's direct ALSA integration for minimal latency audio processing. ALSA (Advanced Linux Sound Architecture) provides the lowest possible latency by bypassing higher-level audio servers.

## Table of Contents

- [Overview](#overview)
- [When to Use ALSA Direct](#when-to-use-alsa-direct)
- [Configuration](#configuration)
- [Usage](#usage)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

---

## Overview

### ALSA Direct Mode

ALSA direct mode provides:

- **Ultra-Low Latency**: Sub-3ms round-trip times possible
- **Hardware Control**: Direct access to audio interface features
- **Deterministic Performance**: Predictable timing without audio server overhead
- **Professional Audio**: Ideal for studio recording and live performance

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GhostWave     │    │      ALSA       │    │   Hardware      │
│   Application   │    │     Kernel      │    │   Interface     │
│                 │    │     Driver      │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │   Noise   │  │◄──►│  │   ALSA    │  │◄──►│  │  Audio    │  │
│  │ Processor │  │    │  │   PCM     │  │    │  │   ADC/    │  │
│  │           │  │    │  │   API     │  │    │  │   DAC     │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
│                 │    │                 │    │                 │
│  Lock-free      │    │  Kernel Audio   │    │  Scarlett Solo  │
│  Processing     │    │  Subsystem      │    │  XLR Interface  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**No Audio Server Overhead** - Direct kernel-to-application communication

---

## When to Use ALSA Direct

### Ideal Use Cases

**Professional Recording:**
- Studio multitrack recording
- Live performance with in-ear monitoring
- Podcast production with real-time monitoring
- Voiceover work requiring instant feedback

**Ultra-Low Latency Requirements:**
- Musicians using software instruments
- Live streaming with real-time interaction
- Gaming with competitive audio requirements
- Audio production with software effects

**Hardware-Specific Features:**
- Custom sample rate requirements (192kHz, etc.)
- Precise buffer size control
- Hardware monitoring capabilities
- Professional audio interface optimization

### When NOT to Use ALSA Direct

- **Desktop Audio**: Use PipeWire for general desktop use
- **Multi-Application**: ALSA direct typically locks the device
- **Complex Routing**: Use JACK for professional routing needs
- **System Integration**: PipeWire provides better system integration

---

## Configuration

### System Setup

**Install ALSA Development Libraries:**
```bash
# Arch Linux
sudo pacman -S alsa-lib alsa-utils alsa-plugins

# Ubuntu/Debian
sudo apt install libasound2-dev alsa-utils

# Fedora
sudo dnf install alsa-lib-devel alsa-utils
```

**Verify ALSA Installation:**
```bash
# List audio devices
aplay -l
arecord -l

# Test playback
speaker-test -t wav -c 2

# Check device capabilities
cat /proc/asound/cards
cat /proc/asound/devices
```

### Device Detection

Find your audio interface:

```bash
# List all ALSA devices
./ghostwave --doctor

# Specific device information
aplay -l | grep -i scarlett
arecord -l | grep -i scarlett

# Hardware parameters
cat /proc/asound/Gen/stream0  # Scarlett Solo 4th Gen
```

Example output for Scarlett Solo 4th Gen:
```
Focusrite Scarlett Solo 4th Gen at usb-0000:00:14.0-2, high speed : USB Audio

Playback:
  Status: Stop
  Interface 1
    Altset 1
    Format: S32_LE
    Channels: 2
    Endpoint: 1 OUT (ASYNC)
    Rates: 44100, 48000, 88200, 96000, 176400, 192000
```

### ALSA Configuration

Create optimized ALSA configuration in `~/.asoundrc`:

```ini
# High-performance configuration for Scarlett Solo 4th Gen
pcm.ghostwave_input {
    type hw
    card Gen           # Scarlett Solo 4th Gen
    device 0
    subdevice 0
    format S32_LE
    rate 192000        # Maximum sample rate
    channels 2

    # Buffer configuration for minimal latency
    period_time 1333   # ~256 frames at 192kHz
    periods 3          # Triple buffering
    buffer_time 4000   # ~768 frames total
}

pcm.ghostwave_output {
    type hw
    card Gen
    device 0
    subdevice 0
    format S32_LE
    rate 192000
    channels 2

    period_time 1333
    periods 3
    buffer_time 4000
}

# Duplex device for simultaneous input/output
pcm.ghostwave_duplex {
    type asym
    playback.pcm "ghostwave_output"
    capture.pcm "ghostwave_input"
}
```

### GhostWave ALSA Configuration

Configure GhostWave for optimal ALSA performance:

```rust
// Custom ALSA configuration
let alsa_config = AlsaConfig {
    device_name: "ghostwave_duplex".to_string(),
    sample_rate: 192000,
    buffer_size: 256,      // Ultra-low latency
    periods: 3,            // Triple buffering
    format: AlsaFormat::S32LE,
    channels: 2,
};
```

---

## Usage

### Basic ALSA Mode

Start GhostWave in ALSA direct mode:

```bash
# Basic ALSA operation
./ghostwave --alsa

# Specify device and parameters
./ghostwave --alsa --samplerate 192000 --frames 128

# Studio profile with ALSA
./ghostwave --alsa --profile studio --verbose

# Monitor performance
./ghostwave --alsa --bench --frames 64
```

### Advanced Configuration

**Ultra-Low Latency Setup:**
```bash
# Minimal latency configuration
./ghostwave --alsa --samplerate 96000 --frames 64 --profile studio

# Performance monitoring
./ghostwave --alsa --bench --verbose --frames 32
```

**Professional Recording:**
```bash
# High-quality recording setup
./ghostwave --alsa --samplerate 192000 --frames 256 --profile studio

# With real-time monitoring
./ghostwave --alsa --profile studio --verbose --bench
```

### Device-Specific Usage

**Scarlett Solo 4th Gen Optimization:**
```bash
# Auto-detected optimal settings
./ghostwave --doctor  # Verify detection
./ghostwave --alsa --profile studio

# Manual optimization
./ghostwave --alsa --samplerate 192000 --frames 128 --verbose
```

**Generic USB Audio:**
```bash
# Conservative settings for stability
./ghostwave --alsa --samplerate 48000 --frames 512

# Test optimal settings
./ghostwave --alsa --bench --samplerate 96000 --frames 256
```

---

## Performance Optimization

### 1. Real-Time Scheduling

Enable real-time priority for optimal ALSA performance:

```bash
# Set real-time capabilities
sudo setcap cap_sys_nice+ep ./ghostwave

# Configure system limits
sudo tee -a /etc/security/limits.conf << EOF
@audio - rtprio 95
@audio - memlock unlimited
@audio - nice -10
EOF

# Add user to audio group
sudo usermod -a -G audio $USER
```

### 2. Kernel and System Optimization

**Real-Time Kernel (Optional):**
```bash
# Arch Linux
sudo pacman -S linux-rt linux-rt-headers

# Ubuntu
sudo apt install linux-lowlatency
```

**System Tuning:**
```bash
# CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable audio power management
echo 0 | sudo tee /sys/module/snd_hda_intel/parameters/power_save

# Increase audio buffer limits
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'kernel.sched_rt_runtime_us=-1' | sudo tee -a /etc/sysctl.conf
```

### 3. ALSA Driver Optimization

**USB Audio Optimization:**
```bash
# Reduce USB audio buffering
echo 'options snd-usb-audio nrpacks=1' | sudo tee -a /etc/modprobe.d/alsa-base.conf

# Reload ALSA modules
sudo modprobe -r snd-usb-audio
sudo modprobe snd-usb-audio
```

**Hardware-Specific Tuning:**
```bash
# For Focusrite interfaces
echo 'options snd-usb-audio vid=0x1235 pid=0x8211 device_setup=1' | sudo tee -a /etc/modprobe.d/focusrite.conf
```

### 4. Application-Level Optimization

```rust
use ghostwave_core::alsa::AlsaOptimizer;

struct OptimizedAlsaProcessor {
    module: AlsaModule,
    optimizer: AlsaOptimizer,
}

impl OptimizedAlsaProcessor {
    pub fn new(config: Config) -> anyhow::Result<Self> {
        // Pin to high-performance CPU cores
        let mut cpu_set = nix::sched::CpuSet::new();
        cpu_set.set(2)?; // Use core 2
        cpu_set.set(3)?; // Use core 3
        nix::sched::sched_setaffinity(nix::unistd::Pid::this(), &cpu_set)?;

        // Set real-time priority
        let rt_param = libc::sched_param { sched_priority: 80 };
        unsafe {
            libc::sched_setscheduler(0, libc::SCHED_FIFO, &rt_param);
        }

        let module = AlsaModule::new(config)?;
        let optimizer = AlsaOptimizer::new();

        Ok(Self { module, optimizer })
    }

    pub fn process_with_optimization(&mut self, input: &[f32], output: &mut [f32]) -> anyhow::Result<()> {
        // Pre-flight checks
        self.optimizer.check_buffer_underrun()?;
        self.optimizer.monitor_latency_drift()?;

        // Process audio with timing
        let start = std::time::Instant::now();
        self.module.process_audio_block(input, output)?;
        let processing_time = start.elapsed();

        // Record performance metrics
        self.optimizer.record_processing_time(processing_time);

        // Adaptive optimization
        if processing_time > std::time::Duration::from_micros(1000) {
            self.optimizer.suggest_buffer_increase();
        }

        Ok(())
    }
}
```

---

## Integration Examples

### 1. Professional Recording Setup

```rust
use ghostwave_core::alsa::{AlsaModule, AlsaConfig};

async fn setup_recording_session() -> anyhow::Result<()> {
    // Configure for Scarlett Solo 4th Gen
    let alsa_config = AlsaConfig {
        device_name: "hw:Gen,0".to_string(),
        sample_rate: 192000,
        buffer_size: 128,      // ~0.67ms latency
        periods: 2,            // Double buffering for lowest latency
        format: AlsaFormat::S32LE,
        channels: 2,
    };

    let config = Config::load("studio")?
        .with_alsa_config(alsa_config);

    let mut alsa_module = AlsaModule::new(config)?;

    // Start real-time processing
    alsa_module.start_audio_processing().await?;

    // Monitor for the session
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

        let stats = alsa_module.get_performance_stats();
        if stats.xrun_count > 0 {
            eprintln!("Warning: {} XRuns detected", stats.xrun_count);
        }

        println!("Latency: {:.2}ms, CPU: {:.1}%",
                 stats.avg_latency_ms, stats.cpu_usage);
    }
}
```

### 2. Live Performance Monitoring

```rust
struct LiveMonitor {
    alsa: AlsaModule,
    monitoring_enabled: bool,
}

impl LiveMonitor {
    pub fn new() -> anyhow::Result<Self> {
        let config = Config::load("studio")?;
        let alsa = AlsaModule::new(config)?;

        Ok(Self {
            alsa,
            monitoring_enabled: true,
        })
    }

    pub async fn run_live_session(&mut self) -> anyhow::Result<()> {
        // Enable hardware monitoring if available
        self.alsa.enable_hardware_monitoring()?;

        // Process with minimal latency
        let mut input_buffer = vec![0.0f32; 64];   // 64 samples = ~0.33ms at 192kHz
        let mut output_buffer = vec![0.0f32; 64];

        loop {
            // Capture audio
            self.alsa.read_audio(&mut input_buffer)?;

            // Process with GhostWave
            self.alsa.process_audio_block(&input_buffer, &mut output_buffer)?;

            // Output processed audio
            self.alsa.write_audio(&output_buffer)?;

            // Check for problems
            if self.alsa.detect_xrun() {
                eprintln!("XRun detected - consider increasing buffer size");
            }
        }
    }
}
```

### 3. Automated Device Configuration

```bash
#!/bin/bash
# auto-setup-alsa.sh - Automatic ALSA configuration for GhostWave

echo "🔍 Detecting audio interfaces..."

# Detect Scarlett Solo 4th Gen
if aplay -l | grep -qi "scarlett solo 4th gen"; then
    echo "✅ Scarlett Solo 4th Gen detected"
    DEVICE="hw:Gen,0"
    SAMPLE_RATE=192000
    BUFFER_SIZE=128
    PROFILE="studio"
else
    echo "ℹ️  Using generic USB audio setup"
    DEVICE="hw:0,0"
    SAMPLE_RATE=48000
    BUFFER_SIZE=256
    PROFILE="balanced"
fi

echo "🚀 Starting GhostWave with optimal settings..."
echo "   Device: $DEVICE"
echo "   Sample Rate: $SAMPLE_RATE Hz"
echo "   Buffer Size: $BUFFER_SIZE frames"
echo "   Profile: $PROFILE"

./ghostwave --alsa \
    --device "$DEVICE" \
    --samplerate "$SAMPLE_RATE" \
    --frames "$BUFFER_SIZE" \
    --profile "$PROFILE" \
    --verbose
```

---

## Troubleshooting

### Common Issues

**1. Device Busy/Access Denied**
```bash
# Check what's using the device
sudo fuser -v /dev/snd/*

# Kill competing processes
sudo pkill pulseaudio
sudo pkill pipewire

# Restart audio services after testing
systemctl --user start pipewire
```

**2. High Latency Despite Small Buffers**
```bash
# Check actual hardware buffer size
cat /proc/asound/Gen/pcm0p/sub0/hw_params

# Verify real-time scheduling is active
ps -eo pid,cls,rtprio,comm | grep ghostwave

# Monitor system performance
./ghostwave --alsa --bench --verbose
```

**3. Audio Dropouts (XRuns)**
```bash
# Increase buffer size temporarily
./ghostwave --alsa --frames 512 --verbose

# Check system load
top -p $(pgrep ghostwave)

# Verify USB bandwidth (for USB interfaces)
lsusb -t
```

**4. Sample Rate Issues**
```bash
# Check supported rates
cat /proc/asound/Gen/stream0

# Force specific rate
./ghostwave --alsa --samplerate 48000 --verbose

# Test with multiple rates
for rate in 44100 48000 96000 192000; do
    echo "Testing $rate Hz..."
    ./ghostwave --alsa --samplerate $rate --frames 256 --bench | head -5
done
```

### Performance Debugging

**Monitor Audio Performance:**
```bash
# Real-time latency monitoring
./ghostwave --alsa --bench --verbose --frames 64

# System performance during audio
htop -p $(pgrep ghostwave)

# Detailed timing analysis
perf record -g ./ghostwave --alsa --bench
perf report
```

**ALSA Debugging:**
```bash
# Enable ALSA debugging
export ALSA_DEBUG=1
./ghostwave --alsa --verbose

# Check kernel audio messages
dmesg | grep -i audio

# Monitor USB audio specifically
usbmon | grep -i audio
```

### Advanced Debugging

**Buffer Analysis:**
```rust
use ghostwave_core::alsa::AlsaDebugger;

let debugger = AlsaDebugger::new();

// Monitor buffer fill levels
debugger.monitor_buffer_levels(|level| {
    if level < 0.1 {
        eprintln!("Warning: Buffer underrun imminent ({:.1}%)", level * 100.0);
    }
});

// Analyze timing consistency
debugger.analyze_callback_timing(|jitter| {
    if jitter > std::time::Duration::from_micros(100) {
        eprintln!("High timing jitter detected: {:?}", jitter);
    }
});
```

**Hardware Capability Testing:**
```bash
# Test maximum performance
for frames in 32 64 128 256 512; do
    echo "Testing $frames frames..."
    timeout 10s ./ghostwave --alsa --frames $frames --bench --samplerate 192000
    echo "---"
done

# USB bandwidth test
arecord -D hw:Gen,0 -f S32_LE -r 192000 -c 2 --test-position /dev/null
```

---

## Best Practices

### 1. Hardware Setup

- Use high-quality USB cables for audio interfaces
- Connect USB audio interfaces to dedicated USB controllers
- Avoid USB hubs for professional audio interfaces
- Use powered USB hubs if hubs are necessary

### 2. System Configuration

- Stop unnecessary audio services during critical work
- Use real-time kernel for ultimate performance
- Configure proper user permissions for audio devices
- Monitor system resources during audio work

### 3. Application Integration

- Start GhostWave before other audio applications
- Use conservative buffer sizes initially, then optimize
- Monitor XRun rates and adjust accordingly
- Implement proper error handling for device disconnections

### 4. Testing Workflow

1. **Verify Hardware**: `aplay -l`, `arecord -l`
2. **Test Basic ALSA**: `speaker-test`, `arecord`
3. **Check GhostWave Detection**: `./ghostwave --doctor`
4. **Start with Safe Settings**: `./ghostwave --alsa --frames 512`
5. **Optimize Gradually**: Reduce buffer size while monitoring XRuns
6. **Performance Test**: `./ghostwave --alsa --bench`

---

This ALSA integration guide provides comprehensive coverage for achieving minimal latency audio processing. For PipeWire integration, see [PIPEWIRE.md](PIPEWIRE.md). For NVIDIA RTX setup, see [NVIDIA.md](NVIDIA.md).