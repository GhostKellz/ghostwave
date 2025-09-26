# GhostWave Troubleshooting Guide

This guide covers common issues with real-time audio processing including XRuns, audio crackle, and clock drift.

## Table of Contents

1. [XRun Issues](#xrun-issues)
2. [Audio Crackle and Dropouts](#audio-crackle-and-dropouts)
3. [Clock Drift and Timing Issues](#clock-drift-and-timing-issues)
4. [High CPU Usage](#high-cpu-usage)
5. [Device Detection Problems](#device-detection-problems)
6. [PipeWire Integration Issues](#pipewire-integration-issues)
7. [Latency Problems](#latency-problems)
8. [Configuration Issues](#configuration-issues)
9. [Diagnostic Commands](#diagnostic-commands)

## XRun Issues

XRuns (buffer underruns/overruns) are the most common issue in real-time audio processing.

### Symptoms
- Audio dropouts or glitches
- Crackling sounds
- "XRUN detected" messages in logs
- Inconsistent audio processing

### Root Causes
1. **Buffer size too small** - Most common cause
2. **CPU scheduling issues** - Process not getting RT priority
3. **Hardware limitations** - USB audio interfaces on slow buses
4. **System overload** - Too many processes competing for CPU
5. **Driver issues** - Poor ALSA/kernel driver performance

### Solutions

#### 1. Increase Buffer Size
```bash
# Try larger buffer sizes
ghostwave --frames 256    # Default: 128
ghostwave --frames 512    # For problematic systems
ghostwave --frames 1024   # Maximum recommended
```

#### 2. Enable Real-time Priority
```bash
# Set RT capabilities (run once)
sudo setcap cap_sys_nice+ep /usr/local/bin/ghostwave

# Enable RT priority
ghostwave --realtime --profile studio
```

#### 3. Optimize System Settings
```bash
# Check CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Set to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU idle states (aggressive)
echo 1 | sudo tee /sys/devices/system/cpu/cpu*/cpuidle/state*/disable
```

#### 4. Use RT Kernel (Ubuntu/Debian)
```bash
sudo apt install linux-lowlatency
# Reboot and select lowlatency kernel

# Or for full RT kernel
sudo apt install linux-rt
```

#### 5. USB Audio Optimization
```bash
# Add to /etc/default/grub
GRUB_CMDLINE_LINUX="usbcore.autosuspend=-1"

# Update grub
sudo update-grub
sudo reboot
```

### Verification
```bash
# Check for XRuns in real-time
ghostwave --bench --verbose

# Monitor with doctor command
ghostwave --doctor
```

## Audio Crackle and Dropouts

### Symptoms
- Crackling or popping sounds
- Brief audio interruptions
- Static noise during processing

### Common Causes
1. **Sample rate mismatch** between hardware and software
2. **Bit depth conversion** artifacts
3. **Clock synchronization** issues
4. **Electromagnetic interference** (EMI)
5. **Overloaded noise processing**

### Solutions

#### 1. Match Sample Rates
```bash
# Check device capabilities
ghostwave --list-devices

# Set explicit sample rate
ghostwave --samplerate 48000  # Most common
ghostwave --samplerate 44100  # CD quality
ghostwave --samplerate 96000  # Studio profile
```

#### 2. Reduce Processing Load
```bash
# Use lighter profile
ghostwave --profile balanced   # Instead of streaming

# Adjust noise reduction
ghostwave --dry-run  # Test configuration first
```

#### 3. Check Hardware Connections
- Use shorter, higher-quality USB/audio cables
- Avoid USB hubs, connect directly to motherboard
- Keep audio cables away from power cables
- Use ferrite cores on cables if available

#### 4. Test with Different Devices
```bash
# Try different input device
ghostwave --input "Scarlett Solo"
ghostwave --input "USB Audio"

# Test with loopback
ghostwave --input "Monitor of Built-in Audio"
```

## Clock Drift and Timing Issues

### Symptoms
- Audio gradually going out of sync
- Periodic audio glitches
- Sample rate conversion artifacts
- "Clock drift detected" warnings

### Root Causes
1. **Hardware clock differences** between devices
2. **Temperature-related drift** in crystal oscillators
3. **USB timing jitter**
4. **PipeWire/JACK clock configuration**

### Solutions

#### 1. Use Single Clock Source
```bash
# Force specific sample rate
ghostwave --samplerate 48000 --frames 128

# Use hardware device as clock master
ghostwave --input "hw:Scarlett,0" --output "hw:Scarlett,0"
```

#### 2. PipeWire Clock Configuration
Add to `~/.config/pipewire/pipewire.conf.d/99-ghostwave.conf`:
```
context.properties = {
    default.clock.rate = 48000
    default.clock.quantum = 128
    default.clock.min-quantum = 32
    default.clock.max-quantum = 2048
    default.clock.quantum-limit = 8192
}
```

#### 3. Hardware Solutions
- Use audio interfaces with internal clocks
- Prefer professional interfaces (Focusrite, PreSonus)
- Keep devices at stable temperature
- Use powered USB hubs for multiple devices

### Verification
```bash
# Monitor clock status
pw-top  # PipeWire real-time monitoring

# Check device clock rates
cat /proc/asound/card*/stream*
```

## High CPU Usage

### Symptoms
- System becomes sluggish during audio processing
- High CPU usage in system monitor
- Fan noise from increased CPU load
- Other applications becoming unresponsive

### Optimization Strategies

#### 1. Profile Selection
```bash
# Use appropriate profile for use case
ghostwave --profile studio     # Minimal processing
ghostwave --profile balanced   # Moderate processing
ghostwave --profile streaming  # Maximum processing
```

#### 2. Enable Hardware Acceleration
```bash
# Check for RTX support
ghostwave --doctor

# Enable GPU acceleration (if available)
ghostwave --nvidia-rtx --profile studio
```

#### 3. CPU Affinity
```bash
# Bind to specific CPU cores
ghostwave --realtime --cpu-affinity 0,1
```

#### 4. System Optimization
```bash
# Close unnecessary applications
systemctl --user stop evolution-calendar-factory
systemctl --user stop evolution-addressbook-factory

# Disable desktop effects during recording
gsettings set org.gnome.desktop.interface enable-animations false
```

## Device Detection Problems

### Symptoms
- "No audio devices found" error
- Device not listed in `--list-devices`
- Wrong device selected automatically
- USB device not recognized

### Solutions

#### 1. Check Device Permissions
```bash
# Add user to audio group
sudo usermod -a -G audio $USER
# Logout and login again

# Check device permissions
ls -l /dev/snd/
```

#### 2. USB Audio Issues
```bash
# Check USB device detection
lsusb | grep -i audio

# Check ALSA device list
arecord -l
aplay -l

# Test device directly
arecord -D hw:1,0 -f S16_LE -r 48000 test.wav
```

#### 3. PipeWire Device Detection
```bash
# List PipeWire devices
pw-cli list-objects | grep -A5 -B5 audio

# Restart PipeWire services
systemctl --user restart pipewire
systemctl --user restart wireplumber
```

#### 4. Force Device Selection
```bash
# Use specific ALSA device
ghostwave --input "hw:Scarlett,0"

# Use PipeWire device name
ghostwave --input "Scarlett Solo USB"
```

## PipeWire Integration Issues

### Symptoms
- GhostWave node not appearing in audio routing
- Connection failures to other applications
- Audio routing loops
- No audio output despite processing

### Solutions

#### 1. Verify PipeWire Status
```bash
# Check PipeWire is running
systemctl --user status pipewire
systemctl --user status wireplumber

# Check GhostWave node
pw-cli list-objects | grep -i ghostwave
```

#### 2. Node Configuration
```bash
# Start with PipeWire integration
ghostwave --pipewire-module --verbose

# Check node properties
pw-cli info "GhostWave Clean"
```

#### 3. Connection Troubleshooting
```bash
# Use helvum for visual routing
sudo apt install helvum
helvum  # GUI connection manager

# Or command-line connections
pw-link "GhostWave Clean:output_FL" "Built-in Audio:playback_FL"
```

#### 4. Reset Audio Configuration
```bash
# Backup current configuration
cp -r ~/.config/pipewire ~/.config/pipewire.bak

# Reset to defaults
rm -rf ~/.config/pipewire
systemctl --user restart pipewire wireplumber
```

## Latency Problems

### Symptoms
- High input-to-output delay
- Lip sync issues in video calls
- Delayed monitoring feedback
- Sluggish audio response

### Measurement and Optimization

#### 1. Measure Current Latency
```bash
# Built-in latency measurement
ghostwave --bench

# External measurement with jack_iodelay (if JACK available)
# jack_iodelay
```

#### 2. Optimize for Lowest Latency
```bash
# Minimum settings
ghostwave --frames 32 --samplerate 48000 --realtime

# Professional setup
ghostwave --profile studio --frames 64 --realtime --cpu-affinity 0,1
```

#### 3. Hardware Considerations
- Use direct hardware monitoring when possible
- Prefer USB 2.0 over USB 3.0 for audio (counter-intuitive but often true)
- Use dedicated audio interfaces instead of built-in audio
- Consider PCIe audio cards for lowest latency

## Configuration Issues

### Symptoms
- Settings not persisting between sessions
- Config file not found
- Invalid configuration errors
- Default settings always loading

### Solutions

#### 1. Configuration File Location
```bash
# Check config file exists
ls -la ~/.config/ghostwave/config.toml

# Create default config
mkdir -p ~/.config/ghostwave
ghostwave --dry-run  # Validates and shows current config
```

#### 2. Configuration Validation
```bash
# Test configuration
ghostwave --dry-run --verbose

# Validate specific settings
ghostwave --config ~/.config/ghostwave/custom.toml --dry-run
```

#### 3. Reset Configuration
```bash
# Backup current config
cp ~/.config/ghostwave/config.toml ~/.config/ghostwave/config.toml.bak

# Reset to defaults
rm ~/.config/ghostwave/config.toml
ghostwave --profile balanced  # Creates new default config
```

## Diagnostic Commands

### Essential Troubleshooting Commands

#### 1. System Diagnostics
```bash
# Comprehensive system check
ghostwave --doctor

# Performance benchmark
ghostwave --bench --verbose

# Configuration validation
ghostwave --dry-run
```

#### 2. Device Information
```bash
# List all audio devices
ghostwave --list-devices

# List processing profiles
ghostwave --list-profiles

# Show version and build info
ghostwave --version
```

#### 3. Real-time Monitoring
```bash
# Verbose operation mode
ghostwave --verbose --profile balanced

# Monitor with external tools
htop      # System resources
pw-top    # PipeWire real-time monitor
```

#### 4. Log Analysis
```bash
# Check systemd logs
journalctl --user -u ghostwave -f

# Check for audio-related kernel messages
dmesg | grep -i audio
```

### Advanced Diagnostics

#### 1. USB Audio Debugging
```bash
# Enable USB audio debugging
echo 'module snd_usb_audio debug=1' | sudo tee -a /etc/modprobe.d/alsa-debug.conf

# Reload module
sudo modprobe -r snd_usb_audio
sudo modprobe snd_usb_audio

# Check debug output
dmesg | tail -50
```

#### 2. ALSA Debugging
```bash
# Test ALSA directly
arecord -D hw:1,0 -f S16_LE -r 48000 -c 2 -V stereo test.wav

# Check ALSA state
cat /proc/asound/cards
cat /proc/asound/version
```

#### 3. Performance Profiling
```bash
# Profile with perf
perf record -g ghostwave --profile balanced
perf report

# Monitor syscalls
strace -e trace=futex,nanosleep,clock_nanosleep ghostwave --bench
```

## Getting Help

If you continue experiencing issues after following this guide:

1. **Check the logs**:
   ```bash
   ghostwave --verbose 2>&1 | tee ghostwave-debug.log
   ```

2. **Run diagnostics**:
   ```bash
   ghostwave --doctor > system-report.txt
   ghostwave --bench >> system-report.txt
   ```

3. **Gather system information**:
   ```bash
   uname -a > system-info.txt
   lscpu >> system-info.txt
   lsusb >> system-info.txt
   cat /proc/version >> system-info.txt
   ```

4. **Create an issue** at: https://github.com/ghostkellz/ghostwave/issues

Include the generated log files and system information for faster troubleshooting.

## Performance Optimization Checklist

For optimal GhostWave performance:

- [ ] **RT kernel or low-latency kernel installed**
- [ ] **User in audio group** (`groups $USER` shows audio)
- [ ] **RT capabilities set** (`sudo setcap cap_sys_nice+ep /usr/local/bin/ghostwave`)
- [ ] **CPU governor set to performance**
- [ ] **Buffer size appropriate for system** (start with 128, increase if XRuns)
- [ ] **Sample rate matches hardware capabilities**
- [ ] **USB audio connected directly to motherboard** (not through hub)
- [ ] **Audio cables away from power/interference sources**
- [ ] **Unnecessary applications closed during recording**
- [ ] **PipeWire/JACK configuration optimized**
- [ ] **System temperature stable** (good ventilation)

Following this checklist should resolve 90%+ of common audio issues with GhostWave.