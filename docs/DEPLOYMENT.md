# GhostWave Deployment Guide

Complete guide for deploying GhostWave in production environments.

## Table of Contents

- [System Requirements](#system-requirements)
- [Pre-Installation Checklist](#pre-installation-checklist)
- [Installation Methods](#installation-methods)
- [SystemD Service Setup](#systemd-service-setup)
- [PipeWire Configuration](#pipewire-configuration)
- [NVIDIA Driver Setup](#nvidia-driver-setup)
- [Security Hardening](#security-hardening)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Upgrading](#upgrading)
- [Rollback Procedures](#rollback-procedures)

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Linux kernel 5.4+ (6.x recommended) |
| Audio Server | PipeWire 0.3.50+ or ALSA |
| RAM | 2 GB available |
| CPU | x86_64 with SSE4.2 support |
| Storage | 100 MB for binaries + logs |

### Recommended Requirements

| Component | Recommendation |
|-----------|----------------|
| OS | Arch Linux, Fedora 39+, Ubuntu 24.04+ |
| Audio Server | PipeWire 1.0+ with WirePlumber |
| RAM | 4 GB+ available |
| CPU | 4+ cores, 3.0 GHz+ |
| GPU | NVIDIA RTX 2060+ (for RTX acceleration) |

### NVIDIA GPU Requirements (Optional)

| Feature | Minimum GPU | Driver Version |
|---------|-------------|----------------|
| Basic RTX acceleration | RTX 2060 | 535.x+ |
| FP16 acceleration | RTX 3060 | 545.x+ |
| FP4 Tensor Cores | RTX 5090 | 560.x+ |

---

## Pre-Installation Checklist

### 1. Verify Audio Server

```bash
# Check PipeWire is running
systemctl --user status pipewire

# Verify PipeWire version
pw-cli --version

# Check for WirePlumber (recommended)
systemctl --user status wireplumber
```

### 2. Verify Audio Group Membership

```bash
# Check current groups
groups

# Add user to audio group if needed
sudo usermod -aG audio $USER

# Logout/login required after group change
```

### 3. Check Real-Time Limits

```bash
# View current limits
ulimit -r

# Should show non-zero for real-time priority
# If 0, configure /etc/security/limits.conf
```

**Configure real-time limits** (`/etc/security/limits.conf`):

```
@audio   -  rtprio     95
@audio   -  memlock    unlimited
@audio   -  nice       -20
```

### 4. NVIDIA Setup (Optional)

```bash
# Verify NVIDIA driver
nvidia-smi

# Check CUDA availability
nvcc --version  # Optional, only for building with RTX

# Verify compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

---

## Installation Methods

### Method 1: Pre-Built Binary (Recommended)

```bash
# Download latest release
curl -LO https://github.com/ghostkellz/ghostwave/releases/latest/download/ghostwave-linux-x86_64.tar.gz

# Extract
tar xzf ghostwave-linux-x86_64.tar.gz

# Install to user directory
mkdir -p ~/.local/bin
mv ghostwave ~/.local/bin/

# Verify installation
ghostwave --version
ghostwave --doctor
```

### Method 2: Build from Source

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install build dependencies (Arch Linux)
sudo pacman -S pipewire-devel alsa-lib jack2

# Install build dependencies (Ubuntu/Debian)
sudo apt install libpipewire-0.3-dev libasound2-dev libjack-jackd2-dev

# Clone repository
git clone https://github.com/ghostkellz/ghostwave.git
cd ghostwave

# Build (CPU-only)
cargo build --release

# Build (with RTX acceleration - requires CUDA toolkit)
cargo build --release --features nvidia-rtx

# Install
cp target/release/ghostwave ~/.local/bin/
```

### Method 3: Package Manager

**Arch Linux (AUR)**:
```bash
# Using yay
yay -S ghostwave

# Or with paru
paru -S ghostwave
```

**Fedora (COPR)**:
```bash
sudo dnf copr enable ghostkellz/ghostwave
sudo dnf install ghostwave
```

---

## SystemD Service Setup

### User Service (Recommended)

Install the user service for per-user audio processing:

```bash
# Create service directory
mkdir -p ~/.config/systemd/user/

# Copy service file
cp /path/to/ghostwave/systemd/ghostwave.user.service ~/.config/systemd/user/ghostwave.service

# Reload systemd
systemctl --user daemon-reload

# Enable and start
systemctl --user enable ghostwave.service
systemctl --user start ghostwave.service

# Check status
systemctl --user status ghostwave.service
```

### System Service (Multi-User)

For shared systems where multiple users need GhostWave:

```bash
# Install system-wide binary
sudo cp ~/.local/bin/ghostwave /usr/bin/

# Copy system service
sudo cp /path/to/ghostwave/systemd/ghostwave.service /etc/systemd/system/ghostwave@.service

# Enable for specific user
sudo systemctl enable ghostwave@username.service
sudo systemctl start ghostwave@username.service
```

### Service Configuration

Edit the service file to customize:

```ini
[Service]
# Change profile (studio, balanced, streaming)
ExecStart=/usr/bin/ghostwave --pipewire-module --profile studio

# Enable verbose logging
Environment="GHOSTWAVE_LOG_LEVEL=debug"

# Custom config location
Environment="GHOSTWAVE_CONFIG=/etc/ghostwave/config.toml"
```

---

## PipeWire Configuration

### Optimal PipeWire Settings

Create `/etc/pipewire/pipewire.conf.d/ghostwave.conf`:

```
context.properties = {
    # Lower quantum for reduced latency
    default.clock.quantum = 256
    default.clock.min-quantum = 64
    default.clock.max-quantum = 1024

    # Match sample rate to your interface
    default.clock.rate = 48000
    default.clock.allowed-rates = [ 44100 48000 96000 192000 ]
}
```

### WirePlumber Integration

Create `~/.config/wireplumber/main.lua.d/51-ghostwave.lua`:

```lua
-- Auto-connect GhostWave to default sink
rule = {
  matches = {
    {
      { "node.name", "matches", "ghostwave*" },
    },
  },
  apply_properties = {
    ["node.autoconnect"] = true,
    ["priority.session"] = 1000,
  },
}

table.insert(alsa_monitor.rules, rule)
```

### Verify Audio Routing

```bash
# List PipeWire nodes
pw-cli list-objects Node

# Check GhostWave node
pw-cli info $(pw-cli list-objects Node | grep ghostwave | head -1 | awk '{print $2}')

# Monitor links
pw-link -l
```

---

## NVIDIA Driver Setup

### Driver Installation

**Arch Linux**:
```bash
# Open source drivers (recommended for RTX 20-50 series)
sudo pacman -S nvidia-open nvidia-utils

# Proprietary drivers
sudo pacman -S nvidia nvidia-utils
```

**Ubuntu/Debian**:
```bash
# Add NVIDIA repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install driver
sudo apt install nvidia-driver-560
```

### CUDA Runtime (Optional, for RTX features)

```bash
# Arch Linux
sudo pacman -S cuda

# Ubuntu
sudo apt install nvidia-cuda-toolkit
```

### Verify GPU Detection

```bash
# Run GhostWave doctor
ghostwave --doctor

# Expected output includes:
# ✓ NVIDIA driver loaded
# ✓ CUDA runtime available
# ✓ GPU: NVIDIA GeForce RTX XXXX
# ✓ Compute capability: X.X
```

---

## Security Hardening

### File Permissions

```bash
# Restrict binary permissions
chmod 755 ~/.local/bin/ghostwave

# Protect config files
chmod 600 ~/.config/ghostwave/config.toml
```

### SystemD Hardening

The provided service files include security settings:

```ini
[Service]
# Prevent privilege escalation
NoNewPrivileges=yes

# Read-only system
ProtectSystem=strict
ProtectHome=read-only

# Isolation
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
LockPersonality=yes
```

### Network Isolation

GhostWave only needs local IPC:

```ini
[Service]
# Restrict network access (if not using remote IPC)
RestrictAddressFamilies=AF_UNIX
PrivateNetwork=yes
```

### Audit Logging

Enable audit logging for security-sensitive deployments:

```bash
# Create audit rule
sudo auditctl -w /usr/bin/ghostwave -p x -k ghostwave_exec
```

---

## Monitoring & Maintenance

### Health Checks

```bash
# Quick health check
ghostwave --doctor

# Performance benchmark
ghostwave --bench --profile studio

# Check service status
systemctl --user status ghostwave.service
```

### Log Management

```bash
# View service logs
journalctl --user -u ghostwave.service -f

# Set log rotation (logrotate.d/ghostwave)
/var/log/ghostwave/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

### Metrics Collection

GhostWave exposes metrics via IPC:

```bash
# Query metrics via IPC
echo '{"jsonrpc":"2.0","method":"get_metrics","id":1}' | \
  socat - UNIX-CONNECT:/run/user/$(id -u)/ghostwave.sock
```

**Available Metrics**:
- `latency_ms`: Current processing latency
- `cpu_usage_percent`: CPU utilization
- `gpu_usage_percent`: GPU utilization (if RTX enabled)
- `xrun_count`: Audio dropouts since start
- `frames_processed`: Total frames processed

### Performance Monitoring

```bash
# Monitor real-time CPU usage
watch -n 1 "ps -p $(pgrep ghostwave) -o %cpu,%mem,ni,cls"

# Monitor GPU usage (NVIDIA)
watch -n 1 nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv

# Monitor audio latency
pw-top
```

---

## Upgrading

### Backup Before Upgrade

```bash
# Backup configuration
cp -r ~/.config/ghostwave ~/.config/ghostwave.backup.$(date +%Y%m%d)

# Note current version
ghostwave --version > ~/.config/ghostwave.backup.$(date +%Y%m%d)/version.txt
```

### Standard Upgrade Process

```bash
# Stop service
systemctl --user stop ghostwave.service

# Download new version
curl -LO https://github.com/ghostkellz/ghostwave/releases/latest/download/ghostwave-linux-x86_64.tar.gz

# Backup old binary
mv ~/.local/bin/ghostwave ~/.local/bin/ghostwave.old

# Install new version
tar xzf ghostwave-linux-x86_64.tar.gz
mv ghostwave ~/.local/bin/

# Verify new version
ghostwave --version
ghostwave --doctor

# Restart service
systemctl --user start ghostwave.service

# Verify operation
systemctl --user status ghostwave.service
```

### Source Upgrade

```bash
cd /path/to/ghostwave
git pull origin main
cargo build --release
systemctl --user stop ghostwave.service
cp target/release/ghostwave ~/.local/bin/
systemctl --user start ghostwave.service
```

---

## Rollback Procedures

### Quick Rollback

```bash
# Stop current version
systemctl --user stop ghostwave.service

# Restore previous binary
mv ~/.local/bin/ghostwave.old ~/.local/bin/ghostwave

# Restart service
systemctl --user start ghostwave.service
```

### Full Rollback

```bash
# Stop service
systemctl --user stop ghostwave.service

# Restore configuration
rm -rf ~/.config/ghostwave
cp -r ~/.config/ghostwave.backup.YYYYMMDD ~/.config/ghostwave

# Restore binary
mv ~/.local/bin/ghostwave.old ~/.local/bin/ghostwave

# Restart
systemctl --user start ghostwave.service
```

### Emergency Recovery

If GhostWave won't start:

```bash
# Disable service
systemctl --user disable ghostwave.service

# Remove problematic config
mv ~/.config/ghostwave ~/.config/ghostwave.broken

# Run with defaults
ghostwave --profile balanced --verbose

# If successful, recreate config
ghostwave --init-config
```

---

## Production Checklist

Use this checklist before deploying to production:

- [ ] System meets minimum requirements
- [ ] User is member of `audio` group
- [ ] Real-time limits configured in `limits.conf`
- [ ] PipeWire running and configured
- [ ] NVIDIA drivers installed (if using RTX)
- [ ] `ghostwave --doctor` passes all checks
- [ ] SystemD service installed and enabled
- [ ] Log rotation configured
- [ ] Backup procedures documented
- [ ] Monitoring/alerting configured
- [ ] Rollback procedure tested

---

## Troubleshooting Deployment Issues

### Service Won't Start

```bash
# Check detailed logs
journalctl --user -u ghostwave.service --no-pager -n 50

# Common issues:
# - Missing audio group membership
# - PipeWire not running
# - Invalid config file
# - Binary not found in PATH
```

### Permission Denied

```bash
# Check file permissions
ls -la ~/.local/bin/ghostwave

# Check group membership
groups

# Check real-time limits
ulimit -a
```

### GPU Not Detected

```bash
# Run diagnostics
ghostwave --doctor

# Check NVIDIA driver
nvidia-smi

# Verify CUDA library path
ldconfig -p | grep cuda
```

For more troubleshooting, see [troubleshooting.md](troubleshooting.md).

---

**Last Updated**: December 2025 (v0.3.0)
