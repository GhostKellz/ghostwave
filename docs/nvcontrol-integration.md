# NVControl Integration Guide

This guide covers how to integrate GhostWave with NVControl, an advanced NVIDIA GPU management tool for Linux, to leverage RTX Voice capabilities and GPU acceleration for audio processing.

## Overview

NVControl is "The Ultimate NVIDIA GPU Control Tool for Linux" that provides:
- GPU performance monitoring and control
- Container GPU passthrough support
- RTX Voice container integration
- Wayland-native design for modern Linux environments
- Digital vibrance and display color management

GhostWave can leverage NVControl's infrastructure for RTX Voice-like capabilities and GPU acceleration management.

## Integration Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NVControl     │────▶│    GhostWave     │────▶│  Clean Audio    │
│  (GPU Control)  │    │ (RTX Processing) │    │    Output       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │
        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│   GPU Resources │    │   Performance    │
│   Management    │    │   Monitoring     │
└─────────────────┘    └──────────────────┘
```

## Prerequisites

### System Requirements

1. **NVIDIA GPU with RTX Voice Support**
   ```bash
   # Check GPU compatibility
   nvidia-smi --query-gpu=name,compute_cap --format=csv

   # Minimum requirements:
   # - NVIDIA RTX 20 series or newer
   # - Compute Capability 7.5+
   # - 4GB+ VRAM
   ```

2. **NVIDIA Driver Installation**
   ```bash
   # Install NVIDIA drivers (Arch Linux example)
   sudo pacman -S nvidia nvidia-utils nvidia-settings

   # Verify installation
   nvidia-smi
   ```

3. **CUDA Runtime**
   ```bash
   # Install CUDA toolkit
   sudo pacman -S cuda cuda-tools

   # Add to PATH
   echo 'export PATH=/opt/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   ```

## Integration Methods

### Method 1: GPU Resource Coordination

NVControl manages GPU resources while GhostWave uses them for processing:

#### 1. Install and Configure NVControl

```bash
# Install NVControl
git clone https://github.com/ghostkellz/nvcontrol
cd nvcontrol
cargo build --release

# Start NVControl daemon
sudo ./target/release/nvcontrol daemon --enable-gpu-passthrough
```

#### 2. Configure GhostWave with RTX Support

```bash
# Build GhostWave with NVIDIA RTX support
cd ghostwave
cargo build --release --features nvidia-rtx

# Start with RTX acceleration enabled
./target/release/ghostwave --profile balanced --nvidia-rtx
```

#### 3. GPU Resource Management

```rust
// Example: Coordinate GPU usage between applications
use nvcontrol_api::GpuManager;

pub struct GpuCoordinator {
    nvcontrol: GpuManager,
    ghostwave_allocation: GpuAllocation,
}

impl GpuCoordinator {
    pub async fn allocate_for_ghostwave(&mut self) -> Result<()> {
        // Reserve GPU resources for audio processing
        self.ghostwave_allocation = self.nvcontrol.allocate_compute_units(
            ComputeUnits::new()
                .with_sm_count(16)  // 16 Streaming Multiprocessors
                .with_memory_mb(1024)  // 1GB VRAM
                .with_priority(Priority::High)
        ).await?;

        println!("✅ Allocated GPU resources for GhostWave RTX processing");
        Ok(())
    }

    pub async fn monitor_usage(&self) -> GpuMetrics {
        self.nvcontrol.get_current_usage(self.ghostwave_allocation.id).await
    }
}
```

### Method 2: Container-Based Integration

Use NVControl's container GPU passthrough for isolated audio processing:

#### 1. Create GhostWave Container

```dockerfile
# Dockerfile.ghostwave-rtx
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Install Rust and dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential pkg-config libasound2-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Copy GhostWave source
COPY . /ghostwave
WORKDIR /ghostwave

# Build with RTX support
RUN cargo build --release --features nvidia-rtx

# Expose audio processing service
EXPOSE 8080
CMD ["./target/release/ghostwave", "--ipc-server", "--bind", "0.0.0.0:8080"]
```

#### 2. NVControl Container Management

```bash
# Build container
docker build -f Dockerfile.ghostwave-rtx -t ghostwave-rtx:latest .

# Run with NVControl GPU passthrough
nvcontrol container run \
  --gpu-profile audio-processing \
  --memory-limit 2G \
  --priority high \
  --name ghostwave-rtx \
  ghostwave-rtx:latest
```

#### 3. Container Configuration

```toml
# ~/.config/nvcontrol/profiles/audio-processing.toml
[gpu]
sm_allocation = 20  # 20% of streaming multiprocessors
memory_limit = "2GB"
power_limit = 200   # Watts
priority = "high"

[monitoring]
enable_metrics = true
alert_on_overload = true
log_performance = true

[audio]
enable_rtx_voice = true
sample_rate = 48000
latency_target_ms = 10
```

### Method 3: Performance Optimization Integration

Coordinate performance settings between NVControl and GhostWave:

#### 1. Dynamic Performance Scaling

```rust
use nvcontrol_api::{GpuController, PowerProfile};

pub struct AdaptivePerformance {
    nvcontrol: GpuController,
    current_profile: PowerProfile,
}

impl AdaptivePerformance {
    pub async fn optimize_for_audio(&mut self, audio_load: f32) -> Result<()> {
        let target_profile = match audio_load {
            load if load < 0.3 => PowerProfile::Balanced,
            load if load < 0.7 => PowerProfile::Performance,
            _ => PowerProfile::Maximum,
        };

        if target_profile != self.current_profile {
            self.nvcontrol.set_power_profile(target_profile).await?;
            self.current_profile = target_profile;

            println!("🔄 GPU profile switched to {:?} for audio load {:.1}%",
                    target_profile, audio_load * 100.0);
        }

        Ok(())
    }
}
```

#### 2. Thermal Management

```rust
pub struct ThermalManager {
    nvcontrol: GpuController,
    temp_threshold: f32,
}

impl ThermalManager {
    pub async fn monitor_and_adjust(&mut self) -> Result<()> {
        let temp = self.nvcontrol.get_gpu_temperature().await?;

        if temp > self.temp_threshold {
            // Reduce processing intensity to prevent throttling
            self.reduce_audio_processing_load().await?;
            println!("🌡️  GPU temperature high ({:.1}°C), reducing audio processing load", temp);
        }

        Ok(())
    }

    async fn reduce_audio_processing_load(&self) -> Result<()> {
        // Signal GhostWave to reduce processing complexity
        let request = json!({
            "method": "set_param",
            "params": {
                "name": "processing_quality",
                "value": "reduced"
            }
        });

        // Send via IPC...
        Ok(())
    }
}
```

## Real-Time Coordination

### Shared Configuration Management

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct IntegratedConfig {
    pub nvcontrol: NVControlConfig,
    pub ghostwave: GhostWaveConfig,
    pub coordination: CoordinationConfig,
}

#[derive(Serialize, Deserialize)]
pub struct CoordinationConfig {
    pub auto_gpu_scaling: bool,
    pub thermal_protection: bool,
    pub power_efficiency_mode: bool,
    pub performance_monitoring: bool,
}

impl IntegratedConfig {
    pub fn studio_preset() -> Self {
        Self {
            nvcontrol: NVControlConfig {
                power_limit: 250,
                memory_clock_offset: 1000,
                gpu_clock_offset: 100,
                fan_curve: FanCurve::Quiet,
            },
            ghostwave: GhostWaveConfig {
                profile: ProcessingProfile::Studio,
                rtx_quality: RtxQuality::Maximum,
                latency_target_ms: 5,
            },
            coordination: CoordinationConfig {
                auto_gpu_scaling: true,
                thermal_protection: true,
                power_efficiency_mode: false,
                performance_monitoring: true,
            },
        }
    }

    pub fn streaming_preset() -> Self {
        Self {
            nvcontrol: NVControlConfig {
                power_limit: 200,
                memory_clock_offset: 500,
                gpu_clock_offset: 50,
                fan_curve: FanCurve::Balanced,
            },
            ghostwave: GhostWaveConfig {
                profile: ProcessingProfile::Streaming,
                rtx_quality: RtxQuality::Balanced,
                latency_target_ms: 10,
            },
            coordination: CoordinationConfig {
                auto_gpu_scaling: true,
                thermal_protection: true,
                power_efficiency_mode: true,
                performance_monitoring: true,
            },
        }
    }
}
```

### Performance Monitoring Dashboard

```rust
use eframe::egui;

pub struct PerformanceDashboard {
    gpu_usage: f32,
    gpu_memory_usage: f32,
    gpu_temperature: f32,
    audio_latency: f32,
    rtx_processing_time: f32,
}

impl PerformanceDashboard {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("GPU Audio Processing Status");

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label("GPU Usage");
                ui.add(egui::ProgressBar::new(self.gpu_usage / 100.0)
                    .text(format!("{:.1}%", self.gpu_usage)));

                ui.label("VRAM Usage");
                ui.add(egui::ProgressBar::new(self.gpu_memory_usage / 100.0)
                    .text(format!("{:.1}%", self.gpu_memory_usage)));

                ui.label("Temperature");
                let temp_color = if self.gpu_temperature > 80.0 {
                    egui::Color32::RED
                } else if self.gpu_temperature > 70.0 {
                    egui::Color32::YELLOW
                } else {
                    egui::Color32::GREEN
                };
                ui.colored_label(temp_color, format!("{:.1}°C", self.gpu_temperature));
            });

            ui.vertical(|ui| {
                ui.label("Audio Latency");
                let latency_color = if self.audio_latency > 20.0 {
                    egui::Color32::RED
                } else if self.audio_latency > 10.0 {
                    egui::Color32::YELLOW
                } else {
                    egui::Color32::GREEN
                };
                ui.colored_label(latency_color, format!("{:.1}ms", self.audio_latency));

                ui.label("RTX Processing");
                ui.colored_label(
                    egui::Color32::BLUE,
                    format!("{:.2}ms", self.rtx_processing_time)
                );
            });
        });
    }
}
```

## Command Line Integration

### Unified Control Commands

```bash
#!/bin/bash
# ghostwave-nvcontrol-ctl - Unified control script

case "$1" in
    "studio")
        echo "🎛️  Configuring for studio recording..."
        nvcontrol gpu set-power-limit 250
        nvcontrol gpu set-clocks --memory +1000 --core +100
        ghostwave --profile studio --nvidia-rtx --frames 64
        ;;

    "streaming")
        echo "📺 Configuring for streaming..."
        nvcontrol gpu set-power-limit 200
        nvcontrol gpu set-clocks --memory +500 --core +50
        ghostwave --profile streaming --nvidia-rtx --frames 128
        ;;

    "gaming")
        echo "🎮 Configuring for gaming..."
        nvcontrol gpu set-power-limit 300
        nvcontrol gpu set-clocks --memory +1200 --core +150
        ghostwave --profile balanced --nvidia-rtx --frames 256
        ;;

    "monitor")
        echo "📊 Performance monitoring..."
        watch -n 1 '
        echo "=== GPU Status ==="
        nvcontrol gpu status
        echo ""
        echo "=== GhostWave Status ==="
        curl -s http://localhost:8080/status | jq .
        '
        ;;

    *)
        echo "Usage: $0 {studio|streaming|gaming|monitor}"
        exit 1
        ;;
esac
```

## Advanced Features

### AI Model Management

Coordinate AI model loading between NVControl and GhostWave:

```rust
pub struct AIModelManager {
    nvcontrol: GpuController,
    loaded_models: HashMap<String, ModelHandle>,
}

impl AIModelManager {
    pub async fn load_rtx_voice_model(&mut self, quality: RtxQuality) -> Result<()> {
        let model_name = format!("rtx_voice_{:?}", quality);

        if !self.loaded_models.contains_key(&model_name) {
            // Ensure sufficient GPU memory
            let memory_required = self.get_model_memory_requirement(&model_name);
            self.nvcontrol.reserve_memory(memory_required).await?;

            // Load model
            let model_handle = self.load_model_to_gpu(&model_name).await?;
            self.loaded_models.insert(model_name, model_handle);

            println!("🤖 Loaded RTX Voice model: {:?}", quality);
        }

        Ok(())
    }

    pub async fn optimize_model_placement(&mut self) -> Result<()> {
        // Use NVControl to optimize GPU memory layout
        self.nvcontrol.optimize_memory_layout(
            &self.loaded_models.keys().collect::<Vec<_>>()
        ).await?;

        Ok(())
    }
}
```

### Multi-GPU Support

```rust
pub struct MultiGpuManager {
    primary_gpu: GpuId,
    secondary_gpu: Option<GpuId>,
    nvcontrol: GpuController,
}

impl MultiGpuManager {
    pub async fn balance_processing(&mut self, workload: AudioWorkload) -> Result<()> {
        match workload {
            AudioWorkload::Light => {
                // Use only primary GPU
                self.nvcontrol.disable_gpu(self.secondary_gpu).await?;
            }
            AudioWorkload::Heavy => {
                // Use both GPUs
                if let Some(secondary) = self.secondary_gpu {
                    self.nvcontrol.enable_gpu(secondary).await?;
                    self.distribute_processing().await?;
                }
            }
        }
        Ok(())
    }
}
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check NVIDIA driver installation
   nvidia-smi

   # Verify CUDA installation
   nvcc --version

   # Check GhostWave RTX feature compilation
   ghostwave --version | grep -i rtx
   ```

2. **High GPU Temperature**
   ```bash
   # Monitor temperatures
   nvcontrol gpu monitor --temp-alert 80

   # Adjust fan curves
   nvcontrol gpu set-fan-curve aggressive

   # Reduce processing quality temporarily
   ghostwave --set-param rtx_quality reduced
   ```

3. **Memory Allocation Errors**
   ```bash
   # Check VRAM usage
   nvidia-smi --query-gpu=memory.used,memory.total --format=csv

   # Free GPU memory
   nvcontrol gpu memory cleanup

   # Restart with reduced model size
   ghostwave --nvidia-rtx --rtx-model-size small
   ```

### Debug Commands

```bash
# Comprehensive system check
ghostwave doctor --include-gpu

# GPU-specific diagnostics
nvcontrol diagnose --audio-processing

# Real-time monitoring
nvcontrol monitor --audio-metrics &
ghostwave --bench --gpu-metrics

# Integration test
echo '{"method": "gpu_status", "id": 1}' | nc -U /tmp/ghostwave.sock
```

## Best Practices

### Power Management

1. **Use appropriate power limits for different scenarios**
   - Studio: Higher power for maximum quality
   - Streaming: Balanced power for efficiency
   - Idle: Minimum power to save energy

2. **Monitor thermal throttling**
   - Keep GPU temperature below 83°C
   - Use custom fan curves for audio work
   - Consider undervolting for lower temperatures

3. **Memory Management**
   - Preload frequently used models
   - Use model pooling for different qualities
   - Monitor VRAM fragmentation

### Performance Optimization

1. **Profile-Specific GPU Settings**
   ```toml
   [profiles.studio]
   gpu_power_limit = 250
   memory_clock_offset = 1000
   core_clock_offset = 100
   rtx_quality = "maximum"

   [profiles.streaming]
   gpu_power_limit = 200
   memory_clock_offset = 500
   core_clock_offset = 50
   rtx_quality = "balanced"
   ```

2. **Automatic Scaling**
   - Enable dynamic frequency scaling
   - Use load-based performance adjustment
   - Implement thermal protection

## Conclusion

The GhostWave-NVControl integration provides comprehensive GPU management for professional audio processing on Linux. This combination enables:

- Optimal RTX Voice performance through coordinated GPU management
- Thermal and power efficiency for long recording sessions
- Professional-grade audio processing with hardware acceleration
- Seamless container-based deployment for complex setups

For best results:
- Use appropriate GPU profiles for different scenarios
- Monitor performance metrics continuously
- Implement thermal protection for stability
- Leverage container isolation for complex workflows

This integration represents the cutting edge of Linux audio processing, combining advanced noise suppression with intelligent GPU management.