# NVIDIA RTX Integration Guide

This guide covers GhostWave's NVIDIA RTX GPU acceleration for AI-powered noise suppression. RTX acceleration provides superior denoising quality with minimal CPU impact.

**⭐ NEW: RTX 50 Series (Blackwell) Support** - See [RTX 5090 Optimizations](RTX_5090_OPTIMIZATIONS.md) for Blackwell-specific features including FP4 Tensor Core acceleration.

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Driver Installation](#driver-installation)
- [CUDA Setup](#cuda-setup)
- [RTX Integration](#rtx-integration)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

---

## Overview

### RTX-Accelerated Noise Suppression

GhostWave leverages NVIDIA's RTX tensor cores for real-time AI noise suppression:

- **AI-Powered Denoising**: Superior quality compared to traditional filters
- **Real-Time Processing**: Sub-millisecond GPU processing times
- **CPU Offloading**: Frees CPU resources for other tasks
- **Adaptive Models**: Dynamic adjustment to different noise types
- **Professional Quality**: Broadcast-grade noise suppression

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GhostWave Application                    │
├─────────────────────────────────────────────────────────────┤
│  Audio Pipeline                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Input     │───►│     RTX     │───►│   Output    │     │
│  │   Buffer    │    │ Processor   │    │   Buffer    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                            │                               │
├────────────────────────────┼───────────────────────────────┤
│  CUDA Runtime             │                               │
│  ┌─────────────┐    ┌──────▼──────┐    ┌─────────────┐     │
│  │   Memory    │    │   Tensor    │    │    AI       │     │
│  │  Management │    │   Cores     │    │   Models    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
├─────────────────────────────────────────────────────────────┤
│                NVIDIA Driver Stack                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Kernel    │    │     GPU     │    │   Memory    │     │
│  │   Driver    │    │  Scheduler  │    │  Controller │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │     RTX GPU           │
                    │  ┌─────────────────┐  │
                    │  │  Tensor Cores   │  │
                    │  │  (AI Accel)     │  │
                    │  └─────────────────┘  │
                    │  ┌─────────────────┐  │
                    │  │    CUDA Cores   │  │
                    │  │   (General)     │  │
                    │  └─────────────────┘  │
                    │  ┌─────────────────┐  │
                    │  │   GPU Memory    │  │
                    │  │    (VRAM)       │  │
                    │  └─────────────────┘  │
                    └───────────────────────┘
```

---

## Hardware Requirements

### Supported GPUs

**RTX 20 Series (Minimum):**
- RTX 2060, 2060 Super
- RTX 2070, 2070 Super
- RTX 2080, 2080 Super, 2080 Ti

**RTX 30 Series (Recommended):**
- RTX 3060, 3060 Ti
- RTX 3070, 3070 Ti
- RTX 3080, 3080 Ti
- RTX 3090, 3090 Ti

**RTX 40 Series (High-End):**
- RTX 4060, 4060 Ti
- RTX 4070, 4070 Ti, 4070 Super
- RTX 4080, 4080 Super
- RTX 4090

**RTX 50 Series (Elite) - NEW:**
- RTX 5060, 5060 Ti
- RTX 5070, 5070 Ti
- RTX 5080
- **RTX 5090 (Best Performance)**
- **ASUS ROG Astral RTX 5090 (Recommended)**

### Compute Capability Requirements

```bash
# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# GhostWave requirements:
# Minimum: Compute 7.5 (RTX 2060+)
# Recommended: Compute 8.6 (RTX 3060+)
# Elite: Compute 10.0 (RTX 5090) - Blackwell with FP4 Tensor Cores
# Optimal: Compute 8.9 (RTX 4060+)
```

### Memory Requirements

**VRAM Usage by Model:**
- **Basic Model**: 2GB VRAM minimum
- **Enhanced Model**: 4GB VRAM recommended
- **Professional Model**: 8GB VRAM optimal

**System Memory:**
- 16GB RAM minimum (32GB recommended for professional use)
- Fast system memory (DDR4-3200+ or DDR5)
- NVMe SSD for model loading

---

## Driver Installation

### NVIDIA Open Kernel Modules (Recommended)

**Arch Linux:**
```bash
# Install open kernel modules (580+ for RTX 50 series)
sudo pacman -S nvidia-open nvidia-utils

# Verify installation and driver version
modinfo nvidia | grep -i version
nvidia-smi  # Should show 580.105.08+ for RTX 5090

# Enable persistence mode
sudo nvidia-persistenced --persistence-mode

# For RTX 5090: Verify compute capability 10.0
nvidia-smi --query-gpu=compute_cap --format=csv
```

**Ubuntu 22.04+:**
```bash
# Add NVIDIA repository
sudo apt update
sudo apt install nvidia-driver-XXX-open  # Replace XXX with latest version

# Alternative: Use graphics PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-XXX-open
```

**Fedora 36+:**
```bash
# Enable RPM Fusion repositories
sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

# Install open drivers
sudo dnf install nvidia-driver-cuda
```

### Proprietary Drivers (Alternative)

If open drivers are unavailable:

```bash
# Arch Linux
sudo pacman -S nvidia nvidia-utils

# Ubuntu
sudo apt install nvidia-driver-XXX

# Fedora
sudo dnf install akmod-nvidia
```

### Driver Verification

```bash
# Check driver version
nvidia-smi

# Verify CUDA capability
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Test GPU accessibility
./ghostwave --doctor
```

Expected output:
```
🩺 GhostWave System Diagnostics
✅ NVIDIA kernel modules loaded
✅ NVIDIA DRM module loaded (good for Wayland)
✅ CUDA runtime library found
🚀 RTX Acceleration: Available
   Compute: 8.6
   Memory: 12.0 GB
   RTX Voice: Supported
```

---

## CUDA Setup

### CUDA Toolkit Installation

**Arch Linux:**
```bash
# Install CUDA toolkit
sudo pacman -S cuda cuda-tools

# Add to PATH
echo 'export PATH=/opt/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Ubuntu:**
```bash
# Install from NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Fedora:**
```bash
# Install CUDA from NVIDIA
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora37/x86_64/cuda-fedora37.repo
sudo dnf install cuda-toolkit

# Configure environment
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### CUDA Verification

```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Test CUDA installation
cuda-install-samples-11.8.sh ~
cd ~/NVIDIA_CUDA-11.8_Samples/1_Utilities/deviceQuery
make
./deviceQuery
```

### cuDNN Installation (Optional)

For enhanced AI model performance:

```bash
# Download cuDNN from NVIDIA Developer website
# https://developer.nvidia.com/cudnn

# Extract and install
tar -xzf cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.9.2.26_cuda12-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp cudnn-linux-x86_64-8.9.2.26_cuda12-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

---

## RTX Integration

### Building with RTX Support

**Compile GhostWave with RTX acceleration:**
```bash
# Build with NVIDIA RTX features
cargo build --features "nvidia-rtx,pipewire-backend" --release

# Verify RTX compilation
./ghostwave --doctor | grep -i rtx
```

### RTX Configuration

**Enable RTX acceleration in configuration:**
```rust
use ghostwave_core::{Config, RtxAccelerator};

// Load configuration with RTX
let mut config = Config::load("studio")?;

// Verify RTX availability
let rtx = RtxAccelerator::new()?;
if rtx.is_rtx_available() {
    println!("RTX acceleration enabled");

    if let Some(caps) = rtx.get_capabilities() {
        println!("GPU: Compute {}.{}", caps.compute_capability.0, caps.compute_capability.1);
        println!("Memory: {:.1} GB", caps.memory_gb);
        println!("Tensor Cores: {}", caps.has_tensor_cores);
    }
}
```

### RTX Model Selection

**Available noise suppression models:**
```rust
pub enum RtxModel {
    Basic,       // 2GB VRAM, good quality
    Enhanced,    // 4GB VRAM, excellent quality
    Professional // 8GB VRAM, broadcast quality
}

// Configure model based on available VRAM
let model = match rtx.get_vram_gb() {
    vram if vram >= 8.0 => RtxModel::Professional,
    vram if vram >= 4.0 => RtxModel::Enhanced,
    _ => RtxModel::Basic,
};

let processor = NoiseProcessor::with_rtx_model(&config, model)?;
```

---

## Performance Optimization

### 1. GPU Memory Optimization

**VRAM Management:**
```rust
use ghostwave_core::rtx::MemoryOptimizer;

struct OptimizedRtxProcessor {
    processor: NoiseProcessor,
    memory_optimizer: MemoryOptimizer,
}

impl OptimizedRtxProcessor {
    pub fn new(config: Config) -> anyhow::Result<Self> {
        // Pre-allocate GPU memory
        let memory_optimizer = MemoryOptimizer::new()?;
        memory_optimizer.preallocate_buffers(4096)?; // 4MB audio buffers

        let processor = NoiseProcessor::new(&config.noise_suppression)?;

        Ok(Self {
            processor,
            memory_optimizer,
        })
    }

    pub fn process_optimized(&mut self, input: &[f32], output: &mut [f32]) -> anyhow::Result<()> {
        // Reuse GPU memory allocations
        self.memory_optimizer.prepare_buffers(input.len())?;

        // Process with optimized memory usage
        self.processor.process(input, output)?;

        Ok(())
    }
}
```

### 2. CUDA Stream Optimization

**Concurrent Processing:**
```rust
use ghostwave_core::rtx::CudaStreamManager;

struct StreamOptimizedProcessor {
    processor: NoiseProcessor,
    stream_manager: CudaStreamManager,
}

impl StreamOptimizedProcessor {
    pub fn new() -> anyhow::Result<Self> {
        let config = Config::load("studio")?;
        let processor = NoiseProcessor::new(&config.noise_suppression)?;

        // Create multiple CUDA streams for overlap
        let stream_manager = CudaStreamManager::new(4)?; // 4 concurrent streams

        Ok(Self {
            processor,
            stream_manager,
        })
    }

    pub async fn process_concurrent(&mut self, audio_chunks: &[Vec<f32>]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();

        // Process chunks concurrently on different streams
        for (i, chunk) in audio_chunks.iter().enumerate() {
            let stream_id = i % 4;
            let mut output = vec![0.0f32; chunk.len()];

            self.stream_manager.process_on_stream(
                stream_id,
                |processor| processor.process(chunk, &mut output)
            )?;

            results.push(output);
        }

        // Synchronize all streams
        self.stream_manager.synchronize_all()?;

        Ok(results)
    }
}
```

### 3. Power and Thermal Management

**GPU Performance Tuning:**
```bash
# Set maximum performance mode
sudo nvidia-smi -pm 1

# Set power limit (adjust based on cooling)
sudo nvidia-smi -pl 350  # 350W for RTX 4090

# Set memory and core clocks
sudo nvidia-smi -ac 10501,2230  # Memory, Core clocks

# Monitor temperatures
watch -n 1 nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv
```

### 4. Application-Level Optimization

**Real-time Processing Pipeline:**
```rust
struct RealTimeRtxPipeline {
    processor: NoiseProcessor,
    gpu_buffer_pool: GpuBufferPool,
    processing_queue: crossbeam::queue::SegQueue<AudioFrame>,
}

impl RealTimeRtxPipeline {
    pub async fn run_pipeline(&mut self) -> anyhow::Result<()> {
        // Warm up GPU (important for consistent latency)
        self.warmup_gpu()?;

        loop {
            // Non-blocking audio frame retrieval
            if let Some(frame) = self.processing_queue.pop() {
                let start = std::time::Instant::now();

                // Get pre-allocated GPU buffer
                let gpu_buffer = self.gpu_buffer_pool.get_buffer()?;

                // Process on GPU
                self.processor.process_gpu(&frame.data, gpu_buffer)?;

                // Return buffer to pool
                self.gpu_buffer_pool.return_buffer(gpu_buffer);

                let processing_time = start.elapsed();

                // Monitor for performance issues
                if processing_time > std::time::Duration::from_micros(500) {
                    eprintln!("High GPU processing time: {:?}", processing_time);
                }
            }

            // Yield to prevent busy waiting
            tokio::task::yield_now().await;
        }
    }

    fn warmup_gpu(&mut self) -> anyhow::Result<()> {
        // Process dummy data to initialize GPU state
        let dummy_input = vec![0.01f32; 1024];
        let mut dummy_output = vec![0.0f32; 1024];

        for _ in 0..10 {
            self.processor.process(&dummy_input, &mut dummy_output)?;
        }

        Ok(())
    }
}
```

---

## Integration Examples

### 1. Gaming Setup with RTX

```rust
use ghostwave_core::{Config, NoiseProcessor, RtxAccelerator};

async fn setup_gaming_rtx() -> anyhow::Result<()> {
    // Configure for gaming with RTX
    let config = Config::load("streaming")?; // Lower latency for gaming

    // Initialize RTX processor
    let mut processor = NoiseProcessor::new(&config.noise_suppression)?;

    // Verify RTX is working
    if !processor.has_rtx_acceleration() {
        eprintln!("Warning: RTX acceleration not available, using CPU fallback");
    }

    println!("🎮 Gaming mode with RTX noise suppression active");
    println!("Processing mode: {}", processor.get_processing_mode());

    // Process game audio in real-time
    let mut audio_buffer = vec![0.0f32; 1024];
    loop {
        // Capture microphone input (Discord, etc.)
        capture_microphone_input(&mut audio_buffer)?;

        // Apply RTX noise suppression
        let mut clean_audio = vec![0.0f32; 1024];
        processor.process(&audio_buffer, &mut clean_audio)?;

        // Send to voice chat application
        send_to_voice_chat(&clean_audio)?;

        tokio::time::sleep(tokio::time::Duration::from_millis(21)).await; // ~48kHz
    }
}

fn capture_microphone_input(buffer: &mut [f32]) -> anyhow::Result<()> {
    // Implementation depends on audio backend
    Ok(())
}

fn send_to_voice_chat(audio: &[f32]) -> anyhow::Result<()> {
    // Send to Discord, TeamSpeak, etc.
    Ok(())
}
```

### 2. Professional Streaming with RTX

```rust
struct StreamingSetup {
    rtx_processor: NoiseProcessor,
    backup_processor: NoiseProcessor, // CPU fallback
    current_mode: ProcessingMode,
}

enum ProcessingMode {
    RTX,
    CPU,
    Hybrid,
}

impl StreamingSetup {
    pub fn new() -> anyhow::Result<Self> {
        let config = Config::load("professional")?;

        let rtx_processor = NoiseProcessor::new(&config.noise_suppression)?;
        let backup_processor = NoiseProcessor::new_cpu_only(&config.noise_suppression)?;

        let current_mode = if rtx_processor.has_rtx_acceleration() {
            ProcessingMode::RTX
        } else {
            ProcessingMode::CPU
        };

        Ok(Self {
            rtx_processor,
            backup_processor,
            current_mode,
        })
    }

    pub fn process_stream_audio(&mut self, input: &[f32], output: &mut [f32]) -> anyhow::Result<()> {
        match self.current_mode {
            ProcessingMode::RTX => {
                // Try RTX first
                match self.rtx_processor.process(input, output) {
                    Ok(_) => Ok(()),
                    Err(e) => {
                        eprintln!("RTX processing error, falling back to CPU: {}", e);
                        self.current_mode = ProcessingMode::CPU;
                        self.backup_processor.process(input, output)
                    }
                }
            }
            ProcessingMode::CPU => {
                self.backup_processor.process(input, output)
            }
            ProcessingMode::Hybrid => {
                // Use RTX for complex audio, CPU for simple audio
                let complexity = self.analyze_audio_complexity(input);
                if complexity > 0.5 {
                    self.rtx_processor.process(input, output)
                } else {
                    self.backup_processor.process(input, output)
                }
            }
        }
    }

    fn analyze_audio_complexity(&self, audio: &[f32]) -> f32 {
        // Simple complexity metric based on spectral content
        let mut energy = 0.0f32;
        for &sample in audio {
            energy += sample * sample;
        }
        energy.sqrt() / audio.len() as f32
    }
}
```

---

## Troubleshooting

### Common Issues

**1. RTX Not Detected**
```bash
# Check GPU visibility
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check GhostWave detection
./ghostwave --doctor

# Test with verbose output
./ghostwave --verbose --profile studio
```

**2. CUDA Memory Errors**
```bash
# Monitor VRAM usage
nvidia-smi -l 1

# Check for memory leaks
./ghostwave --bench --verbose

# Reduce model size if needed
./ghostwave --profile balanced  # Uses less VRAM
```

**3. Performance Issues**
```bash
# Check GPU utilization
nvidia-smi dmon -s pu

# Monitor thermal throttling
nvidia-smi -q -d TEMPERATURE

# Verify power settings
nvidia-smi -q -d POWER
```

**4. Driver Conflicts**
```bash
# Check loaded modules
lsmod | grep nvidia

# Reload drivers
sudo rmmod nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia nvidia_modeset nvidia_drm

# Verify installation
./ghostwave --doctor
```

### Performance Debugging

**GPU Profiling:**
```bash
# Profile GPU usage
nsys profile ./ghostwave --bench --profile studio

# Analyze CUDA kernels
ncu --target-processes all ./ghostwave --bench

# Memory profiling
compute-sanitizer --tool=memcheck ./ghostwave --bench
```

**Temperature Monitoring:**
```bash
# Continuous monitoring
while true; do
    nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.gr,clocks.mem --format=csv,noheader,nounits
    sleep 1
done
```

### Advanced Debugging

**CUDA Context Issues:**
```rust
use ghostwave_core::rtx::CudaDebugger;

let debugger = CudaDebugger::new();

// Monitor CUDA context health
debugger.monitor_context_health(|status| {
    match status {
        CudaStatus::Healthy => {},
        CudaStatus::MemoryPressure => {
            eprintln!("Warning: GPU memory pressure detected");
        },
        CudaStatus::ContextLost => {
            eprintln!("Error: CUDA context lost, reinitializing...");
            // Implement recovery
        }
    }
});
```

**Model Loading Issues:**
```bash
# Check model file integrity
sha256sum ~/.local/share/ghostwave/models/*.bin

# Verify model loading
./ghostwave --verbose 2>&1 | grep -i model

# Test with different models
./ghostwave --profile basic    # Smallest model
./ghostwave --profile balanced # Medium model
./ghostwave --profile studio   # Largest model
```

---

## Best Practices

### 1. System Setup

- Keep GPU drivers updated
- Monitor GPU temperatures during extended use
- Use adequate PSU for high-power GPUs
- Ensure proper GPU cooling

### 2. Application Integration

- Always check RTX availability before using
- Implement CPU fallback for reliability
- Pre-warm GPU for consistent latency
- Monitor VRAM usage to prevent OOM

### 3. Performance Optimization

- Use appropriate model size for available VRAM
- Batch process when possible for efficiency
- Implement proper memory management
- Monitor for thermal throttling

### 4. Development Workflow

1. **Verify Hardware**: `nvidia-smi`, `./ghostwave --doctor`
2. **Test Basic RTX**: `./ghostwave --verbose`
3. **Benchmark Performance**: `./ghostwave --bench --profile studio`
4. **Monitor Resources**: `nvidia-smi dmon`
5. **Optimize Settings**: Adjust model/buffer sizes as needed

---

This NVIDIA RTX integration guide ensures optimal GPU-accelerated noise suppression performance. For complete system integration, combine with [PIPEWIRE.md](PIPEWIRE.md) or [ALSA.md](ALSA.md) for audio backend configuration.