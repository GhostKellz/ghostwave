# GhostWave Integration Guide

This guide shows how to integrate `ghostwave-core` into your Rust projects for real-time audio processing with NVIDIA RTX-powered noise suppression.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Advanced Integration](#advanced-integration)
- [PhantomLink Integration](#phantomlink-integration)
- [Performance Optimization](#performance-optimization)
- [Examples](#examples)

---

## Quick Start

Add GhostWave to your project:

```toml
[dependencies]
ghostwave-core = { git = "https://github.com/ghostkellz/ghostwave", features = ["pipewire-backend", "nvidia-rtx"] }
tokio = { version = "1.0", features = ["full"] }
anyhow = "1.0"
```

Basic noise suppression:

```rust
use ghostwave_core::{Config, NoiseProcessor};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load configuration
    let config = Config::load("balanced")?;

    // Create noise processor
    let mut processor = NoiseProcessor::new(&config.noise_suppression)?;

    // Process audio frame
    let input = vec![0.1f32; 1024];  // Your audio input
    let mut output = vec![0.0f32; 1024];

    processor.process(&input, &mut output)?;

    // Output now contains noise-suppressed audio
    println!("Processing mode: {}", processor.get_processing_mode());

    Ok(())
}
```

---

## Installation

### Feature Flags

Choose the features you need:

```toml
[dependencies]
ghostwave-core = {
    git = "https://github.com/ghostkellz/ghostwave",
    features = [
        "pipewire-backend",  # Modern Linux audio
        "alsa-backend",      # Direct hardware access
        "jack-backend",      # Professional audio
        "cpal-backend",      # Cross-platform (default)
        "nvidia-rtx"         # GPU acceleration
    ]
}
```

### Minimal Installation

For basic audio processing without GPU acceleration:

```toml
[dependencies]
ghostwave-core = {
    git = "https://github.com/ghostkellz/ghostwave",
    default-features = false,
    features = ["cpal-backend"]
}
```

### System Dependencies

Ensure you have the required system libraries:

```bash
# Ubuntu/Debian
sudo apt install libasound2-dev pkg-config

# Arch Linux
sudo pacman -S alsa-lib pkg-config

# For NVIDIA RTX features
sudo apt install nvidia-cuda-toolkit  # Ubuntu
sudo pacman -S cuda                    # Arch
```

---

## Basic Usage

### 1. Configuration Management

```rust
use ghostwave_core::{Config, AudioConfig, NoiseSuppressionConfig};

// Load built-in profile
let config = Config::load("studio")?;

// Create custom configuration
let custom_config = Config {
    profile: ProfileConfig {
        name: "custom".to_string(),
        description: "Custom profile".to_string(),
    },
    audio: AudioConfig {
        sample_rate: 48000,
        buffer_size: 256,
        channels: 2,
        input_device: None,
        output_device: None,
    },
    noise_suppression: NoiseSuppressionConfig {
        enabled: true,
        strength: 0.8,
        gate_threshold: -30.0,
        release_time: 0.1,
    },
};

// Override specific settings
let config = Config::load("balanced")?
    .with_overrides(Some(96000), Some(128)); // 96kHz, 128 frames
```

### 2. Audio Device Detection

```rust
use ghostwave_core::{DeviceDetector, AudioDeviceType};

let detector = DeviceDetector::new();

// Detect all audio devices
let devices = detector.detect_devices().await?;
for device in devices {
    println!("Found: {} {} ({})", device.vendor, device.model, device.name);
    if device.is_xlr_interface {
        println!("  XLR interface - Recommended: {}", device.recommended_profile);
    }
}

// Find specific device
if let Some(scarlett) = detector.find_scarlett_solo_4th_gen().await? {
    println!("Scarlett Solo detected!");
    let optimal_config = detector.get_optimal_config_for_device(&scarlett).await?;
    println!("Optimal sample rates: {:?}", scarlett.sample_rates);
}
```

### 3. Audio Backend Selection

```rust
use ghostwave_core::AudioBackend;

// Check available backends
let backends = AudioBackend::available_backends();
println!("Available backends: {:?}", backends);

// Get recommended backend for system
if let Some(backend) = AudioBackend::recommended() {
    println!("Recommended backend: {}", backend);

    if backend.is_available() {
        println!("Backend is ready for use");
    }
}

// Check specific backend
if AudioBackend::PipeWire.is_available() {
    println!("PipeWire is available");
}
```

---

## Advanced Integration

### 1. Real-Time Audio Processing

```rust
use ghostwave_core::{
    NoiseProcessor, RealTimeScheduler, LockFreeAudioBuffer,
    AudioBenchmark, TARGET_LATENCY_MS
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

struct RealTimeAudioProcessor {
    processor: NoiseProcessor,
    scheduler: RealTimeScheduler,
    input_buffer: LockFreeAudioBuffer,
    output_buffer: LockFreeAudioBuffer,
    benchmark: AudioBenchmark,
    running: Arc<AtomicBool>,
}

impl RealTimeAudioProcessor {
    pub fn new(config: Config) -> anyhow::Result<Self> {
        let processor = NoiseProcessor::new(&config.noise_suppression)?;
        let scheduler = RealTimeScheduler::new(
            config.audio.sample_rate,
            config.audio.buffer_size as usize
        );

        let buffer_capacity = config.audio.buffer_size as usize * 4;
        let input_buffer = LockFreeAudioBuffer::new(buffer_capacity, config.audio.sample_rate);
        let output_buffer = LockFreeAudioBuffer::new(buffer_capacity, config.audio.sample_rate);

        let benchmark = AudioBenchmark::new(
            config.audio.sample_rate,
            config.audio.buffer_size as usize
        );

        Ok(Self {
            processor,
            scheduler,
            input_buffer,
            output_buffer,
            benchmark,
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn start_processing(&mut self) -> anyhow::Result<()> {
        // Optimize thread for real-time audio
        RealTimeScheduler::optimize_thread_for_audio()?;

        self.running.store(true, Ordering::SeqCst);

        let buffer_size = 1024; // Process in chunks
        let mut input_chunk = vec![0.0f32; buffer_size];
        let mut output_chunk = vec![0.0f32; buffer_size];

        while self.running.load(Ordering::SeqCst) {
            let frame_start = std::time::Instant::now();

            // Read from input buffer
            if self.input_buffer.read(&mut input_chunk)? > 0 {
                // Process audio
                self.processor.process(&input_chunk, &mut output_chunk)?;

                // Write to output buffer
                self.output_buffer.write(&output_chunk)?;

                // Record performance
                let processing_time = frame_start.elapsed();
                self.benchmark.record_frame_processing(processing_time);
            }

            // Maintain frame timing
            self.scheduler.sleep_until_next_frame(frame_start);
        }

        Ok(())
    }

    pub fn write_input(&self, audio_data: &[f32]) -> anyhow::Result<usize> {
        Ok(self.input_buffer.write(audio_data)?)
    }

    pub fn read_output(&self, audio_data: &mut [f32]) -> anyhow::Result<usize> {
        Ok(self.output_buffer.read(audio_data)?)
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    pub fn get_performance_stats(&self) {
        self.benchmark.report_stats();
    }
}
```

### 2. Multi-threaded Integration

```rust
use std::sync::mpsc;
use std::thread;

struct AudioPipeline {
    input_thread: Option<thread::JoinHandle<()>>,
    processing_thread: Option<thread::JoinHandle<()>>,
    output_thread: Option<thread::JoinHandle<()>>,
    shutdown_tx: mpsc::Sender<()>,
}

impl AudioPipeline {
    pub fn new(config: Config) -> anyhow::Result<Self> {
        let (shutdown_tx, shutdown_rx) = mpsc::channel();

        // Create lock-free buffers for inter-thread communication
        let input_buffer = Arc::new(LockFreeAudioBuffer::new(8192, config.audio.sample_rate));
        let output_buffer = Arc::new(LockFreeAudioBuffer::new(8192, config.audio.sample_rate));

        // Processing thread
        let processor = NoiseProcessor::new(&config.noise_suppression)?;
        let processing_buffer = input_buffer.clone();
        let processing_output = output_buffer.clone();
        let processing_shutdown = shutdown_rx;

        let processing_thread = thread::spawn(move || {
            let mut temp_input = vec![0.0f32; 1024];
            let mut temp_output = vec![0.0f32; 1024];

            loop {
                // Check for shutdown signal
                if processing_shutdown.try_recv().is_ok() {
                    break;
                }

                // Process available audio
                if processing_buffer.read(&mut temp_input).unwrap() > 0 {
                    processor.process(&temp_input, &mut temp_output).unwrap();
                    processing_output.write(&temp_output).unwrap();
                }

                thread::sleep(std::time::Duration::from_micros(100));
            }
        });

        Ok(Self {
            input_thread: None,
            processing_thread: Some(processing_thread),
            output_thread: None,
            shutdown_tx,
        })
    }

    pub fn shutdown(self) -> anyhow::Result<()> {
        self.shutdown_tx.send(())?;

        if let Some(thread) = self.processing_thread {
            thread.join().map_err(|_| anyhow::anyhow!("Failed to join processing thread"))?;
        }

        Ok(())
    }
}
```

---

## PhantomLink Integration

PhantomLink allows GhostWave to appear as a virtual audio device for seamless integration:

### 1. JSON-RPC API Integration

```rust
use ghostwave_core::ipc::{GhostWaveRpc, DeviceInfo, AudioStats};
use jsonrpc_core::{Result as RpcResult, Error as RpcError};

struct PhantomLinkIntegration {
    ghostwave: RealTimeAudioProcessor,
}

impl GhostWaveRpc for PhantomLinkIntegration {
    fn register_xlr_device(&self, device_name: String, channels: u8) -> RpcResult<DeviceInfo> {
        Ok(DeviceInfo {
            device_id: "ghostwave-xlr".to_string(),
            name: device_name,
            channels,
            sample_rate: 48000,
            buffer_size: 256,
            latency_ms: 5.3,
            processing_mode: self.ghostwave.processor.get_processing_mode(),
        })
    }

    fn set_noise_suppression(&self, enabled: bool, strength: f32) -> RpcResult<bool> {
        // Update noise processor settings
        Ok(true)
    }

    fn get_audio_stats(&self) -> RpcResult<AudioStats> {
        let stats = self.ghostwave.benchmark.get_stats();
        Ok(AudioStats {
            frames_processed: stats.total_frames,
            xrun_count: stats.xrun_count,
            avg_latency_ms: stats.target_frame_time.as_secs_f64() * 1000.0,
            cpu_usage: 15.2, // Example value
        })
    }
}

// Start IPC server
async fn start_phantomlink_server() -> anyhow::Result<()> {
    let integration = PhantomLinkIntegration {
        ghostwave: RealTimeAudioProcessor::new(Config::load("streaming")?)?,
    };

    let mut io = jsonrpc_core::IoHandler::new();
    io.extend_with(integration.to_delegate());

    let server = jsonrpc_ipc_server::ServerBuilder::new(io)
        .start("/tmp/ghostwave.sock")?;

    server.wait();
    Ok(())
}
```

### 2. Virtual Audio Device

```rust
struct VirtualAudioDevice {
    ghostwave: RealTimeAudioProcessor,
    phantom_config: PhantomLinkConfig,
}

impl VirtualAudioDevice {
    pub fn new(config: PhantomLinkConfig) -> anyhow::Result<Self> {
        let ghostwave_config = Config::load(&config.profile)?;
        let ghostwave = RealTimeAudioProcessor::new(ghostwave_config)?;

        Ok(Self {
            ghostwave,
            phantom_config: config,
        })
    }

    pub async fn process_xlr_input(&mut self, xlr_data: &[f32]) -> anyhow::Result<Vec<f32>> {
        // Apply GhostWave processing to XLR input
        let mut output = vec![0.0f32; xlr_data.len()];
        self.ghostwave.processor.process(xlr_data, &mut output)?;
        Ok(output)
    }
}
```

---

## Performance Optimization

### 1. Memory Pool Usage

```rust
use ghostwave_core::AudioMemoryPool;

struct OptimizedProcessor {
    processor: NoiseProcessor,
    memory_pool: AudioMemoryPool,
}

impl OptimizedProcessor {
    pub fn new(config: Config) -> anyhow::Result<Self> {
        let processor = NoiseProcessor::new(&config.noise_suppression)?;
        let memory_pool = AudioMemoryPool::new(
            config.audio.buffer_size as usize,
            16 // Pool size
        );

        Ok(Self { processor, memory_pool })
    }

    pub fn process_with_pool(&mut self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        // Get buffer from pool (zero-allocation)
        let mut output = self.memory_pool.get_buffer()
            .unwrap_or_else(|| vec![0.0f32; input.len()]);

        // Resize if needed
        output.resize(input.len(), 0.0);

        // Process audio
        self.processor.process(input, &mut output)?;

        // Return buffer to pool for reuse
        let result = output.clone();
        self.memory_pool.return_buffer(output);

        Ok(result)
    }
}
```

### 2. Batch Processing

```rust
struct BatchProcessor {
    processor: NoiseProcessor,
    batch_size: usize,
}

impl BatchProcessor {
    pub fn process_batch(&mut self, audio_stream: &[f32]) -> anyhow::Result<Vec<f32>> {
        let mut results = Vec::with_capacity(audio_stream.len());

        for chunk in audio_stream.chunks(self.batch_size) {
            let mut output = vec![0.0f32; chunk.len()];
            self.processor.process(chunk, &mut output)?;
            results.extend_from_slice(&output);
        }

        Ok(results)
    }
}
```

---

## Examples

### 1. Discord Bot Integration

```rust
use ghostwave_core::{Config, NoiseProcessor};

struct DiscordAudioProcessor {
    processor: NoiseProcessor,
}

impl DiscordAudioProcessor {
    pub fn new() -> anyhow::Result<Self> {
        let config = Config::load("streaming")?; // Optimized for streaming
        let processor = NoiseProcessor::new(&config.noise_suppression)?;

        Ok(Self { processor })
    }

    pub fn process_voice_packet(&mut self, voice_data: &[f32]) -> anyhow::Result<Vec<f32>> {
        let mut clean_audio = vec![0.0f32; voice_data.len()];
        self.processor.process(voice_data, &mut clean_audio)?;
        Ok(clean_audio)
    }
}
```

### 2. OBS Plugin Integration

```rust
struct ObsAudioFilter {
    ghostwave: RealTimeAudioProcessor,
}

impl ObsAudioFilter {
    pub fn new() -> anyhow::Result<Self> {
        let config = Config::load("balanced")?;
        let ghostwave = RealTimeAudioProcessor::new(config)?;

        Ok(Self { ghostwave })
    }

    pub fn filter_audio_frame(&mut self, audio_frame: &mut [f32]) -> anyhow::Result<()> {
        let mut temp_output = vec![0.0f32; audio_frame.len()];
        self.ghostwave.processor.process(audio_frame, &mut temp_output)?;
        audio_frame.copy_from_slice(&temp_output);
        Ok(())
    }
}
```

### 3. Real-time Audio Stream

```rust
use tokio::time::{interval, Duration};

async fn real_time_streaming_example() -> anyhow::Result<()> {
    let config = Config::load("studio")?;
    let mut processor = NoiseProcessor::new(&config.noise_suppression)?;

    let mut interval = interval(Duration::from_millis(21)); // ~48kHz/1024 frames

    loop {
        interval.tick().await;

        // Simulate audio input (replace with actual audio capture)
        let input_frame = generate_audio_frame(1024);
        let mut output_frame = vec![0.0f32; 1024];

        // Process with GhostWave
        processor.process(&input_frame, &mut output_frame)?;

        // Send to audio output (replace with actual audio playback)
        send_to_audio_output(output_frame);
    }
}

fn generate_audio_frame(size: usize) -> Vec<f32> {
    // Replace with actual audio capture
    vec![0.1f32; size]
}

fn send_to_audio_output(audio: Vec<f32>) {
    // Replace with actual audio playback
    println!("Playing {} samples", audio.len());
}
```

---

## Troubleshooting

### Common Integration Issues

**1. Feature Compilation Errors**
```bash
# Ensure you have the required system dependencies
sudo apt install libasound2-dev pkg-config  # For ALSA
sudo apt install nvidia-cuda-toolkit         # For RTX features
```

**2. Runtime Performance Issues**
```rust
// Enable real-time scheduling
RealTimeScheduler::optimize_thread_for_audio()?;

// Use appropriate buffer sizes
let optimal_buffer = RealTimeScheduler::get_optimal_buffer_size(48000, 15);
```

**3. Memory Allocation in Audio Path**
```rust
// Pre-allocate buffers
let mut output_buffer = vec![0.0f32; max_expected_size];

// Reuse buffers instead of allocating
output_buffer.resize(input.len(), 0.0);
processor.process(input, &mut output_buffer)?;
```

### Performance Monitoring

```rust
// Monitor processing performance
let benchmark = AudioBenchmark::new(48000, 512);
let start = std::time::Instant::now();

processor.process(input, output)?;

let processing_time = start.elapsed();
benchmark.record_frame_processing(processing_time);

// Report statistics
benchmark.report_stats();
```

---

This integration guide provides comprehensive examples for embedding GhostWave into your applications. For more specific use cases or advanced optimization techniques, refer to the [DOCS.md](DOCS.md) technical documentation.