# GhostWave API Reference

Complete API documentation for integrating GhostWave into your applications.

## Table of Contents

- [Core Audio Processing](#core-audio-processing)
- [Device Detection](#device-detection)
- [Real-Time Optimization](#real-time-optimization)
- [Backend Selection](#backend-selection)
- [Configuration](#configuration)

---

## Core Audio Processing

### NoiseProcessor

The main audio processing interface for noise suppression.

```rust
use ghostwave_core::{Config, NoiseProcessor};

// Initialize processor
let config = Config::load("studio")?;
let mut processor = NoiseProcessor::new(&config.noise_suppression)?;

// Process audio frame
let input = vec![0.1f32; 1024];
let mut output = vec![0.0f32; 1024];
processor.process(&input, &mut output)?;

// Check processing mode
let mode = processor.get_processing_mode(); // "RTX GPU + CPU Gate" or "CPU Only"
```

#### Methods

##### `NoiseProcessor::new(config: &NoiseSuppressionConfig) -> Result<Self>`

Creates a new noise processor with the specified configuration.

**Parameters:**
- `config`: Noise suppression settings

**Returns:**
- `Result<NoiseProcessor>`: Initialized processor or error

**Example:**
```rust
let config = NoiseSuppressionConfig {
    strength: 0.8,
    gate_threshold_db: -40.0,
    use_rtx: true,
};
let processor = NoiseProcessor::new(&config)?;
```

##### `processor.process(&input: &[f32], output: &mut [f32]) -> Result<()>`

Process an audio buffer with noise suppression.

**Parameters:**
- `input`: Input audio samples (interleaved if multi-channel)
- `output`: Output buffer (must be same size as input)

**Returns:**
- `Result<()>`: Success or processing error

**Performance:** <1ms per frame on RTX GPUs

##### `processor.get_processing_mode() -> &str`

Returns current processing mode.

**Returns:**
- `"RTX GPU"` - Using NVIDIA GPU acceleration
- `"CPU Fallback"` - Using CPU spectral filtering

---

## Device Detection

### DeviceDetector

Auto-detects and configures audio hardware.

```rust
use ghostwave_core::DeviceDetector;

// Auto-detect audio devices
let detector = DeviceDetector::new();
let devices = detector.detect_devices().await?;

// Find specific device
let scarlett = detector.find_scarlett_solo_4th_gen().await?;
if let Some(device) = scarlett {
    let config = detector.get_optimal_config_for_device(&device).await?;
}
```

#### Methods

##### `DeviceDetector::new() -> Self`

Creates a new device detector.

##### `detector.detect_devices() -> Result<Vec<AudioDevice>>`

Scans for available audio devices.

**Returns:**
- `Vec<AudioDevice>`: List of detected devices

##### `detector.find_scarlett_solo_4th_gen() -> Result<Option<AudioDevice>>`

Finds Focusrite Scarlett Solo 4th Gen XLR interface.

**Returns:**
- `Some(AudioDevice)` if found
- `None` if not present

##### `detector.get_optimal_config_for_device(device: &AudioDevice) -> Result<AudioConfig>`

Generates optimal configuration for a specific device.

**Parameters:**
- `device`: Target audio device

**Returns:**
- `AudioConfig`: Recommended settings (sample rate, buffer size, etc.)

---

## Real-Time Optimization

### RealTimeScheduler

Manages real-time thread scheduling and optimization.

```rust
use ghostwave_core::RealTimeScheduler;

// Set up real-time scheduling
RealTimeScheduler::optimize_thread_for_audio()?;

// Create scheduler for specific configuration
let scheduler = RealTimeScheduler::new(48000, 512);
let optimal_buffer = RealTimeScheduler::get_optimal_buffer_size(48000, 15);
```

#### Methods

##### `RealTimeScheduler::optimize_thread_for_audio() -> Result<()>`

Optimizes current thread for audio processing.

**Effects:**
- Sets FIFO scheduling policy
- Increases thread priority
- Pins to performance cores (if available)

**Requires:** CAP_SYS_NICE capability

```bash
sudo setcap cap_sys_nice+ep ./your_binary
```

##### `RealTimeScheduler::new(sample_rate: u32, buffer_size: usize) -> Self`

Creates a scheduler for specific audio parameters.

**Parameters:**
- `sample_rate`: Audio sample rate (Hz)
- `buffer_size`: Buffer size in frames

##### `RealTimeScheduler::get_optimal_buffer_size(sample_rate: u32, target_latency_ms: u32) -> usize`

Calculates optimal buffer size for target latency.

**Parameters:**
- `sample_rate`: Audio sample rate
- `target_latency_ms`: Desired latency in milliseconds

**Returns:**
- `usize`: Recommended buffer size (power of 2)

**Example:**
```rust
// For 48kHz audio with <10ms latency
let buffer = RealTimeScheduler::get_optimal_buffer_size(48000, 10);
// Returns: 512 frames (~10.67ms latency)
```

---

## Backend Selection

### AudioBackend

Enumerate and select audio backends.

```rust
use ghostwave_core::AudioBackend;

// Enumerate available backends
let backends = AudioBackend::available_backends();

// Check backend availability
if AudioBackend::PipeWire.is_available() {
    println!("PipeWire is available");
}

// Get recommended backend
let recommended = AudioBackend::recommended();
```

#### Enum Variants

```rust
pub enum AudioBackend {
    PipeWire,  // Modern Linux audio (recommended)
    ALSA,      // Direct hardware access (lowest latency)
    JACK,      // Professional audio production
    CPAL,      // Cross-platform fallback
}
```

#### Methods

##### `AudioBackend::available_backends() -> Vec<AudioBackend>`

Returns list of available backends on current system.

##### `backend.is_available() -> bool`

Checks if specific backend is available.

##### `AudioBackend::recommended() -> AudioBackend`

Returns recommended backend for current system.

**Priority:**
1. PipeWire (if available)
2. JACK (if running)
3. ALSA (direct hardware)
4. CPAL (fallback)

---

## Configuration

### Config

Main configuration structure for GhostWave.

```rust
use ghostwave_core::Config;

// Load built-in profile
let config = Config::load("studio")?;

// Load from file
let config = Config::load_from_file("my_config.json")?;

// Create custom config
let config = Config {
    profile: ProfileConfig::studio(),
    audio: AudioConfig {
        sample_rate: 48000,
        buffer_size: 256,
        channels: 2,
        input_device: None,
        output_device: None,
    },
    noise_suppression: NoiseSuppressionConfig {
        strength: 0.8,
        gate_threshold_db: -40.0,
        use_rtx: true,
    },
};
```

### ProfileConfig

Pre-configured profiles for different use cases.

#### Built-in Profiles

```rust
// Studio profile - ultra-low latency
let studio = ProfileConfig::studio();
// - Buffer: 128 frames
// - Latency: <3ms
// - CPU: Higher usage

// Balanced profile - general purpose
let balanced = ProfileConfig::balanced();
// - Buffer: 512 frames
// - Latency: ~10ms
// - CPU: Moderate usage

// Streaming profile - content creation
let streaming = ProfileConfig::streaming();
// - Buffer: 1024 frames
// - Latency: ~20ms
// - CPU: Lower usage
```

### NoiseSuppressionConfig

Configuration for noise suppression engine.

```rust
pub struct NoiseSuppressionConfig {
    /// Strength (0.0-1.0)
    pub strength: f32,

    /// Gate threshold in dB (-60.0 to 0.0)
    pub gate_threshold_db: f32,

    /// Enable RTX acceleration
    pub use_rtx: bool,
}
```

**Recommended Settings:**

| Use Case | Strength | Threshold | RTX |
|----------|----------|-----------|-----|
| Gaming/Streaming | 0.7 | -40dB | true |
| Professional Recording | 0.8 | -45dB | true |
| Conference Calls | 0.6 | -35dB | true |
| Maximum Quality | 0.9 | -50dB | true |

---

## Error Handling

All functions return `Result<T, NvControlError>` for comprehensive error handling.

```rust
use ghostwave_core::{NoiseProcessor, NvControlError};

match NoiseProcessor::new(&config) {
    Ok(processor) => {
        // Use processor
    }
    Err(NvControlError::CudaNotAvailable) => {
        eprintln!("CUDA not available, using CPU fallback");
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

### Error Types

```rust
pub enum NvControlError {
    CudaNotAvailable,
    InvalidConfig(String),
    DeviceNotFound,
    AudioProcessingError(String),
    // ... more variants
}
```

---

## See Also

- [Integration Guide](INTEGRATION.md) - Complete integration examples
- [Architecture](ARCHITECTURE.md) - System architecture overview
- [Performance](PERFORMANCE.md) - Optimization techniques
