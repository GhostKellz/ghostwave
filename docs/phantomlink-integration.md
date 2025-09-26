# PhantomLink Integration Guide

This guide covers how to integrate GhostWave with PhantomLink, a high-performance professional audio mixing application for Linux, to create a complete audio production environment.

## Overview

PhantomLink is a Rust-based professional audio mixing and routing application that provides:
- 4-channel professional mixer with real-time spectrum analysis
- Low-latency audio processing (<20ms)
- VST plugin integration
- JACK audio server integration
- Multi-device audio routing

GhostWave can integrate with PhantomLink as both an input source (providing clean, processed audio) and as a processing component within PhantomLink's modular architecture.

## Integration Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Microphone    │────▶│    GhostWave     │────▶│   PhantomLink   │
│   (Raw Input)   │    │  (Noise Removal) │    │   (Mixing/FX)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌──────────────────┐
                       │    IPC Control   │
                       │   (Real-time     │
                       │   Parameter      │
                       │   Adjustment)    │
                       └──────────────────┘
```

## Integration Methods

### Method 1: JACK Audio Integration (Recommended)

Use JACK as the audio server to route audio between GhostWave and PhantomLink:

#### 1. Setup JACK with Low Latency

```bash
# Install JACK
sudo pacman -S jack2 qjackctl

# Start JACK with optimal settings for real-time audio
jackd -R -P 70 -d alsa -d hw:0 -r 48000 -p 128 -n 2
```

#### 2. Configure GhostWave as JACK Client

```bash
# Start GhostWave in JACK mode
ghostwave --jack --profile studio --frames 128 --samplerate 48000
```

#### 3. Configure PhantomLink for JACK

```bash
# PhantomLink will automatically detect JACK
phantomlink --audio-backend jack --low-latency
```

#### 4. Route Audio Connections

```bash
# Use qjackctl or command line to connect:
jack_connect "GhostWave Clean:output" "PhantomLink:input_1"
```

### Method 2: IPC Integration for Real-Time Control

GhostWave can be controlled by PhantomLink through its IPC interface:

#### 1. Start GhostWave IPC Server

```bash
# Start GhostWave with IPC server enabled
ghostwave --ipc-server --phantomlink
```

#### 2. PhantomLink Integration Code

```rust
use serde_json::json;
use tokio::net::UnixStream;

// Connect to GhostWave IPC
let stream = UnixStream::connect("/tmp/ghostwave.sock").await?;

// Example: Change processing profile
let request = json!({
    "method": "set_profile",
    "params": {
        "profile": "streaming"
    },
    "id": 1
});

// Send command to GhostWave
stream.write_all(request.to_string().as_bytes()).await?;
```

### Method 3: Embedded Integration (Advanced)

For tight integration, PhantomLink can embed GhostWave directly:

#### 1. Add GhostWave as Dependency

```toml
# In PhantomLink's Cargo.toml
[dependencies]
ghostwave-core = { path = "../ghostwave/ghostwave-core" }
```

#### 2. Embedded Processing Chain

```rust
use ghostwave_core::{GhostWaveProcessor, ProcessingProfile, FrameFormat};

pub struct PhantomLinkChannel {
    ghostwave: GhostWaveProcessor,
    // Other channel components...
}

impl PhantomLinkChannel {
    pub fn new() -> Result<Self> {
        let format = FrameFormat::studio(); // High quality for mixing
        let mut processor = GhostWaveProcessor::new(Default::default())?;
        processor.init(format.sample_rate, format.channels as u32, format.buffer_size)?;

        Ok(Self {
            ghostwave: processor,
        })
    }

    pub fn process_audio(&mut self, buffer: &mut [f32]) -> Result<()> {
        // Apply GhostWave processing before mixing
        self.ghostwave.process_inplace(buffer, buffer.len() / 2)?;

        // Continue with PhantomLink's processing chain...
        Ok(())
    }
}
```

## Real-Time Control Integration

### GhostWave Control Panel in PhantomLink

PhantomLink can provide a UI for GhostWave parameters:

```rust
// Example PhantomLink UI integration
impl PhantomLinkUI {
    fn render_ghostwave_controls(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label("GhostWave Noise Suppression");

            // Profile selection
            egui::ComboBox::from_label("Profile")
                .selected_text(format!("{:?}", self.ghostwave_profile))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.ghostwave_profile, Profile::Balanced, "Balanced");
                    ui.selectable_value(&mut self.ghostwave_profile, Profile::Streaming, "Streaming");
                    ui.selectable_value(&mut self.ghostwave_profile, Profile::Studio, "Studio");
                });

            // Noise reduction strength
            ui.add(egui::Slider::new(&mut self.noise_strength, 0.0..=1.0)
                .text("Noise Reduction"));

            // Voice enhancement
            ui.add(egui::Slider::new(&mut self.voice_enhancement, 0.0..=1.0)
                .text("Voice Enhancement"));
        });
    }
}
```

## Performance Optimization

### Shared Buffer Pool

Minimize allocations by sharing buffers between GhostWave and PhantomLink:

```rust
use ghostwave_core::LockFreeAudioBuffer;

pub struct SharedAudioSystem {
    buffer_pool: Arc<AudioMemoryPool>,
    ghostwave_ring: Arc<LockFreeAudioBuffer>,
}

impl SharedAudioSystem {
    pub fn new(buffer_size: usize, pool_size: usize) -> Self {
        Self {
            buffer_pool: Arc::new(AudioMemoryPool::new(buffer_size, pool_size)),
            ghostwave_ring: Arc::new(LockFreeAudioBuffer::new_with_channels(
                buffer_size * 4, 48000, 2
            )),
        }
    }

    pub fn process_chain(&self, input: &[f32]) -> Result<Vec<f32>> {
        // Get buffer from pool (zero allocation)
        let mut buffer = self.buffer_pool.get_buffer()
            .ok_or_else(|| anyhow::anyhow!("Buffer pool exhausted"))?;

        // Copy input
        buffer[..input.len()].copy_from_slice(input);

        // Write to GhostWave ring buffer
        self.ghostwave_ring.write(&buffer)?;

        // Process...

        // Return buffer to pool
        self.buffer_pool.return_buffer(buffer);

        Ok(processed)
    }
}
```

## Configuration Examples

### Studio Configuration

```toml
# ~/.config/phantomlink/studio.toml
[ghostwave]
profile = "studio"
sample_rate = 96000
buffer_size = 256
noise_reduction_strength = 0.3
voice_enhancement = 0.2
highpass_frequency = 40.0

[phantomlink.mixer]
channel_1_source = "ghostwave_clean"
channel_1_gain = 0.8
channel_1_effects = ["compressor", "eq"]
```

### Streaming Configuration

```toml
# ~/.config/phantomlink/streaming.toml
[ghostwave]
profile = "streaming"
sample_rate = 48000
buffer_size = 128
noise_reduction_strength = 0.85
voice_enhancement = 0.7
gate_threshold = -40.0

[phantomlink.output]
format = "mp3"
bitrate = 320
target_lufs = -16.0
```

## Monitoring and Diagnostics

### Real-Time Performance Monitoring

```rust
// Monitor integration performance
pub struct IntegrationMonitor {
    ghostwave_latency: MovingAverage,
    phantomlink_latency: MovingAverage,
    total_latency: MovingAverage,
}

impl IntegrationMonitor {
    pub fn measure_chain_latency(&mut self) {
        let start = Instant::now();

        // Measure GhostWave processing
        let gw_start = Instant::now();
        // ... GhostWave processing ...
        self.ghostwave_latency.update(gw_start.elapsed());

        // Measure PhantomLink processing
        let pl_start = Instant::now();
        // ... PhantomLink processing ...
        self.phantomlink_latency.update(pl_start.elapsed());

        self.total_latency.update(start.elapsed());
    }

    pub fn report(&self) {
        println!("Integration Performance:");
        println!("  GhostWave: {:.2}ms", self.ghostwave_latency.avg() * 1000.0);
        println!("  PhantomLink: {:.2}ms", self.phantomlink_latency.avg() * 1000.0);
        println!("  Total Chain: {:.2}ms", self.total_latency.avg() * 1000.0);
    }
}
```

## Troubleshooting

### Common Issues

1. **High Latency**
   - Reduce buffer sizes in both applications
   - Use JACK with real-time priority
   - Check CPU usage and optimize processing

2. **Audio Dropouts**
   - Increase buffer sizes
   - Check for XRuns in JACK
   - Monitor system load

3. **IPC Connection Issues**
   - Verify GhostWave IPC server is running
   - Check socket permissions
   - Ensure correct socket path

### Debug Commands

```bash
# Check JACK status
jack_lsp -c

# Monitor GhostWave performance
ghostwave --bench --profile streaming

# Test IPC connection
echo '{"method": "get_profile", "id": 1}' | nc -U /tmp/ghostwave.sock

# Check audio routing
pactl list sources | grep -i phantomlink
```

## Advanced Features

### Automatic Profile Switching

PhantomLink can automatically adjust GhostWave settings based on context:

```rust
pub struct AdaptiveProcessor {
    current_scene: Scene,
    ghostwave_client: GhostWaveIPC,
}

impl AdaptiveProcessor {
    pub fn on_scene_change(&mut self, new_scene: Scene) {
        let profile = match new_scene {
            Scene::Recording => ProcessingProfile::Studio,
            Scene::Streaming => ProcessingProfile::Streaming,
            Scene::Gaming => ProcessingProfile::Balanced,
        };

        self.ghostwave_client.set_profile(profile).await?;
    }
}
```

### Multi-Input Processing

Handle multiple microphones through separate GhostWave instances:

```rust
pub struct MultiInputProcessor {
    processors: HashMap<String, GhostWaveProcessor>,
}

impl MultiInputProcessor {
    pub fn process_input(&mut self, device_name: &str, buffer: &mut [f32]) -> Result<()> {
        if let Some(processor) = self.processors.get_mut(device_name) {
            processor.process_inplace(buffer, buffer.len() / 2)?;
        }
        Ok(())
    }
}
```

## Conclusion

The GhostWave-PhantomLink integration provides a powerful, professional audio processing chain for Linux users. By leveraging both applications' strengths, users can achieve studio-quality audio processing with real-time mixing capabilities.

For the best experience:
- Use JACK for low-latency audio routing
- Enable IPC for real-time control
- Monitor performance with built-in diagnostics
- Configure profiles based on use case

This integration bridges the gap between noise suppression and professional mixing, providing a complete audio solution for content creators, podcasters, and audio professionals.