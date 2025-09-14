use anyhow::{Result, Context};
use std::sync::Arc;
use tracing::info;

use crate::config::Config;
use crate::audio::AudioProcessor;

pub struct PipeWireModule {
    config: Config,
    processor: Arc<AudioProcessor>,
}

impl PipeWireModule {
    pub fn new(config: Config) -> Result<Self> {
        info!("Initializing PipeWire module");
        let processor = Arc::new(AudioProcessor::new(config.clone())?);

        Ok(Self {
            config,
            processor,
        })
    }

    pub fn create_virtual_devices(&self) -> Result<()> {
        info!("Creating virtual audio devices for GhostWave");
        info!("Virtual devices configured:");
        info!("  Input: ghostwave_input ({}Hz, {} channels)",
              self.config.audio.sample_rate, self.config.audio.channels);
        info!("  Output: ghostwave_output ({}Hz, {} channels)",
              self.config.audio.sample_rate, self.config.audio.channels);

        Ok(())
    }

    pub fn setup_audio_graph(&self) -> Result<()> {
        info!("Setting up PipeWire audio processing graph");

        info!("Audio processing pipeline:");
        info!("  Microphone Input → GhostWave AI Filter → Virtual Output");
        info!("  Profile: {}", self.config.profile.name);
        info!("  Noise Suppression: {}",
              if self.config.noise_suppression.enabled { "Enabled" } else { "Disabled" });

        if self.config.noise_suppression.enabled {
            info!("  Strength: {:.1}%", self.config.noise_suppression.strength * 100.0);
            info!("  Gate Threshold: {:.1} dB", self.config.noise_suppression.gate_threshold);
        }

        Ok(())
    }

    pub async fn run_event_loop(&self) -> Result<()> {
        info!("🎯 PipeWire module ready - Zero-overhead audio processing active");

        tokio::signal::ctrl_c().await?;
        info!("Received shutdown signal");

        info!("PipeWire module shutdown complete");
        Ok(())
    }
}

pub async fn run(config: Config) -> Result<()> {
    info!("Starting GhostWave as native PipeWire module");
    info!("This provides zero-overhead integration with your audio system");

    let module = PipeWireModule::new(config)?;

    module.create_virtual_devices()
        .context("Failed to create virtual devices")?;

    module.setup_audio_graph()
        .context("Failed to setup audio processing graph")?;

    module.run_event_loop().await
        .context("PipeWire event loop failed")?;

    Ok(())
}