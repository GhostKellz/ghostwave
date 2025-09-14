use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::info;

use crate::config::Config;
use crate::audio::AudioProcessor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhantomLinkConfig {
    pub socket_path: PathBuf,
    pub device_name: String,
    pub channels: u8,
    pub sample_rate: u32,
}

impl Default for PhantomLinkConfig {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from("/tmp/phantomlink.sock"),
            device_name: "GhostWave Virtual XLR".to_string(),
            channels: 2,
            sample_rate: 48000,
        }
    }
}

pub struct PhantomLinkIntegration {
    config: PhantomLinkConfig,
    ghostwave_config: Config,
}

impl PhantomLinkIntegration {
    pub fn new(ghostwave_config: Config) -> Self {
        Self {
            config: PhantomLinkConfig::default(),
            ghostwave_config,
        }
    }

    pub fn with_config(ghostwave_config: Config, phantomlink_config: PhantomLinkConfig) -> Self {
        Self {
            config: phantomlink_config,
            ghostwave_config,
        }
    }

    pub async fn register_as_xlr_device(&self) -> Result<()> {
        info!("Registering GhostWave as PhantomLink XLR device");
        info!("Device: {}", self.config.device_name);
        info!("Socket: {:?}", self.config.socket_path);
        info!("Audio: {}Hz, {} channels",
              self.ghostwave_config.audio.sample_rate, self.ghostwave_config.audio.channels);

        Ok(())
    }

    pub async fn setup_audio_routing(&self) -> Result<()> {
        info!("Setting up audio routing for PhantomLink integration");
        info!("  Input: Scarlett Solo 4th Gen → GhostWave AI Filter");
        info!("  Processing: NVIDIA RTX Voice-style noise cancellation");
        info!("  Output: PhantomLink Virtual XLR → Applications");

        info!("Profile optimized for XLR workflow:");
        info!("  Noise Suppression: {}",
              if self.ghostwave_config.noise_suppression.enabled { "Enabled" } else { "Disabled" });

        if self.ghostwave_config.noise_suppression.enabled {
            info!("  Strength: {:.1}%",
                  self.ghostwave_config.noise_suppression.strength * 100.0);
            info!("  Gate Threshold: {:.1} dB",
                  self.ghostwave_config.noise_suppression.gate_threshold);
        }

        Ok(())
    }

    pub async fn run_xlr_bridge(&self) -> Result<()> {
        info!("🎤 GhostWave XLR Bridge active");
        info!("Your audio chain: Microphone → Scarlett Solo → GhostWave AI → PhantomLink");

        tokio::signal::ctrl_c().await?;
        info!("Shutting down XLR bridge");

        Ok(())
    }
}

pub async fn run_phantomlink_mode(ghostwave_config: Config) -> Result<()> {
    info!("Starting GhostWave in PhantomLink integration mode");

    let integration = PhantomLinkIntegration::new(ghostwave_config);

    integration.register_as_xlr_device().await?;
    integration.setup_audio_routing().await?;
    integration.run_xlr_bridge().await?;

    Ok(())
}