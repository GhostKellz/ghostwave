use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub profile: ProfileConfig,
    pub audio: AudioConfig,
    pub noise_suppression: NoiseSuppressionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub buffer_size: u32,
    pub channels: u8,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseSuppressionConfig {
    pub enabled: bool,
    pub strength: f32,
    pub gate_threshold: f32,
    pub release_time: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self::balanced()
    }
}

impl Config {
    pub fn load(profile_name: &str) -> Result<Self> {
        let mut config = match profile_name {
            "balanced" => Self::balanced(),
            "streaming" => Self::streaming(),
            "studio" => Self::studio(),
            _ => {
                let config_path = Self::config_path(profile_name)?;
                let content = std::fs::read_to_string(&config_path)
                    .with_context(|| format!("Failed to read config from {:?}", config_path))?;
                serde_json::from_str(&content)
                    .with_context(|| "Failed to parse config JSON")?
            }
        };

        // Apply environment variable overrides
        if let Ok(sample_rate) = std::env::var("GHOSTWAVE_SAMPLE_RATE") {
            if let Ok(rate) = sample_rate.parse::<u32>() {
                config.audio.sample_rate = rate;
            }
        }

        if let Ok(buffer_size) = std::env::var("GHOSTWAVE_FRAMES") {
            if let Ok(frames) = buffer_size.parse::<u32>() {
                config.audio.buffer_size = frames;
            }
        }

        if let Ok(channels) = std::env::var("GHOSTWAVE_CHANNELS") {
            if let Ok(ch) = channels.parse::<u8>() {
                config.audio.channels = ch;
            }
        }

        Ok(config)
    }

    pub fn with_overrides(mut self, sample_rate: Option<u32>, frames: Option<u32>) -> Self {
        if let Some(rate) = sample_rate {
            self.audio.sample_rate = rate;
        }
        if let Some(buffer_size) = frames {
            self.audio.buffer_size = buffer_size;
        }
        self
    }

    fn config_path(profile_name: &str) -> Result<PathBuf> {
        let mut path = dirs::config_dir()
            .context("Failed to get config directory")?;
        path.push("ghostwave");
        path.push(format!("{}.json", profile_name));
        Ok(path)
    }

    fn balanced() -> Self {
        Self {
            profile: ProfileConfig {
                name: "Balanced".to_string(),
                description: "48kHz, 128 frames, 2ch".to_string(),
            },
            audio: AudioConfig {
                sample_rate: 48000,
                buffer_size: 128,
                channels: 2,
                input_device: None,
                output_device: None,
            },
            noise_suppression: NoiseSuppressionConfig {
                enabled: true,
                strength: 0.7,
                gate_threshold: -40.0,
                release_time: 0.3,
            },
        }
    }

    fn streaming() -> Self {
        Self {
            profile: ProfileConfig {
                name: "Streaming".to_string(),
                description: "48kHz, 128 frames, 1-2ch, extra denoise stage".to_string(),
            },
            audio: AudioConfig {
                sample_rate: 48000,
                buffer_size: 128,
                channels: 2,
                input_device: None,
                output_device: None,
            },
            noise_suppression: NoiseSuppressionConfig {
                enabled: true,
                strength: 0.85,
                gate_threshold: -35.0,
                release_time: 0.2,
            },
        }
    }

    fn studio() -> Self {
        Self {
            profile: ProfileConfig {
                name: "Studio".to_string(),
                description: "96kHz, 256 frames, 2ch, gentler gate".to_string(),
            },
            audio: AudioConfig {
                sample_rate: 96000,
                buffer_size: 256,
                channels: 2,
                input_device: None,
                output_device: None,
            },
            noise_suppression: NoiseSuppressionConfig {
                enabled: true,
                strength: 0.5,
                gate_threshold: -50.0,
                release_time: 0.5,
            },
        }
    }
}