use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, debug, warn};

#[derive(Debug, Clone)]
struct AlsaCard {
    short_name: String,
    long_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AudioDevice {
    pub name: String,
    pub id: String,
    pub device_type: AudioDeviceType,
    pub channels: u8,
    pub sample_rates: Vec<u32>,
    pub supported_sample_rates: Vec<u32>,
    pub supported_buffer_sizes: Vec<usize>,
    pub vendor: String,
    pub model: String,
    pub is_xlr_interface: bool,
    pub recommended_profile: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AudioDeviceType {
    XlrInterface,
    UsbMicrophone,
    UsbAudio,
    Microphone,
    Headset,
    LineIn,
    Internal,
    BuiltIn,
    Virtual,
    Unknown,
}

#[derive(Clone)]
pub struct DeviceDetector {
    known_devices: HashMap<String, AudioDevice>,
}

impl DeviceDetector {
    pub fn new() -> Self {
        let mut detector = Self {
            known_devices: HashMap::new(),
        };
        detector.load_known_devices();
        detector
    }

    fn load_known_devices(&mut self) {
        // Focusrite Scarlett Solo 4th Gen
        self.known_devices.insert("scarlett_solo_4th".to_string(), AudioDevice {
            name: "Scarlett Solo USB".to_string(),
            id: "scarlett_solo_4th".to_string(),
            device_type: AudioDeviceType::XlrInterface,
            channels: 2,
            sample_rates: vec![44100, 48000, 88200, 96000, 176400, 192000],
            supported_sample_rates: vec![44100, 48000, 88200, 96000, 176400, 192000],
            supported_buffer_sizes: vec![64, 128, 256, 512, 1024],
            vendor: "Focusrite".to_string(),
            model: "Scarlett Solo 4th Gen".to_string(),
            is_xlr_interface: true,
            recommended_profile: "studio".to_string(),
        });

        // Focusrite Scarlett 2i2 4th Gen
        self.known_devices.insert("scarlett_2i2_4th".to_string(), AudioDevice {
            name: "Scarlett 2i2 USB".to_string(),
            id: "scarlett_2i2_4th".to_string(),
            device_type: AudioDeviceType::XlrInterface,
            channels: 2,
            sample_rates: vec![44100, 48000, 88200, 96000, 176400, 192000],
            supported_sample_rates: vec![44100, 48000, 88200, 96000, 176400, 192000],
            supported_buffer_sizes: vec![64, 128, 256, 512, 1024],
            vendor: "Focusrite".to_string(),
            model: "Scarlett 2i2 4th Gen".to_string(),
            is_xlr_interface: true,
            recommended_profile: "studio".to_string(),
        });

        // Add more known XLR interfaces
        self.known_devices.insert("behringer_u_phoria_um2".to_string(), AudioDevice {
            name: "U-PHORIA UM2".to_string(),
            id: "behringer_u_phoria_um2".to_string(),
            device_type: AudioDeviceType::XlrInterface,
            channels: 2,
            sample_rates: vec![44100, 48000],
            supported_sample_rates: vec![44100, 48000],
            supported_buffer_sizes: vec![128, 256, 512, 1024],
            vendor: "Behringer".to_string(),
            model: "U-PHORIA UM2".to_string(),
            is_xlr_interface: true,
            recommended_profile: "balanced".to_string(),
        });

        info!("Loaded {} known audio devices", self.known_devices.len());
    }

    pub async fn detect_devices(&self) -> Result<Vec<AudioDevice>> {
        let mut detected_devices = Vec::new();

        info!("🔍 Detecting audio devices...");

        // First, try to read from /proc/asound/cards for more detailed info
        let alsa_cards = self.read_alsa_cards().await;

        // Use cpal to enumerate devices
        let host = cpal::default_host();

        // Check input devices
        match host.input_devices() {
            Ok(input_devices) => {
                for device in input_devices {
                    if let Ok(name) = device.name() {
                        debug!("Found input device: {}", name);

                        // Try to match with ALSA card info first
                        let mut identified_device = None;
                        for alsa_card in &alsa_cards {
                            if name.to_lowercase().contains(&alsa_card.short_name.to_lowercase()) {
                                debug!("Matched ALSA card: {} -> {}", name, alsa_card.long_name);
                                identified_device = self.identify_device_from_alsa(alsa_card);
                                break;
                            }
                        }

                        // Fall back to regular identification
                        if identified_device.is_none() {
                            identified_device = self.identify_device(&name);
                        }

                        if let Some(known_device) = identified_device {
                            info!("✅ Identified: {} ({})", known_device.model, known_device.vendor);
                            detected_devices.push(known_device);
                        } else {
                            // Create generic device entry
                            let generic_device = AudioDevice {
                                name: name.clone(),
                                id: format!("generic_{}", name.replace(" ", "_").to_lowercase()),
                                device_type: AudioDeviceType::Unknown,
                                channels: 2, // Default assumption
                                sample_rates: vec![44100, 48000],
                                supported_sample_rates: vec![44100, 48000],
                                supported_buffer_sizes: vec![128, 256, 512, 1024],
                                vendor: "Unknown".to_string(),
                                model: name,
                                is_xlr_interface: false,
                                recommended_profile: "balanced".to_string(),
                            };
                            detected_devices.push(generic_device);
                        }
                    }
                }
            }
            Err(e) => warn!("Failed to enumerate input devices: {}", e),
        }

        Ok(detected_devices)
    }

    async fn read_alsa_cards(&self) -> Vec<AlsaCard> {
        let mut cards = Vec::new();

        if let Ok(content) = std::fs::read_to_string("/proc/asound/cards") {
            for line in content.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                // Parse lines like: " 3 [Gen            ]: USB-Audio - Scarlett Solo 4th Gen"
                if let Some(bracket_start) = line.find('[') {
                    if let Some(bracket_end) = line.find(']') {
                        if let Some(dash_pos) = line.rfind(" - ") {
                            let short_name = line[bracket_start + 1..bracket_end].trim().to_string();
                            let long_name = line[dash_pos + 3..].trim().to_string();
                            cards.push(AlsaCard {
                                short_name,
                                long_name,
                            });
                            debug!("Found ALSA card: {} -> {}", cards.last().unwrap().short_name, cards.last().unwrap().long_name);
                        }
                    }
                }
            }
        }

        cards
    }

    fn identify_device_from_alsa(&self, alsa_card: &AlsaCard) -> Option<AudioDevice> {
        let long_name_lower = alsa_card.long_name.to_lowercase();

        // Check for Scarlett Solo 4th Gen specifically
        if long_name_lower.contains("scarlett solo 4th gen") {
            info!("🎯 Found Scarlett Solo 4th Gen via ALSA: {}", alsa_card.long_name);
            return self.known_devices.get("scarlett_solo_4th").cloned();
        }

        // Check for other Scarlett devices
        if long_name_lower.contains("scarlett") {
            info!("🎤 Found Focusrite Scarlett device: {}", alsa_card.long_name);
            return Some(AudioDevice {
                name: alsa_card.long_name.clone(),
                id: format!("focusrite_{}", alsa_card.short_name.replace(" ", "_").to_lowercase()),
                device_type: AudioDeviceType::XlrInterface,
                channels: 2,
                sample_rates: vec![44100, 48000, 96000, 192000],
                supported_sample_rates: vec![44100, 48000, 96000, 192000],
                supported_buffer_sizes: vec![64, 128, 256, 512, 1024],
                vendor: "Focusrite".to_string(),
                model: alsa_card.long_name.clone(),
                is_xlr_interface: true,
                recommended_profile: "studio".to_string(),
            });
        }

        None
    }

    fn identify_device(&self, device_name: &str) -> Option<AudioDevice> {
        let name_lower = device_name.to_lowercase();

        // Check for Scarlett Solo 4th Gen - also check for CARD=Gen pattern
        if name_lower.contains("scarlett") && name_lower.contains("solo") {
            return self.known_devices.get("scarlett_solo_4th").cloned();
        }

        // Check for the ALSA device name pattern that maps to Scarlett Solo
        if name_lower.contains("card=gen") || name_lower == "gen" {
            info!("Detected Scarlett Solo 4th Gen via ALSA card mapping: {}", device_name);
            return self.known_devices.get("scarlett_solo_4th").cloned();
        }

        // Check for Scarlett 2i2 4th Gen
        if name_lower.contains("scarlett") && name_lower.contains("2i2") {
            return self.known_devices.get("scarlett_2i2_4th").cloned();
        }

        // Check for Behringer U-PHORIA UM2
        if name_lower.contains("u-phoria") || name_lower.contains("um2") {
            return self.known_devices.get("behringer_u_phoria_um2").cloned();
        }

        // Add more pattern matching for common XLR interfaces
        if name_lower.contains("focusrite") {
            info!("Detected Focusrite device: {}", device_name);
            // Return a generic Focusrite device
            return Some(AudioDevice {
                name: device_name.to_string(),
                id: format!("focusrite_{}", device_name.replace(" ", "_").to_lowercase()),
                device_type: AudioDeviceType::XlrInterface,
                channels: 2,
                sample_rates: vec![44100, 48000, 96000, 192000],
                supported_sample_rates: vec![44100, 48000, 96000, 192000],
                supported_buffer_sizes: vec![64, 128, 256, 512, 1024],
                vendor: "Focusrite".to_string(),
                model: device_name.to_string(),
                is_xlr_interface: true,
                recommended_profile: "studio".to_string(),
            });
        }

        None
    }

    pub async fn find_scarlett_solo_4th_gen(&self) -> Result<Option<AudioDevice>> {
        info!("🎤 Searching specifically for Scarlett Solo 4th Gen...");

        let devices = self.detect_devices().await?;

        for device in devices {
            if device.id == "scarlett_solo_4th" {
                info!("🎯 Found Scarlett Solo 4th Gen: {}", device.name);
                return Ok(Some(device));
            }
        }

        info!("⚠️  Scarlett Solo 4th Gen not found");
        Ok(None)
    }

    pub async fn get_optimal_config_for_device(&self, device: &AudioDevice) -> Result<crate::config::Config> {
        info!("Generating optimal config for: {} {}", device.vendor, device.model);

        let mut config = crate::config::Config::load(&device.recommended_profile)?;

        // Optimize for XLR interfaces
        if device.is_xlr_interface {
            info!("Optimizing for XLR interface workflow");

            // Use highest supported sample rate for studio quality
            if device.sample_rates.contains(&192000) {
                config.audio.sample_rate = 192000;
                config.audio.buffer_size = 256; // Larger buffer for high sample rates
            } else if device.sample_rates.contains(&96000) {
                config.audio.sample_rate = 96000;
                config.audio.buffer_size = 256;
            } else {
                config.audio.sample_rate = 48000;
                config.audio.buffer_size = 128;
            }

            // XLR-specific noise suppression settings
            config.noise_suppression.enabled = true;
            config.noise_suppression.strength = 0.6; // Moderate for XLR
            config.noise_suppression.gate_threshold = -45.0; // Lower threshold for condenser mics
            config.noise_suppression.release_time = 0.4; // Smoother release

            // Set device names
            config.audio.input_device = Some(device.name.clone());
        }

        Ok(config)
    }

    pub async fn auto_configure_for_phantomlink(&self) -> Result<Option<crate::config::Config>> {
        info!("🔧 Auto-configuring GhostWave for PhantomLink + XLR workflow");

        // First try to find Scarlett Solo 4th Gen specifically
        if let Some(device) = self.find_scarlett_solo_4th_gen().await? {
            info!("Using Scarlett Solo 4th Gen configuration");
            return Ok(Some(self.get_optimal_config_for_device(&device).await?));
        }

        // Fall back to any XLR interface
        let devices = self.detect_devices().await?;
        for device in devices {
            if device.is_xlr_interface {
                info!("Using {} {} as XLR interface", device.vendor, device.model);
                return Ok(Some(self.get_optimal_config_for_device(&device).await?));
            }
        }

        warn!("No XLR interface found - using default configuration");
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_device_detection() {
        let detector = DeviceDetector::new();
        let devices = detector.detect_devices().await.unwrap();

        // Should find at least one device (even if it's a virtual/default one)
        assert!(!devices.is_empty());
    }

    #[test]
    fn test_scarlett_identification() {
        let detector = DeviceDetector::new();

        let device = detector.identify_device("Scarlett Solo USB");
        assert!(device.is_some());

        let device = device.unwrap();
        assert_eq!(device.vendor, "Focusrite");
        assert_eq!(device.model, "Scarlett Solo 4th Gen");
        assert!(device.is_xlr_interface);
    }
}