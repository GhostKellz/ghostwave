//! # Device Manager with Auto Selection and Hotplug Support
//!
//! Provides automatic audio device selection and hotplug detection with debouncing.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time;
use tracing::{info, warn, debug};
use crate::device_detection::{DeviceDetector, AudioDevice, AudioDeviceType};

/// Device selection criteria and preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSelectionConfig {
    /// Preferred device types in order of priority
    pub preferred_types: Vec<AudioDeviceType>,
    /// Minimum sample rate requirement
    pub min_sample_rate: u32,
    /// Maximum acceptable latency in milliseconds
    pub max_latency_ms: f32,
    /// Preferred vendor names (case-insensitive substrings)
    pub preferred_vendors: Vec<String>,
    /// Preferred device name patterns (case-insensitive substrings)
    pub preferred_names: Vec<String>,
    /// Enable XLR interface detection for professional audio
    pub prefer_xlr_interfaces: bool,
    /// Enable USB audio device preference
    pub prefer_usb_audio: bool,
    /// Hotplug debounce time in milliseconds
    pub hotplug_debounce_ms: u64,
    /// Auto-switch to better devices when available
    pub auto_switch: bool,
}

impl Default for DeviceSelectionConfig {
    fn default() -> Self {
        Self {
            preferred_types: vec![
                AudioDeviceType::XlrInterface,
                AudioDeviceType::UsbAudio,
                AudioDeviceType::Headset,
                AudioDeviceType::Microphone,
                AudioDeviceType::LineIn,
                AudioDeviceType::Internal,
            ],
            min_sample_rate: 44100,
            max_latency_ms: 20.0,
            preferred_vendors: vec![
                "Focusrite".to_string(),
                "PreSonus".to_string(),
                "Scarlett".to_string(),
                "Blue".to_string(),
                "Audio-Technica".to_string(),
                "Shure".to_string(),
                "Rode".to_string(),
                "Zoom".to_string(),
                "Behringer".to_string(),
            ],
            preferred_names: vec![
                "Solo".to_string(),
                "2i2".to_string(),
                "4i4".to_string(),
                "Yeti".to_string(),
                "Podcast".to_string(),
                "Studio".to_string(),
                "Pro".to_string(),
            ],
            prefer_xlr_interfaces: true,
            prefer_usb_audio: true,
            hotplug_debounce_ms: 1000,
            auto_switch: true,
        }
    }
}

/// Device scoring for automatic selection
#[derive(Debug, Clone)]
struct DeviceScore {
    device: AudioDevice,
    score: u32,
    reasons: Vec<String>,
}

impl DeviceScore {
    fn new(device: AudioDevice) -> Self {
        Self {
            device,
            score: 0,
            reasons: Vec::new(),
        }
    }

    fn add_score(&mut self, points: u32, reason: &str) {
        self.score += points;
        self.reasons.push(reason.to_string());
    }
}

/// Hotplug event types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HotplugEvent {
    DeviceConnected(AudioDevice),
    DeviceDisconnected(String), // device name
    DeviceChanged(AudioDevice),
}

/// Device manager with auto selection and hotplug support
pub struct DeviceManager {
    config: DeviceSelectionConfig,
    detector: DeviceDetector,
    current_device: Arc<Mutex<Option<AudioDevice>>>,
    last_scan: Arc<Mutex<Instant>>,
    pending_events: Arc<Mutex<Vec<(Instant, HotplugEvent)>>>,
    hotplug_callbacks: Arc<Mutex<Vec<Box<dyn Fn(&HotplugEvent) + Send + Sync>>>>,
    is_monitoring: Arc<Mutex<bool>>,
}

impl DeviceManager {
    pub fn new(config: DeviceSelectionConfig) -> Self {
        Self {
            config,
            detector: DeviceDetector::new(),
            current_device: Arc::new(Mutex::new(None)),
            last_scan: Arc::new(Mutex::new(Instant::now())),
            pending_events: Arc::new(Mutex::new(Vec::new())),
            hotplug_callbacks: Arc::new(Mutex::new(Vec::new())),
            is_monitoring: Arc::new(Mutex::new(false)),
        }
    }

    /// Select the best available device based on configuration
    pub async fn select_best_device(&self) -> Result<Option<AudioDevice>> {
        info!("🔍 Scanning for best audio device...");

        let devices = self.detector.detect_devices().await?;
        if devices.is_empty() {
            warn!("No audio devices detected");
            return Ok(None);
        }

        info!("Found {} audio devices", devices.len());

        let mut scored_devices: Vec<DeviceScore> = devices
            .into_iter()
            .map(DeviceScore::new)
            .collect();

        // Score devices based on preferences
        for score in &mut scored_devices {
            self.score_device(score);
        }

        // Sort by score (highest first)
        scored_devices.sort_by(|a, b| b.score.cmp(&a.score));

        if let Some(best) = scored_devices.first() {
            info!("🏆 Selected device: {} {} (score: {})",
                  best.device.vendor, best.device.model, best.score);
            for reason in &best.reasons {
                debug!("  • {}", reason);
            }

            let selected = best.device.clone();
            *self.current_device.lock().unwrap() = Some(selected.clone());
            Ok(Some(selected))
        } else {
            warn!("No suitable devices found");
            Ok(None)
        }
    }

    /// Score a device based on selection criteria
    fn score_device(&self, score: &mut DeviceScore) {
        let device = &score.device;

        // Base score for functionality
        score.add_score(10, "Base functionality");

        // Type preference scoring
        for (index, preferred_type) in self.config.preferred_types.iter().enumerate() {
            if device.device_type == *preferred_type {
                let type_score = 100 - (index * 10) as u32;
                score.add_score(type_score, &format!("Preferred device type: {:?}", preferred_type));
                break;
            }
        }

        // XLR interface bonus
        if device.is_xlr_interface && self.config.prefer_xlr_interfaces {
            score.add_score(50, "XLR professional interface");
        }

        // USB audio preference
        if device.name.to_lowercase().contains("usb") && self.config.prefer_usb_audio {
            score.add_score(30, "USB audio device");
        }

        // Vendor preference
        for preferred_vendor in &self.config.preferred_vendors {
            if device.vendor.to_lowercase().contains(&preferred_vendor.to_lowercase()) {
                score.add_score(40, &format!("Preferred vendor: {}", preferred_vendor));
                break;
            }
        }

        // Name preference
        for preferred_name in &self.config.preferred_names {
            if device.name.to_lowercase().contains(&preferred_name.to_lowercase()) ||
               device.model.to_lowercase().contains(&preferred_name.to_lowercase()) {
                score.add_score(30, &format!("Preferred name pattern: {}", preferred_name));
                break;
            }
        }

        // Sample rate capability
        if let Some(max_sr) = device.supported_sample_rates.iter().max() {
            if *max_sr >= 96000 {
                score.add_score(25, "High sample rate support (96kHz+)");
            } else if *max_sr >= 48000 {
                score.add_score(15, "Standard sample rate support (48kHz+)");
            }

            if *max_sr < self.config.min_sample_rate {
                score.add_score(0, "Below minimum sample rate requirement");
                return; // Don't score further if it doesn't meet requirements
            }
        }

        // Channel count
        match device.channels {
            1 => score.add_score(10, "Mono input"),
            2 => score.add_score(20, "Stereo input"),
            n if n > 2 => score.add_score(15, &format!("{} channel input", n)),
            _ => {}
        }

        // Latency estimate based on buffer size
        if let Some(min_buffer) = device.supported_buffer_sizes.iter().min() {
            let estimated_latency = (*min_buffer as f32 / 48000.0) * 1000.0; // Rough estimate at 48kHz
            if estimated_latency <= self.config.max_latency_ms {
                let latency_score = ((self.config.max_latency_ms - estimated_latency) * 2.0) as u32;
                score.add_score(latency_score.min(20), &format!("Low latency: {:.1}ms", estimated_latency));
            }
        }

        // Known good devices
        let device_key = format!("{} {}", device.vendor, device.model).to_lowercase();
        if device_key.contains("focusrite") && device_key.contains("scarlett") {
            score.add_score(60, "Focusrite Scarlett (known excellent)");
        } else if device_key.contains("presonus") {
            score.add_score(50, "PreSonus interface (known good)");
        } else if device_key.contains("blue") && device_key.contains("yeti") {
            score.add_score(45, "Blue Yeti (popular choice)");
        } else if device_key.contains("rode") {
            score.add_score(40, "Rode microphone (quality brand)");
        }
    }

    /// Get the currently selected device
    pub fn current_device(&self) -> Option<AudioDevice> {
        self.current_device.lock().unwrap().clone()
    }

    /// Register a callback for hotplug events
    pub fn add_hotplug_callback<F>(&self, callback: F)
    where
        F: Fn(&HotplugEvent) + Send + Sync + 'static,
    {
        self.hotplug_callbacks.lock().unwrap().push(Box::new(callback));
    }

    /// Start monitoring for device changes
    pub async fn start_monitoring(&self) -> Result<()> {
        if *self.is_monitoring.lock().unwrap() {
            return Ok(());
        }

        *self.is_monitoring.lock().unwrap() = true;
        info!("🔍 Starting device hotplug monitoring");

        let config = self.config.clone();
        let detector = self.detector.clone();
        let current_device = Arc::clone(&self.current_device);
        let pending_events = Arc::clone(&self.pending_events);
        let hotplug_callbacks = Arc::clone(&self.hotplug_callbacks);
        let is_monitoring = Arc::clone(&self.is_monitoring);

        tokio::spawn(async move {
            let mut last_devices: HashMap<String, AudioDevice> = HashMap::new();
            let mut check_interval = time::interval(Duration::from_millis(500));

            while *is_monitoring.lock().unwrap() {
                check_interval.tick().await;

                // Detect current devices
                if let Ok(devices) = detector.detect_devices().await {
                    let mut current_devices: HashMap<String, AudioDevice> = HashMap::new();

                    for device in devices {
                        current_devices.insert(device.name.clone(), device);
                    }

                    // Check for new devices
                    for (name, device) in &current_devices {
                        if !last_devices.contains_key(name) {
                            let event = HotplugEvent::DeviceConnected(device.clone());
                            pending_events.lock().unwrap().push((Instant::now(), event));
                            debug!("Device connected: {}", name);
                        }
                    }

                    // Check for removed devices
                    for name in last_devices.keys() {
                        if !current_devices.contains_key(name) {
                            let event = HotplugEvent::DeviceDisconnected(name.clone());
                            pending_events.lock().unwrap().push((Instant::now(), event));
                            debug!("Device disconnected: {}", name);
                        }
                    }

                    last_devices = current_devices;
                }

                // Process debounced events
                let now = Instant::now();
                let debounce_duration = Duration::from_millis(config.hotplug_debounce_ms);
                let mut events_to_process = Vec::new();

                {
                    let mut pending = pending_events.lock().unwrap();
                    pending.retain(|(timestamp, event)| {
                        if now.duration_since(*timestamp) >= debounce_duration {
                            events_to_process.push(event.clone());
                            false // Remove from pending
                        } else {
                            true // Keep in pending
                        }
                    });
                }

                // Process events and notify callbacks
                for event in events_to_process {
                    info!("Processing hotplug event: {:?}", event);

                    // Handle auto-switching
                    if config.auto_switch {
                        if let HotplugEvent::DeviceConnected(ref new_device) = event {
                            if let Some(current) = current_device.lock().unwrap().as_ref() {
                                // Check if the new device would score higher
                                let mut current_score = DeviceScore::new(current.clone());
                                Self::score_device_static(&config, &mut current_score);

                                let mut new_score = DeviceScore::new(new_device.clone());
                                Self::score_device_static(&config, &mut new_score);

                                if new_score.score > current_score.score + 20 {
                                    info!("🔄 Auto-switching to better device: {} {} (score: {} vs {})",
                                          new_device.vendor, new_device.model,
                                          new_score.score, current_score.score);
                                    *current_device.lock().unwrap() = Some(new_device.clone());
                                }
                            } else {
                                // No current device, auto-select this one if it's good
                                let mut score = DeviceScore::new(new_device.clone());
                                Self::score_device_static(&config, &mut score);

                                if score.score >= 50 {
                                    info!("🎯 Auto-selecting new device: {} {} (score: {})",
                                          new_device.vendor, new_device.model, score.score);
                                    *current_device.lock().unwrap() = Some(new_device.clone());
                                }
                            }
                        } else if let HotplugEvent::DeviceDisconnected(ref name) = event {
                            if let Some(current) = current_device.lock().unwrap().as_ref() {
                                if current.name == *name {
                                    warn!("⚠️ Current device disconnected, will need to re-select");
                                    *current_device.lock().unwrap() = None;
                                }
                            }
                        }
                    }

                    // Notify callbacks
                    let callbacks = hotplug_callbacks.lock().unwrap();
                    for callback in callbacks.iter() {
                        callback(&event);
                    }
                }
            }

            info!("Device monitoring stopped");
        });

        Ok(())
    }

    /// Stop monitoring for device changes
    pub fn stop_monitoring(&self) {
        *self.is_monitoring.lock().unwrap() = false;
        info!("Stopping device hotplug monitoring");
    }

    /// Force a device rescan and reselection
    pub async fn rescan_and_select(&self) -> Result<Option<AudioDevice>> {
        info!("🔄 Forcing device rescan and reselection");
        self.select_best_device().await
    }

    /// Get all currently available devices with scores
    pub async fn get_scored_devices(&self) -> Result<Vec<(AudioDevice, u32, Vec<String>)>> {
        let devices = self.detector.detect_devices().await?;

        let mut scored_devices: Vec<DeviceScore> = devices
            .into_iter()
            .map(DeviceScore::new)
            .collect();

        for score in &mut scored_devices {
            self.score_device(score);
        }

        Ok(scored_devices.into_iter()
            .map(|s| (s.device, s.score, s.reasons))
            .collect())
    }

    /// Static scoring method for use in async context
    fn score_device_static(config: &DeviceSelectionConfig, score: &mut DeviceScore) {
        let device = &score.device;

        score.add_score(10, "Base functionality");

        for (index, preferred_type) in config.preferred_types.iter().enumerate() {
            if device.device_type == *preferred_type {
                let type_score = 100 - (index * 10) as u32;
                score.add_score(type_score, &format!("Preferred device type: {:?}", preferred_type));
                break;
            }
        }

        if device.is_xlr_interface && config.prefer_xlr_interfaces {
            score.add_score(50, "XLR professional interface");
        }

        for preferred_vendor in &config.preferred_vendors {
            if device.vendor.to_lowercase().contains(&preferred_vendor.to_lowercase()) {
                score.add_score(40, &format!("Preferred vendor: {}", preferred_vendor));
                break;
            }
        }
    }
}

/// Device manager configuration builder
pub struct DeviceManagerBuilder {
    config: DeviceSelectionConfig,
}

impl DeviceManagerBuilder {
    pub fn new() -> Self {
        Self {
            config: DeviceSelectionConfig::default(),
        }
    }

    pub fn prefer_xlr(mut self, prefer: bool) -> Self {
        self.config.prefer_xlr_interfaces = prefer;
        self
    }

    pub fn prefer_usb(mut self, prefer: bool) -> Self {
        self.config.prefer_usb_audio = prefer;
        self
    }

    pub fn auto_switch(mut self, auto: bool) -> Self {
        self.config.auto_switch = auto;
        self
    }

    pub fn debounce_ms(mut self, ms: u64) -> Self {
        self.config.hotplug_debounce_ms = ms;
        self
    }

    pub fn min_sample_rate(mut self, rate: u32) -> Self {
        self.config.min_sample_rate = rate;
        self
    }

    pub fn max_latency(mut self, latency_ms: f32) -> Self {
        self.config.max_latency_ms = latency_ms;
        self
    }

    pub fn add_preferred_vendor(mut self, vendor: &str) -> Self {
        self.config.preferred_vendors.push(vendor.to_string());
        self
    }

    pub fn add_preferred_name(mut self, name: &str) -> Self {
        self.config.preferred_names.push(name.to_string());
        self
    }

    pub fn build(self) -> DeviceManager {
        DeviceManager::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_device_manager_creation() {
        let manager = DeviceManagerBuilder::new()
            .prefer_xlr(true)
            .auto_switch(true)
            .debounce_ms(500)
            .build();

        // Should be able to select devices even if none available
        let result = manager.select_best_device().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_device_scoring() {
        let config = DeviceSelectionConfig::default();

        let test_device = AudioDevice {
            name: "Focusrite Scarlett Solo 4th Gen".to_string(),
            vendor: "Focusrite".to_string(),
            model: "Scarlett Solo".to_string(),
            device_type: AudioDeviceType::XlrInterface,
            is_xlr_interface: true,
            channels: 2,
            supported_sample_rates: vec![44100, 48000, 96000, 192000],
            supported_buffer_sizes: vec![32, 64, 128, 256, 512],
            recommended_profile: "studio".to_string(),
        };

        let mut score = DeviceScore::new(test_device);
        DeviceManager::score_device_static(&config, &mut score);

        // Should get high score for XLR interface + Focusrite brand + high sample rates
        assert!(score.score > 200);
        assert!(!score.reasons.is_empty());
    }

    #[test]
    fn test_builder_pattern() {
        let manager = DeviceManagerBuilder::new()
            .prefer_xlr(false)
            .prefer_usb(true)
            .auto_switch(false)
            .debounce_ms(2000)
            .min_sample_rate(48000)
            .max_latency(10.0)
            .add_preferred_vendor("Blue")
            .add_preferred_name("Yeti")
            .build();

        assert_eq!(manager.config.prefer_xlr_interfaces, false);
        assert_eq!(manager.config.prefer_usb_audio, true);
        assert_eq!(manager.config.auto_switch, false);
        assert_eq!(manager.config.hotplug_debounce_ms, 2000);
        assert!(manager.config.preferred_vendors.contains(&"Blue".to_string()));
    }
}