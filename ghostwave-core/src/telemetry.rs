//! # Telemetry Module
//!
//! Real-time metrics collection and exposure for GhostWave.
//! Provides performance, GPU, and audio metrics for monitoring and debugging.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Maximum number of latency samples to keep
const LATENCY_SAMPLE_WINDOW: usize = 1000;

/// Performance metrics collected in real-time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average processing latency in microseconds
    pub avg_latency_us: f64,
    /// 99th percentile latency in microseconds
    pub p99_latency_us: f64,
    /// Maximum latency observed in microseconds
    pub max_latency_us: f64,
    /// Minimum latency observed in microseconds
    pub min_latency_us: f64,
    /// Number of XRuns (buffer underruns)
    pub xrun_count: u64,
    /// XRun rate as percentage
    pub xrun_rate_pct: f64,
    /// Total frames processed
    pub frames_processed: u64,
    /// Estimated CPU usage percentage
    pub cpu_usage_pct: f32,
    /// Buffer utilization percentage
    pub buffer_utilization_pct: f32,
    /// Current sample rate
    pub sample_rate: u32,
    /// Current buffer size in frames
    pub buffer_size: usize,
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

/// GPU/RTX specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// GPU is available for acceleration
    pub available: bool,
    /// GPU name/model
    pub gpu_name: String,
    /// Compute capability (major.minor)
    pub compute_capability: String,
    /// GPU architecture generation
    pub architecture: String,
    /// GPU memory total in GB
    pub memory_total_gb: f32,
    /// GPU memory used in GB
    pub memory_used_gb: f32,
    /// GPU utilization percentage (if available)
    pub gpu_utilization_pct: Option<f32>,
    /// Tensor Core generation
    pub tensor_core_gen: u8,
    /// FP4 support (Blackwell only)
    pub supports_fp4: bool,
    /// RTX Voice compatible
    pub rtx_voice_compatible: bool,
    /// Current processing mode
    pub processing_mode: String,
}

/// Audio level metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetrics {
    /// Input level in dB
    pub input_level_db: f32,
    /// Output level in dB
    pub output_level_db: f32,
    /// Peak input level in dB
    pub peak_input_db: f32,
    /// Peak output level in dB
    pub peak_output_db: f32,
    /// Noise reduction amount in dB
    pub noise_reduction_db: f32,
    /// Voice activity detected
    pub voice_active: bool,
    /// Gate is currently active
    pub gate_active: bool,
    /// Current noise floor estimate in dB
    pub noise_floor_db: f32,
}

/// Combined telemetry snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    /// Timestamp of snapshot
    pub timestamp: u64,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// GPU metrics
    pub gpu: GpuMetrics,
    /// Audio metrics
    pub audio: AudioMetrics,
    /// Current profile name
    pub active_profile: String,
    /// Processing is enabled
    pub processing_enabled: bool,
}

/// Telemetry collector for real-time metrics
pub struct TelemetryCollector {
    /// Latency samples ring buffer
    latency_samples: RwLock<VecDeque<u64>>,
    /// Xrun counter
    xrun_count: AtomicU64,
    /// Total frames processed
    frames_processed: AtomicU64,
    /// Start time
    start_time: Instant,
    /// Processing enabled flag
    processing_enabled: AtomicBool,
    /// Current sample rate
    sample_rate: AtomicU64,
    /// Current buffer size
    buffer_size: AtomicU64,
    /// Active profile name
    active_profile: RwLock<String>,
    /// Audio levels (input_db, output_db, peak_in, peak_out)
    audio_levels: RwLock<(f32, f32, f32, f32)>,
    /// Voice active flag
    voice_active: AtomicBool,
    /// Gate active flag
    gate_active: AtomicBool,
    /// Current noise floor estimate
    noise_floor_db: RwLock<f32>,
    /// Noise reduction amount
    noise_reduction_db: RwLock<f32>,
    /// GPU info cache
    gpu_info: RwLock<Option<GpuMetrics>>,
}

impl TelemetryCollector {
    /// Create a new telemetry collector
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            latency_samples: RwLock::new(VecDeque::with_capacity(LATENCY_SAMPLE_WINDOW)),
            xrun_count: AtomicU64::new(0),
            frames_processed: AtomicU64::new(0),
            start_time: Instant::now(),
            processing_enabled: AtomicBool::new(true),
            sample_rate: AtomicU64::new(48000),
            buffer_size: AtomicU64::new(256),
            active_profile: RwLock::new("balanced".to_string()),
            audio_levels: RwLock::new((-60.0, -60.0, -60.0, -60.0)),
            voice_active: AtomicBool::new(false),
            gate_active: AtomicBool::new(false),
            noise_floor_db: RwLock::new(-60.0),
            noise_reduction_db: RwLock::new(0.0),
            gpu_info: RwLock::new(None),
        })
    }

    /// Record a frame processing latency sample (in microseconds)
    pub fn record_latency(&self, latency_us: u64) {
        if let Ok(mut samples) = self.latency_samples.write() {
            if samples.len() >= LATENCY_SAMPLE_WINDOW {
                samples.pop_front();
            }
            samples.push_back(latency_us);
        }
    }

    /// Record an XRun event
    pub fn record_xrun(&self) {
        self.xrun_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record frames processed
    pub fn record_frames(&self, count: u64) {
        self.frames_processed.fetch_add(count, Ordering::Relaxed);
    }

    /// Update audio config
    pub fn update_config(&self, sample_rate: u32, buffer_size: usize) {
        self.sample_rate.store(sample_rate as u64, Ordering::Relaxed);
        self.buffer_size.store(buffer_size as u64, Ordering::Relaxed);
    }

    /// Update active profile
    pub fn set_profile(&self, profile: &str) {
        if let Ok(mut p) = self.active_profile.write() {
            *p = profile.to_string();
        }
    }

    /// Update audio levels (in dB)
    pub fn update_audio_levels(&self, input_db: f32, output_db: f32) {
        if let Ok(mut levels) = self.audio_levels.write() {
            levels.0 = input_db;
            levels.1 = output_db;
            // Update peaks
            if input_db > levels.2 {
                levels.2 = input_db;
            }
            if output_db > levels.3 {
                levels.3 = output_db;
            }
        }
    }

    /// Reset peak levels
    pub fn reset_peaks(&self) {
        if let Ok(mut levels) = self.audio_levels.write() {
            levels.2 = -60.0;
            levels.3 = -60.0;
        }
    }

    /// Update voice activity status
    pub fn set_voice_active(&self, active: bool) {
        self.voice_active.store(active, Ordering::Relaxed);
    }

    /// Update gate status
    pub fn set_gate_active(&self, active: bool) {
        self.gate_active.store(active, Ordering::Relaxed);
    }

    /// Update noise reduction metrics
    pub fn update_noise_reduction(&self, noise_floor_db: f32, reduction_db: f32) {
        if let Ok(mut nf) = self.noise_floor_db.write() {
            *nf = noise_floor_db;
        }
        if let Ok(mut nr) = self.noise_reduction_db.write() {
            *nr = reduction_db;
        }
    }

    /// Set/update GPU info
    pub fn set_gpu_info(&self, gpu: GpuMetrics) {
        if let Ok(mut info) = self.gpu_info.write() {
            *info = Some(gpu);
        }
    }

    /// Set processing enabled state
    pub fn set_processing_enabled(&self, enabled: bool) {
        self.processing_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        let (avg, p99, max, min) = self.calculate_latency_stats();
        let frames = self.frames_processed.load(Ordering::Relaxed);
        let xruns = self.xrun_count.load(Ordering::Relaxed);
        let sample_rate = self.sample_rate.load(Ordering::Relaxed) as u32;
        let buffer_size = self.buffer_size.load(Ordering::Relaxed) as usize;
        let uptime = self.start_time.elapsed().as_secs();

        // Estimate CPU usage based on processing time vs frame time
        let frame_time_us = (buffer_size as f64 / sample_rate as f64) * 1_000_000.0;
        let cpu_usage = if frame_time_us > 0.0 {
            ((avg / frame_time_us) * 100.0) as f32
        } else {
            0.0
        };

        let xrun_rate = if frames > 0 {
            (xruns as f64 / frames as f64) * 100.0
        } else {
            0.0
        };

        PerformanceMetrics {
            avg_latency_us: avg,
            p99_latency_us: p99,
            max_latency_us: max,
            min_latency_us: min,
            xrun_count: xruns,
            xrun_rate_pct: xrun_rate,
            frames_processed: frames,
            cpu_usage_pct: cpu_usage,
            buffer_utilization_pct: cpu_usage.min(100.0),
            sample_rate,
            buffer_size,
            uptime_seconds: uptime,
        }
    }

    /// Get GPU metrics
    pub fn get_gpu_metrics(&self) -> GpuMetrics {
        if let Ok(info) = self.gpu_info.read() {
            if let Some(gpu) = info.as_ref() {
                return gpu.clone();
            }
        }

        // Return default "not available" metrics
        GpuMetrics {
            available: false,
            gpu_name: "Not detected".to_string(),
            compute_capability: "N/A".to_string(),
            architecture: "N/A".to_string(),
            memory_total_gb: 0.0,
            memory_used_gb: 0.0,
            gpu_utilization_pct: None,
            tensor_core_gen: 0,
            supports_fp4: false,
            rtx_voice_compatible: false,
            processing_mode: "CPU".to_string(),
        }
    }

    /// Get audio metrics
    pub fn get_audio_metrics(&self) -> AudioMetrics {
        let (input, output, peak_in, peak_out) = self.audio_levels
            .read()
            .map(|l| *l)
            .unwrap_or((-60.0, -60.0, -60.0, -60.0));

        let noise_floor = self.noise_floor_db
            .read()
            .map(|n| *n)
            .unwrap_or(-60.0);

        let noise_reduction = self.noise_reduction_db
            .read()
            .map(|n| *n)
            .unwrap_or(0.0);

        AudioMetrics {
            input_level_db: input,
            output_level_db: output,
            peak_input_db: peak_in,
            peak_output_db: peak_out,
            noise_reduction_db: noise_reduction,
            voice_active: self.voice_active.load(Ordering::Relaxed),
            gate_active: self.gate_active.load(Ordering::Relaxed),
            noise_floor_db: noise_floor,
        }
    }

    /// Get complete telemetry snapshot
    pub fn snapshot(&self) -> TelemetrySnapshot {
        let profile = self.active_profile
            .read()
            .map(|p| p.clone())
            .unwrap_or_else(|_| "unknown".to_string());

        TelemetrySnapshot {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            performance: self.get_performance_metrics(),
            gpu: self.get_gpu_metrics(),
            audio: self.get_audio_metrics(),
            active_profile: profile,
            processing_enabled: self.processing_enabled.load(Ordering::Relaxed),
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.xrun_count.store(0, Ordering::Relaxed);
        self.frames_processed.store(0, Ordering::Relaxed);
        if let Ok(mut samples) = self.latency_samples.write() {
            samples.clear();
        }
        self.reset_peaks();
    }

    /// Calculate latency statistics from samples
    fn calculate_latency_stats(&self) -> (f64, f64, f64, f64) {
        if let Ok(samples) = self.latency_samples.read() {
            if samples.is_empty() {
                return (0.0, 0.0, 0.0, 0.0);
            }

            let mut sorted: Vec<u64> = samples.iter().copied().collect();
            sorted.sort_unstable();

            let sum: u64 = sorted.iter().sum();
            let avg = sum as f64 / sorted.len() as f64;
            let min = *sorted.first().unwrap_or(&0) as f64;
            let max = *sorted.last().unwrap_or(&0) as f64;

            // P99
            let p99_idx = (sorted.len() as f64 * 0.99) as usize;
            let p99 = sorted.get(p99_idx.min(sorted.len() - 1))
                .copied()
                .unwrap_or(0) as f64;

            (avg, p99, max, min)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        }
    }
}

impl Default for TelemetryCollector {
    fn default() -> Self {
        Arc::try_unwrap(Self::new()).unwrap_or_else(|arc| (*arc).clone())
    }
}

impl Clone for TelemetryCollector {
    fn clone(&self) -> Self {
        Self {
            latency_samples: RwLock::new(
                self.latency_samples.read().map(|s| s.clone()).unwrap_or_default()
            ),
            xrun_count: AtomicU64::new(self.xrun_count.load(Ordering::Relaxed)),
            frames_processed: AtomicU64::new(self.frames_processed.load(Ordering::Relaxed)),
            start_time: self.start_time,
            processing_enabled: AtomicBool::new(self.processing_enabled.load(Ordering::Relaxed)),
            sample_rate: AtomicU64::new(self.sample_rate.load(Ordering::Relaxed)),
            buffer_size: AtomicU64::new(self.buffer_size.load(Ordering::Relaxed)),
            active_profile: RwLock::new(
                self.active_profile.read().map(|p| p.clone()).unwrap_or_default()
            ),
            audio_levels: RwLock::new(
                self.audio_levels.read().map(|l| *l).unwrap_or((-60.0, -60.0, -60.0, -60.0))
            ),
            voice_active: AtomicBool::new(self.voice_active.load(Ordering::Relaxed)),
            gate_active: AtomicBool::new(self.gate_active.load(Ordering::Relaxed)),
            noise_floor_db: RwLock::new(
                self.noise_floor_db.read().map(|n| *n).unwrap_or(-60.0)
            ),
            noise_reduction_db: RwLock::new(
                self.noise_reduction_db.read().map(|n| *n).unwrap_or(0.0)
            ),
            gpu_info: RwLock::new(
                self.gpu_info.read().map(|g| g.clone()).unwrap_or(None)
            ),
        }
    }
}

/// Global telemetry instance
static GLOBAL_TELEMETRY: std::sync::OnceLock<Arc<TelemetryCollector>> = std::sync::OnceLock::new();

/// Initialize global telemetry collector
pub fn init_telemetry() -> Arc<TelemetryCollector> {
    GLOBAL_TELEMETRY.get_or_init(TelemetryCollector::new).clone()
}

/// Get global telemetry collector
pub fn telemetry() -> Option<Arc<TelemetryCollector>> {
    GLOBAL_TELEMETRY.get().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_collector() {
        let collector = TelemetryCollector::new();

        // Record some latencies
        for i in 0..100 {
            collector.record_latency(100 + i);
        }
        collector.record_frames(100);
        collector.record_xrun();

        let metrics = collector.get_performance_metrics();
        assert!(metrics.avg_latency_us > 0.0);
        assert_eq!(metrics.xrun_count, 1);
        assert_eq!(metrics.frames_processed, 100);
    }

    #[test]
    fn test_audio_metrics() {
        let collector = TelemetryCollector::new();

        collector.update_audio_levels(-12.0, -15.0);
        collector.set_voice_active(true);

        let audio = collector.get_audio_metrics();
        assert_eq!(audio.input_level_db, -12.0);
        assert_eq!(audio.output_level_db, -15.0);
        assert!(audio.voice_active);
    }

    #[test]
    fn test_snapshot() {
        let collector = TelemetryCollector::new();
        collector.set_profile("studio");
        collector.update_config(96000, 128);

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.active_profile, "studio");
        assert_eq!(snapshot.performance.sample_rate, 96000);
        assert_eq!(snapshot.performance.buffer_size, 128);
    }
}
