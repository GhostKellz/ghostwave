//! # Latency Optimization Module
//!
//! Provides comprehensive latency optimization for achieving <15ms E2E latency
//! with support for both standard (48kHz/128f) and studio (96kHz/256f) configurations.

use anyhow::Result;
use std::time::Duration;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{info, warn, debug};

/// Latency optimization configuration
#[derive(Debug, Clone)]
pub struct LatencyConfig {
    /// Target end-to-end latency in milliseconds
    pub target_latency_ms: f32,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Buffer size in frames
    pub buffer_size: usize,
    /// Number of buffers for double/triple buffering
    pub buffer_count: usize,
    /// Enable aggressive optimizations
    pub aggressive_mode: bool,
    /// CPU core affinity
    pub cpu_affinity: Option<Vec<usize>>,
}

impl LatencyConfig {
    /// Standard low-latency configuration (48kHz, 128 frames, <15ms)
    pub fn standard() -> Self {
        Self {
            target_latency_ms: 15.0,
            sample_rate: 48000,
            buffer_size: 128,
            buffer_count: 2,
            aggressive_mode: true,
            cpu_affinity: None,
        }
    }

    /// Studio configuration (96kHz, 256 frames, optimal quality)
    pub fn studio() -> Self {
        Self {
            target_latency_ms: 15.0,
            sample_rate: 96000,
            buffer_size: 256,
            buffer_count: 2,
            aggressive_mode: true,
            cpu_affinity: None,
        }
    }

    /// Calculate actual latency in milliseconds
    pub fn calculate_latency(&self) -> f32 {
        let frames_total = self.buffer_size * self.buffer_count;
        (frames_total as f32 / self.sample_rate as f32) * 1000.0
    }

    /// Verify if configuration meets target latency
    pub fn meets_target(&self) -> bool {
        self.calculate_latency() <= self.target_latency_ms
    }

    /// Optimize buffer size for target latency
    pub fn optimize_buffer_size(&mut self) -> Result<()> {
        let target_frames = (self.sample_rate as f32 * self.target_latency_ms / 1000.0) as usize;

        // Find optimal power-of-2 buffer size
        let mut optimal_size = 32;
        while optimal_size * self.buffer_count < target_frames && optimal_size < 2048 {
            optimal_size *= 2;
        }

        // Adjust if we went over
        if optimal_size * self.buffer_count > target_frames && optimal_size > 32 {
            optimal_size /= 2;
        }

        self.buffer_size = optimal_size;

        let actual_latency = self.calculate_latency();
        if actual_latency > self.target_latency_ms {
            warn!("Cannot achieve {}ms latency with current settings. Actual: {:.2}ms",
                  self.target_latency_ms, actual_latency);
        } else {
            info!("Optimized for {:.2}ms latency with {} frame buffers at {}Hz",
                  actual_latency, self.buffer_size, self.sample_rate);
        }

        Ok(())
    }
}

/// Real-time priority manager with graceful fallback
pub struct RtPriorityManager {
    priority_set: AtomicBool,
    #[allow(dead_code)] // Reserved for priority restoration on drop
    original_priority: i32,
}

impl RtPriorityManager {
    pub fn new() -> Self {
        Self {
            priority_set: AtomicBool::new(false),
            original_priority: 0,
        }
    }

    /// Set real-time priority with graceful fallback
    pub fn set_realtime_priority(&self, priority: i32) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            use libc::{sched_param, sched_setscheduler, SCHED_FIFO};
            use std::mem;

            unsafe {
                // Save original priority
                let mut original_param: sched_param = mem::zeroed();
                let _original_policy = libc::sched_getscheduler(0);
                libc::sched_getparam(0, &mut original_param);

                // Try to set FIFO real-time priority
                let param = sched_param {
                    sched_priority: priority.clamp(1, 99),
                };

                let result = sched_setscheduler(0, SCHED_FIFO, &param);

                if result == 0 {
                    self.priority_set.store(true, Ordering::Release);
                    info!("✅ Set real-time priority to {}", priority);
                    Ok(())
                } else {
                    // Graceful fallback to nice level
                    let nice_result = libc::nice(-20);
                    if nice_result >= 0 {
                        info!("⚠️ RT priority unavailable, using nice level -20");
                        Ok(())
                    } else {
                        warn!("❌ Cannot set RT priority or nice level. Running with normal priority.");
                        warn!("   Consider: sudo setcap cap_sys_nice+ep /path/to/ghostwave");
                        Ok(())
                    }
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            warn!("Real-time priority not supported on this platform");
            Ok(())
        }
    }

    /// Set CPU affinity for audio thread
    pub fn set_cpu_affinity(&self, cpu_cores: &[usize]) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            use libc::{cpu_set_t, CPU_SET, CPU_ZERO, sched_setaffinity};
            use std::mem;

            unsafe {
                let mut cpu_set: cpu_set_t = mem::zeroed();
                CPU_ZERO(&mut cpu_set);

                for &core in cpu_cores {
                    CPU_SET(core, &mut cpu_set);
                }

                let result = sched_setaffinity(0, mem::size_of::<cpu_set_t>(), &cpu_set);

                if result == 0 {
                    info!("✅ Set CPU affinity to cores: {:?}", cpu_cores);
                } else {
                    warn!("⚠️ Failed to set CPU affinity");
                }
            }
        }

        Ok(())
    }

    /// Optimize thread for audio processing
    pub fn optimize_thread(&self) -> Result<()> {
        // Disable CPU frequency scaling if possible
        #[cfg(target_os = "linux")]
        {
            if let Ok(governor) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") {
                if !governor.trim().eq("performance") {
                    warn!("CPU governor is '{}', consider setting to 'performance' for lowest latency", governor.trim());
                }
            }
        }

        // Set thread name
        #[cfg(target_os = "linux")]
        unsafe {
            let name = b"ghostwave-audio\0";
            libc::pthread_setname_np(libc::pthread_self(), name.as_ptr() as *const libc::c_char);
        }

        // Prefault stack pages to avoid page faults during audio processing
        self.prefault_stack()?;

        // Lock memory to prevent swapping
        self.lock_memory()?;

        Ok(())
    }

    /// Prefault stack to avoid page faults
    fn prefault_stack(&self) -> Result<()> {
        const STACK_SIZE: usize = 1024 * 1024; // 1MB
        let mut dummy = vec![0u8; STACK_SIZE];

        // Touch all pages
        for i in (0..STACK_SIZE).step_by(4096) {
            dummy[i] = 1;
        }

        // Prevent optimization
        std::hint::black_box(&dummy);

        debug!("Prefaulted {}KB of stack", STACK_SIZE / 1024);
        Ok(())
    }

    /// Lock memory pages to prevent swapping
    fn lock_memory(&self) -> Result<()> {
        #[cfg(target_os = "linux")]
        unsafe {
            let result = libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE);
            if result == 0 {
                debug!("✅ Memory locked to prevent swapping");
            } else {
                debug!("⚠️ Could not lock memory (needs CAP_IPC_LOCK)");
            }
        }
        Ok(())
    }
}

/// Latency measurement and optimization engine
pub struct LatencyOptimizer {
    config: LatencyConfig,
    rt_manager: RtPriorityManager,
    measurements: Arc<LatencyMeasurements>,
}

#[derive(Debug)]
struct LatencyMeasurements {
    input_latency: AtomicU64,
    processing_latency: AtomicU64,
    output_latency: AtomicU64,
    total_latency: AtomicU64,
    measurement_count: AtomicU64,
}

impl LatencyOptimizer {
    pub fn new(config: LatencyConfig) -> Self {
        Self {
            config,
            rt_manager: RtPriorityManager::new(),
            measurements: Arc::new(LatencyMeasurements {
                input_latency: AtomicU64::new(0),
                processing_latency: AtomicU64::new(0),
                output_latency: AtomicU64::new(0),
                total_latency: AtomicU64::new(0),
                measurement_count: AtomicU64::new(0),
            }),
        }
    }

    /// Apply all optimizations
    pub fn optimize(&mut self) -> Result<()> {
        info!("🚀 Applying latency optimizations for <{}ms target", self.config.target_latency_ms);

        // Optimize buffer size
        self.config.optimize_buffer_size()?;

        // Set real-time priority
        self.rt_manager.set_realtime_priority(80)?;

        // Set CPU affinity if specified
        if let Some(ref cores) = self.config.cpu_affinity {
            self.rt_manager.set_cpu_affinity(cores)?;
        }

        // Optimize thread
        self.rt_manager.optimize_thread()?;

        // Apply kernel-level optimizations if available
        self.apply_kernel_optimizations()?;

        info!("✅ Latency optimizations applied. Expected latency: {:.2}ms",
              self.config.calculate_latency());

        Ok(())
    }

    /// Apply kernel-level optimizations
    fn apply_kernel_optimizations(&self) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            // Check for low-latency or RT kernel
            if let Ok(kernel) = std::fs::read_to_string("/proc/version") {
                if kernel.contains("PREEMPT_RT") || kernel.contains("lowlatency") {
                    info!("✅ Running on optimized kernel (RT/lowlatency)");
                } else {
                    debug!("Standard kernel detected. Consider RT kernel for best latency.");
                }
            }

            // Disable CPU idle states for lowest latency
            if self.config.aggressive_mode {
                self.configure_cpu_idle_states()?;
            }

            // Configure interrupt affinity
            self.configure_irq_affinity()?;
        }

        Ok(())
    }

    /// Configure CPU idle states for minimum latency
    fn configure_cpu_idle_states(&self) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            let idle_path = "/sys/devices/system/cpu/cpu0/cpuidle/state0/disable";
            if std::path::Path::new(idle_path).exists() {
                debug!("CPU idle state configuration available");
                // Note: Would need root to actually modify
            }
        }
        Ok(())
    }

    /// Configure IRQ affinity for audio devices
    fn configure_irq_affinity(&self) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            // This would require identifying audio device IRQs
            // and setting their affinity masks
            debug!("IRQ affinity configuration skipped (requires root)");
        }
        Ok(())
    }

    /// Measure and record latency
    pub fn measure_latency(&self, stage: LatencyStage, duration: Duration) {
        let micros = duration.as_micros() as u64;

        match stage {
            LatencyStage::Input => {
                self.measurements.input_latency.store(micros, Ordering::Relaxed);
            }
            LatencyStage::Processing => {
                self.measurements.processing_latency.store(micros, Ordering::Relaxed);
            }
            LatencyStage::Output => {
                self.measurements.output_latency.store(micros, Ordering::Relaxed);
            }
            LatencyStage::Total => {
                self.measurements.total_latency.store(micros, Ordering::Relaxed);
                self.measurements.measurement_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Get current latency statistics
    pub fn get_stats(&self) -> LatencyStats {
        LatencyStats {
            input_us: self.measurements.input_latency.load(Ordering::Relaxed),
            processing_us: self.measurements.processing_latency.load(Ordering::Relaxed),
            output_us: self.measurements.output_latency.load(Ordering::Relaxed),
            total_us: self.measurements.total_latency.load(Ordering::Relaxed),
            measurement_count: self.measurements.measurement_count.load(Ordering::Relaxed),
            target_ms: self.config.target_latency_ms,
            actual_ms: self.measurements.total_latency.load(Ordering::Relaxed) as f32 / 1000.0,
        }
    }

    /// Auto-tune for optimal latency
    pub fn auto_tune(&mut self) -> Result<()> {
        info!("🔧 Auto-tuning for optimal latency...");

        let mut best_config = self.config.clone();
        let mut best_latency = f32::MAX;

        // Test different buffer sizes
        for buffer_size in [32, 64, 128, 256, 512].iter() {
            self.config.buffer_size = *buffer_size;
            let latency = self.config.calculate_latency();

            if latency <= self.config.target_latency_ms && latency < best_latency {
                best_latency = latency;
                best_config = self.config.clone();
            }
        }

        self.config = best_config;
        info!("✅ Auto-tuned to {:.2}ms latency with {} frame buffers",
              best_latency, self.config.buffer_size);

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LatencyStage {
    Input,
    Processing,
    Output,
    Total,
}

#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub input_us: u64,
    pub processing_us: u64,
    pub output_us: u64,
    pub total_us: u64,
    pub measurement_count: u64,
    pub target_ms: f32,
    pub actual_ms: f32,
}

impl LatencyStats {
    pub fn is_meeting_target(&self) -> bool {
        self.actual_ms <= self.target_ms
    }

    pub fn report(&self) {
        info!("📊 Latency Statistics:");
        info!("  Input:      {:.2}ms", self.input_us as f32 / 1000.0);
        info!("  Processing: {:.2}ms", self.processing_us as f32 / 1000.0);
        info!("  Output:     {:.2}ms", self.output_us as f32 / 1000.0);
        info!("  Total:      {:.2}ms (target: {:.2}ms)", self.actual_ms, self.target_ms);

        if self.is_meeting_target() {
            info!("  ✅ Meeting target latency");
        } else {
            warn!("  ⚠️ Exceeding target by {:.2}ms", self.actual_ms - self.target_ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_calculation() {
        let config = LatencyConfig {
            target_latency_ms: 15.0,
            sample_rate: 48000,
            buffer_size: 128,
            buffer_count: 2,
            aggressive_mode: true,
            cpu_affinity: None,
        };

        let latency = config.calculate_latency();
        assert!(latency < 15.0);
        assert!(config.meets_target());
    }

    #[test]
    fn test_studio_config() {
        let config = LatencyConfig::studio();
        let latency = config.calculate_latency();

        // 256 frames * 2 buffers / 96000 Hz * 1000 = 5.33ms
        assert!(latency < 6.0);
        assert!(config.meets_target());
    }

    #[test]
    fn test_buffer_optimization() {
        let mut config = LatencyConfig::standard();
        config.target_latency_ms = 10.0;
        config.optimize_buffer_size().unwrap();

        assert!(config.meets_target());
    }
}