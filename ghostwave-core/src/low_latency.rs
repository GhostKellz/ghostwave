use anyhow::Result;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{info, debug, warn};

/// Target latency for real-time audio processing
pub const TARGET_LATENCY_MS: u32 = 15;

/// Lock-free ring buffer for zero-copy audio processing
pub struct LockFreeAudioBuffer {
    buffer: Vec<f32>,
    capacity: usize,
    channels: u8,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
    sample_rate: u32,
}

impl LockFreeAudioBuffer {
    pub fn new(capacity_frames: usize, sample_rate: u32) -> Self {
        Self::new_with_channels(capacity_frames, sample_rate, 1)
    }

    pub fn new_with_channels(capacity_frames: usize, sample_rate: u32, channels: u8) -> Self {
        let total_samples = capacity_frames * channels as usize;
        info!("Creating lock-free audio buffer: {} frames, {}Hz, {} channels ({} samples)",
              capacity_frames, sample_rate, channels, total_samples);

        Self {
            buffer: vec![0.0; total_samples],
            capacity: total_samples,
            channels,
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            sample_rate,
        }
    }

    pub fn write(&self, data: &[f32]) -> Result<usize> {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        let read_pos = self.read_pos.load(Ordering::Acquire);

        let available = if write_pos >= read_pos {
            self.capacity - write_pos + read_pos - 1
        } else {
            read_pos - write_pos - 1
        };

        let to_write = std::cmp::min(data.len(), available);

        if to_write == 0 {
            return Ok(0); // Buffer full
        }

        // Write data in two parts if wrapping around
        let end_space = self.capacity - write_pos;
        if to_write <= end_space {
            // Single write
            unsafe {
                let buffer_ptr = self.buffer.as_ptr() as *mut f32;
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    buffer_ptr.add(write_pos),
                    to_write,
                );
            }
        } else {
            // Split write
            unsafe {
                let buffer_ptr = self.buffer.as_ptr() as *mut f32;
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    buffer_ptr.add(write_pos),
                    end_space,
                );
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add(end_space),
                    buffer_ptr,
                    to_write - end_space,
                );
            }
        }

        let new_write_pos = (write_pos + to_write) % self.capacity;
        self.write_pos.store(new_write_pos, Ordering::Release);

        Ok(to_write)
    }

    pub fn read(&self, data: &mut [f32]) -> Result<usize> {
        let read_pos = self.read_pos.load(Ordering::Acquire);
        let write_pos = self.write_pos.load(Ordering::Acquire);

        let available = if write_pos >= read_pos {
            write_pos - read_pos
        } else {
            self.capacity - read_pos + write_pos
        };

        let to_read = std::cmp::min(data.len(), available);

        if to_read == 0 {
            return Ok(0); // Buffer empty
        }

        // Read data in two parts if wrapping around
        let end_space = self.capacity - read_pos;
        if to_read <= end_space {
            // Single read
            unsafe {
                let buffer_ptr = self.buffer.as_ptr();
                std::ptr::copy_nonoverlapping(
                    buffer_ptr.add(read_pos),
                    data.as_mut_ptr(),
                    to_read,
                );
            }
        } else {
            // Split read
            unsafe {
                let buffer_ptr = self.buffer.as_ptr();
                std::ptr::copy_nonoverlapping(
                    buffer_ptr.add(read_pos),
                    data.as_mut_ptr(),
                    end_space,
                );
                std::ptr::copy_nonoverlapping(
                    buffer_ptr,
                    data.as_mut_ptr().add(end_space),
                    to_read - end_space,
                );
            }
        }

        let new_read_pos = (read_pos + to_read) % self.capacity;
        self.read_pos.store(new_read_pos, Ordering::Release);

        Ok(to_read)
    }

    pub fn available_for_write(&self) -> usize {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        let read_pos = self.read_pos.load(Ordering::Acquire);

        if write_pos >= read_pos {
            self.capacity - write_pos + read_pos - 1
        } else {
            read_pos - write_pos - 1
        }
    }

    pub fn available_for_read(&self) -> usize {
        let read_pos = self.read_pos.load(Ordering::Acquire);
        let write_pos = self.write_pos.load(Ordering::Acquire);

        if write_pos >= read_pos {
            write_pos - read_pos
        } else {
            self.capacity - read_pos + write_pos
        }
    }

    /// Clear the buffer by resetting read/write positions
    pub fn clear(&self) {
        self.read_pos.store(0, Ordering::Release);
        self.write_pos.store(0, Ordering::Release);
    }

    /// Get buffer utilization as a percentage (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        let available = self.available_for_read();
        available as f32 / self.capacity as f32
    }

    /// Check if buffer is nearly full (>90% capacity)
    pub fn is_nearly_full(&self) -> bool {
        self.utilization() > 0.9
    }

    /// Check if buffer is nearly empty (<10% capacity)
    pub fn is_nearly_empty(&self) -> bool {
        self.utilization() < 0.1
    }

    /// Get number of channels
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get capacity in frames (not samples)
    pub fn capacity_frames(&self) -> usize {
        self.capacity / self.channels as usize
    }

    /// Non-blocking write - returns immediately if buffer is full
    pub fn try_write(&self, data: &[f32]) -> Result<usize> {
        if self.available_for_write() == 0 {
            return Ok(0);
        }
        self.write(data)
    }

    /// Non-blocking read - returns immediately if buffer is empty
    pub fn try_read(&self, data: &mut [f32]) -> Result<usize> {
        if self.available_for_read() == 0 {
            return Ok(0);
        }
        self.read(data)
    }
}

/// Real-time audio processing scheduler
pub struct RealTimeScheduler {
    sample_rate: u32,
    frame_duration: Duration,
}

impl RealTimeScheduler {
    pub fn new(sample_rate: u32, buffer_size: usize) -> Self {
        let frame_duration = Duration::from_micros(
            (buffer_size as f64 / sample_rate as f64 * 1_000_000.0) as u64
        );
        let target_latency = Duration::from_millis(TARGET_LATENCY_MS as u64);

        info!("RealTime scheduler: {}Hz, {} frames, {:.2}ms target latency",
              sample_rate, buffer_size, target_latency.as_secs_f64() * 1000.0);

        Self {
            sample_rate,
            frame_duration,
        }
    }

    pub fn optimize_thread_for_audio() -> Result<()> {
        info!("Optimizing thread for real-time audio processing");

        // Try to set real-time priority
        #[cfg(target_os = "linux")]
        {
            use std::ffi::c_int;

            unsafe extern "C" {
                fn sched_setscheduler(pid: c_int, policy: c_int, param: *const c_int) -> c_int;
                fn pthread_setname_np(thread: libc::pthread_t, name: *const libc::c_char) -> c_int;
            }

            const SCHED_FIFO: c_int = 1;
            const RT_PRIORITY: c_int = 80; // High priority for audio

            unsafe {
                // Set thread name
                let name = b"ghostwave-rt\0";
                let result = pthread_setname_np(libc::pthread_self(), name.as_ptr() as *const libc::c_char);
                if result == 0 {
                    debug!("Set thread name to 'ghostwave-rt'");
                } else {
                    warn!("Failed to set thread name: {}", result);
                }

                // Try to set real-time priority
                let result = sched_setscheduler(0, SCHED_FIFO, &RT_PRIORITY as *const c_int);
                if result == 0 {
                    info!("✅ Set real-time scheduling with priority {}", RT_PRIORITY);
                } else {
                    warn!("⚠️  Failed to set real-time priority (need CAP_SYS_NICE or run as root)");
                    info!("Consider: sudo setcap cap_sys_nice+ep ./ghostwave");
                }
            }
        }

        Ok(())
    }

    pub fn get_optimal_buffer_size(sample_rate: u32, target_latency_ms: u32) -> usize {
        let target_frames = (sample_rate as f64 * target_latency_ms as f64 / 1000.0) as usize;

        // Round to next power of 2 for efficiency
        let mut buffer_size = 1;
        while buffer_size < target_frames {
            buffer_size *= 2;
        }

        // Clamp to reasonable bounds
        buffer_size = buffer_size.max(32).min(2048);

        info!("Optimal buffer size for {}Hz @ {}ms latency: {} frames",
              sample_rate, target_latency_ms, buffer_size);

        buffer_size
    }

    pub fn calculate_latency(&self, buffer_size: usize) -> Duration {
        Duration::from_micros(
            (buffer_size as f64 / self.sample_rate as f64 * 1_000_000.0) as u64
        )
    }

    pub fn sleep_until_next_frame(&self, start_time: Instant) {
        let elapsed = start_time.elapsed();
        if elapsed < self.frame_duration {
            let sleep_time = self.frame_duration - elapsed;

            // Use high-precision sleep for timing accuracy
            if sleep_time > Duration::from_micros(100) {
                thread::sleep(sleep_time - Duration::from_micros(50));
            }

            // Busy wait for final precision
            while start_time.elapsed() < self.frame_duration {
                std::hint::spin_loop();
            }
        }
    }
}

/// Performance benchmarking for audio processing
pub struct AudioBenchmark {
    frame_count: AtomicU64,
    xrun_count: AtomicU64,
    max_processing_time: AtomicU64, // in nanoseconds
    target_frame_time: Duration,
}

impl AudioBenchmark {
    pub fn new(sample_rate: u32, buffer_size: usize) -> Self {
        let target_frame_time = Duration::from_micros(
            (buffer_size as f64 / sample_rate as f64 * 1_000_000.0) as u64
        );

        info!("Audio benchmark initialized: target frame time = {:.2}μs",
              target_frame_time.as_micros());

        Self {
            frame_count: AtomicU64::new(0),
            xrun_count: AtomicU64::new(0),
            max_processing_time: AtomicU64::new(0),
            target_frame_time,
        }
    }

    pub fn record_frame_processing(&self, processing_time: Duration) {
        self.frame_count.fetch_add(1, Ordering::Relaxed);

        let processing_ns = processing_time.as_nanos() as u64;
        let current_max = self.max_processing_time.load(Ordering::Relaxed);
        if processing_ns > current_max {
            self.max_processing_time.store(processing_ns, Ordering::Relaxed);
        }

        // Check for XRuns (processing took longer than available time)
        if processing_time > self.target_frame_time {
            self.xrun_count.fetch_add(1, Ordering::Relaxed);
            warn!("XRun detected: processing took {:.2}μs (target: {:.2}μs)",
                  processing_time.as_micros(), self.target_frame_time.as_micros());
        }
    }

    pub fn get_stats(&self) -> BenchmarkStats {
        BenchmarkStats {
            total_frames: self.frame_count.load(Ordering::Relaxed),
            xrun_count: self.xrun_count.load(Ordering::Relaxed),
            max_processing_time: Duration::from_nanos(self.max_processing_time.load(Ordering::Relaxed)),
            target_frame_time: self.target_frame_time,
        }
    }

    pub fn report_stats(&self) {
        let stats = self.get_stats();
        let xrun_rate = if stats.total_frames > 0 {
            (stats.xrun_count as f64 / stats.total_frames as f64) * 100.0
        } else {
            0.0
        };

        info!("📊 Audio Performance Stats:");
        info!("  Frames processed: {}", stats.total_frames);
        info!("  XRuns: {} ({:.3}%)", stats.xrun_count, xrun_rate);
        info!("  Max processing time: {:.2}μs", stats.max_processing_time.as_micros());
        info!("  Target frame time: {:.2}μs", stats.target_frame_time.as_micros());

        if xrun_rate > 1.0 {
            warn!("⚠️  High XRun rate detected - consider increasing buffer size");
        } else if xrun_rate == 0.0 {
            info!("✅ Perfect real-time performance - no XRuns detected");
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    pub total_frames: u64,
    pub xrun_count: u64,
    pub max_processing_time: Duration,
    pub target_frame_time: Duration,
}

/// Memory pool for zero-allocation audio buffers
pub struct AudioMemoryPool {
    buffers: crossbeam::channel::Receiver<Vec<f32>>,
    buffer_sender: crossbeam::channel::Sender<Vec<f32>>,
    buffer_size: usize,
}

impl AudioMemoryPool {
    pub fn new(buffer_size: usize, pool_size: usize) -> Self {
        let (sender, receiver) = crossbeam::channel::bounded(pool_size);

        // Pre-allocate buffers
        for _ in 0..pool_size {
            let buffer = vec![0.0f32; buffer_size];
            sender.send(buffer).expect("Failed to initialize memory pool");
        }

        info!("Audio memory pool created: {} buffers of {} samples each",
              pool_size, buffer_size);

        Self {
            buffers: receiver,
            buffer_sender: sender,
            buffer_size,
        }
    }

    pub fn get_buffer(&self) -> Option<Vec<f32>> {
        match self.buffers.try_recv() {
            Ok(mut buffer) => {
                buffer.fill(0.0); // Clear the buffer
                Some(buffer)
            }
            Err(_) => {
                warn!("Memory pool exhausted, allocating new buffer");
                Some(vec![0.0f32; self.buffer_size])
            }
        }
    }

    pub fn return_buffer(&self, buffer: Vec<f32>) {
        if buffer.len() == self.buffer_size {
            let _ = self.buffer_sender.try_send(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_free_buffer() {
        let buffer = LockFreeAudioBuffer::new(1024, 48000);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let written = buffer.write(&input).unwrap();
        assert_eq!(written, 4);

        let mut output = vec![0.0; 4];
        let read = buffer.read(&mut output).unwrap();
        assert_eq!(read, 4);
        assert_eq!(output, input);
    }

    #[test]
    fn test_optimal_buffer_size() {
        let buffer_size = RealTimeScheduler::get_optimal_buffer_size(48000, 10);
        assert!(buffer_size >= 32);
        assert!(buffer_size <= 2048);
        assert_eq!(buffer_size & (buffer_size - 1), 0); // Power of 2
    }

    #[test]
    fn test_memory_pool() {
        let pool = AudioMemoryPool::new(512, 4);

        let buffer1 = pool.get_buffer().unwrap();
        let buffer2 = pool.get_buffer().unwrap();

        assert_eq!(buffer1.len(), 512);
        assert_eq!(buffer2.len(), 512);

        pool.return_buffer(buffer1);
        pool.return_buffer(buffer2);
    }
}