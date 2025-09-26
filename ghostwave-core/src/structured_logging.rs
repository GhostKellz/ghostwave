//! # Structured Logging with Ring Buffer
//!
//! Provides structured logging with a ring buffer for recent errors,
//! enabling efficient error tracking and diagnostics.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{error, warn, info, debug};

/// Maximum number of log entries to keep in ring buffer
const MAX_LOG_ENTRIES: usize = 1000;

/// Log severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LogLevel {
    Error = 4,
    Warn = 3,
    Info = 2,
    Debug = 1,
    Trace = 0,
}

impl From<tracing::Level> for LogLevel {
    fn from(level: tracing::Level) -> Self {
        match level {
            tracing::Level::ERROR => LogLevel::Error,
            tracing::Level::WARN => LogLevel::Warn,
            tracing::Level::INFO => LogLevel::Info,
            tracing::Level::DEBUG => LogLevel::Debug,
            tracing::Level::TRACE => LogLevel::Trace,
        }
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Trace => write!(f, "TRACE"),
        }
    }
}

/// Structured log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Timestamp in milliseconds since Unix epoch
    pub timestamp: u64,
    /// Log level
    pub level: LogLevel,
    /// Component that generated the log
    pub component: String,
    /// Log message
    pub message: String,
    /// Additional structured fields
    pub fields: std::collections::HashMap<String, serde_json::Value>,
    /// Thread ID
    pub thread_id: String,
    /// Source location (file:line)
    pub location: Option<String>,
}

impl LogEntry {
    pub fn new(level: LogLevel, component: &str, message: &str) -> Self {
        Self {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            level,
            component: component.to_string(),
            message: message.to_string(),
            fields: std::collections::HashMap::new(),
            thread_id: format!("{:?}", std::thread::current().id()),
            location: None,
        }
    }

    pub fn with_field<V: Into<serde_json::Value>>(mut self, key: &str, value: V) -> Self {
        self.fields.insert(key.to_string(), value.into());
        self
    }

    pub fn with_location(mut self, location: &str) -> Self {
        self.location = Some(location.to_string());
        self
    }

    pub fn is_error(&self) -> bool {
        self.level == LogLevel::Error
    }

    pub fn is_warning_or_error(&self) -> bool {
        self.level >= LogLevel::Warn
    }

    pub fn age_seconds(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        (now - self.timestamp) / 1000
    }
}

/// Ring buffer for log entries
pub struct LogRingBuffer {
    entries: VecDeque<LogEntry>,
    max_size: usize,
    error_count: u64,
    warning_count: u64,
    total_count: u64,
}

impl LogRingBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_size),
            max_size,
            error_count: 0,
            warning_count: 0,
            total_count: 0,
        }
    }

    pub fn push(&mut self, entry: LogEntry) {
        if self.entries.len() >= self.max_size {
            self.entries.pop_front();
        }

        if entry.is_error() {
            self.error_count += 1;
        } else if entry.level == LogLevel::Warn {
            self.warning_count += 1;
        }

        self.total_count += 1;
        self.entries.push_back(entry);
    }

    pub fn get_recent(&self, count: usize) -> Vec<&LogEntry> {
        self.entries.iter().rev().take(count).collect()
    }

    pub fn get_errors(&self) -> Vec<&LogEntry> {
        self.entries.iter().filter(|e| e.is_error()).collect()
    }

    pub fn get_warnings_and_errors(&self) -> Vec<&LogEntry> {
        self.entries.iter().filter(|e| e.is_warning_or_error()).collect()
    }

    pub fn get_since(&self, timestamp: u64) -> Vec<&LogEntry> {
        self.entries.iter()
            .filter(|e| e.timestamp >= timestamp)
            .collect()
    }

    pub fn get_component_logs(&self, component: &str) -> Vec<&LogEntry> {
        self.entries.iter()
            .filter(|e| e.component == component)
            .collect()
    }

    pub fn stats(&self) -> LogStats {
        LogStats {
            total_entries: self.total_count,
            error_count: self.error_count,
            warning_count: self.warning_count,
            buffer_size: self.entries.len(),
            max_buffer_size: self.max_size,
            oldest_entry_age_seconds: self.entries.front()
                .map(|e| e.age_seconds())
                .unwrap_or(0),
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.error_count = 0;
        self.warning_count = 0;
        self.total_count = 0;
    }
}

/// Log statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogStats {
    pub total_entries: u64,
    pub error_count: u64,
    pub warning_count: u64,
    pub buffer_size: usize,
    pub max_buffer_size: usize,
    pub oldest_entry_age_seconds: u64,
}

/// Structured logger with ring buffer
pub struct StructuredLogger {
    buffer: Arc<Mutex<LogRingBuffer>>,
    min_level: LogLevel,
}

impl StructuredLogger {
    pub fn new(min_level: LogLevel) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(LogRingBuffer::new(MAX_LOG_ENTRIES))),
            min_level,
        }
    }

    pub fn log(&self, level: LogLevel, component: &str, message: &str) {
        if level >= self.min_level {
            let entry = LogEntry::new(level, component, message);
            self.buffer.lock().unwrap().push(entry);

            // Also log to tracing
            match level {
                LogLevel::Error => error!("{}: {}", component, message),
                LogLevel::Warn => warn!("{}: {}", component, message),
                LogLevel::Info => info!("{}: {}", component, message),
                LogLevel::Debug => debug!("{}: {}", component, message),
                LogLevel::Trace => debug!("{}: {}", component, message), // Map trace to debug
            }
        }
    }

    pub fn log_with_fields(&self, level: LogLevel, component: &str, message: &str,
                          fields: std::collections::HashMap<String, serde_json::Value>) {
        if level >= self.min_level {
            let mut entry = LogEntry::new(level, component, message);
            entry.fields = fields;
            self.buffer.lock().unwrap().push(entry);

            // Log to tracing with fields
            match level {
                LogLevel::Error => error!("{}: {} (fields: {:?})", component, message, entry.fields),
                LogLevel::Warn => warn!("{}: {} (fields: {:?})", component, message, entry.fields),
                LogLevel::Info => info!("{}: {} (fields: {:?})", component, message, entry.fields),
                LogLevel::Debug => debug!("{}: {} (fields: {:?})", component, message, entry.fields),
                LogLevel::Trace => debug!("{}: {} (fields: {:?})", component, message, entry.fields),
            }
        }
    }

    pub fn error(&self, component: &str, message: &str) {
        self.log(LogLevel::Error, component, message);
    }

    pub fn warn(&self, component: &str, message: &str) {
        self.log(LogLevel::Warn, component, message);
    }

    pub fn info(&self, component: &str, message: &str) {
        self.log(LogLevel::Info, component, message);
    }

    pub fn debug(&self, component: &str, message: &str) {
        self.log(LogLevel::Debug, component, message);
    }

    pub fn get_recent_logs(&self, count: usize) -> Vec<LogEntry> {
        self.buffer.lock().unwrap().get_recent(count)
            .into_iter().cloned().collect()
    }

    pub fn get_error_logs(&self) -> Vec<LogEntry> {
        self.buffer.lock().unwrap().get_errors()
            .into_iter().cloned().collect()
    }

    pub fn get_stats(&self) -> LogStats {
        self.buffer.lock().unwrap().stats()
    }

    pub fn clear_logs(&self) {
        self.buffer.lock().unwrap().clear();
    }

    pub fn export_logs(&self) -> Result<String> {
        let buffer = self.buffer.lock().unwrap();
        let logs: Vec<&LogEntry> = buffer.entries.iter().collect();
        Ok(serde_json::to_string_pretty(&logs)?)
    }

    pub fn get_buffer_handle(&self) -> Arc<Mutex<LogRingBuffer>> {
        Arc::clone(&self.buffer)
    }
}

/// Logger macros for easier usage
#[macro_export]
macro_rules! log_error {
    ($logger:expr, $component:expr, $($arg:tt)+) => {
        $logger.error($component, &format!($($arg)+))
    };
}

#[macro_export]
macro_rules! log_warn {
    ($logger:expr, $component:expr, $($arg:tt)+) => {
        $logger.warn($component, &format!($($arg)+))
    };
}

#[macro_export]
macro_rules! log_info {
    ($logger:expr, $component:expr, $($arg:tt)+) => {
        $logger.info($component, &format!($($arg)+))
    };
}

#[macro_export]
macro_rules! log_debug {
    ($logger:expr, $component:expr, $($arg:tt)+) => {
        $logger.debug($component, &format!($($arg)+))
    };
}

/// Global logger instance
static mut GLOBAL_LOGGER: Option<StructuredLogger> = None;
static LOGGER_INIT: std::sync::Once = std::sync::Once::new();

/// Initialize the global structured logger
pub fn init_global_logger(min_level: LogLevel) -> &'static StructuredLogger {
    unsafe {
        LOGGER_INIT.call_once(|| {
            GLOBAL_LOGGER = Some(StructuredLogger::new(min_level));
        });
        GLOBAL_LOGGER.as_ref().unwrap()
    }
}

/// Get the global logger instance
pub fn global_logger() -> Option<&'static StructuredLogger> {
    unsafe { GLOBAL_LOGGER.as_ref() }
}

/// Audio processing specific logging components
pub mod components {
    pub const AUDIO_ENGINE: &str = "audio_engine";
    pub const DSP_PIPELINE: &str = "dsp_pipeline";
    pub const PIPEWIRE: &str = "pipewire";
    pub const ALSA: &str = "alsa";
    pub const JACK: &str = "jack";
    pub const NOISE_SUPPRESSION: &str = "noise_suppression";
    pub const LATENCY_OPTIMIZER: &str = "latency_optimizer";
    pub const CONFIG_MANAGER: &str = "config_manager";
    pub const IPC_SERVER: &str = "ipc_server";
    pub const DEVICE_DETECTION: &str = "device_detection";
    pub const RTX_ACCELERATION: &str = "rtx_acceleration";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer() {
        let mut buffer = LogRingBuffer::new(3);

        // Add entries
        buffer.push(LogEntry::new(LogLevel::Info, "test", "message 1"));
        buffer.push(LogEntry::new(LogLevel::Warn, "test", "message 2"));
        buffer.push(LogEntry::new(LogLevel::Error, "test", "message 3"));

        assert_eq!(buffer.entries.len(), 3);
        assert_eq!(buffer.error_count, 1);
        assert_eq!(buffer.warning_count, 1);

        // Add one more to trigger overflow
        buffer.push(LogEntry::new(LogLevel::Info, "test", "message 4"));

        assert_eq!(buffer.entries.len(), 3); // Should still be 3
        assert_eq!(buffer.entries[0].message, "message 2"); // First entry should be removed
    }

    #[test]
    fn test_structured_logger() {
        let logger = StructuredLogger::new(LogLevel::Debug);

        logger.info("test_component", "test message");
        logger.error("test_component", "error message");

        let stats = logger.get_stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.error_count, 1);

        let errors = logger.get_error_logs();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].message, "error message");
    }

    #[test]
    fn test_log_filtering() {
        let buffer = LogRingBuffer::new(10);
        let mut buffer = buffer;

        buffer.push(LogEntry::new(LogLevel::Info, "comp1", "info"));
        buffer.push(LogEntry::new(LogLevel::Error, "comp1", "error"));
        buffer.push(LogEntry::new(LogLevel::Warn, "comp2", "warning"));

        let comp1_logs = buffer.get_component_logs("comp1");
        assert_eq!(comp1_logs.len(), 2);

        let errors = buffer.get_errors();
        assert_eq!(errors.len(), 1);

        let warnings_and_errors = buffer.get_warnings_and_errors();
        assert_eq!(warnings_and_errors.len(), 2);
    }
}