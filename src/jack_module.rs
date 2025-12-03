use anyhow::{Context, Result};
use jack::{AudioIn, AudioOut, Client, ClientOptions, Control, Port, ProcessScope};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

use crate::config::Config;
use crate::low_latency::AudioBenchmark;
use crate::noise_suppression::NoiseProcessor;

pub struct JackModule {
    config: Config,
    client: Option<Client>,
    processor: Arc<Mutex<NoiseProcessor>>,
    benchmark: Arc<Mutex<AudioBenchmark>>,
    running: Arc<AtomicBool>,
}

struct JackProcessor {
    input_ports: Vec<Port<AudioIn>>,
    output_ports: Vec<Port<AudioOut>>,
    processor: Arc<Mutex<NoiseProcessor>>,
    benchmark: Arc<Mutex<AudioBenchmark>>,
    frame_count: u64,
}

impl JackProcessor {
    fn new(
        input_ports: Vec<Port<AudioIn>>,
        output_ports: Vec<Port<AudioOut>>,
        processor: Arc<Mutex<NoiseProcessor>>,
        benchmark: Arc<Mutex<AudioBenchmark>>,
    ) -> Self {
        Self {
            input_ports,
            output_ports,
            processor,
            benchmark,
            frame_count: 0,
        }
    }
}

impl jack::ProcessHandler for JackProcessor {
    fn process(&mut self, _: &Client, ps: &ProcessScope) -> Control {
        let start_time = std::time::Instant::now();
        let buffer_size = ps.n_frames() as usize;

        // Get input audio
        let input_slices: Vec<&[f32]> = self.input_ports.iter().map(|p| p.as_slice(ps)).collect();

        // Get output audio buffers
        let mut output_slices: Vec<&mut [f32]> = self
            .output_ports
            .iter_mut()
            .map(|p| p.as_mut_slice(ps))
            .collect();

        if input_slices.is_empty() || output_slices.is_empty() {
            return Control::Continue;
        }

        // Convert interleaved to planar for processing
        let channels = input_slices.len().min(output_slices.len());
        let total_samples = buffer_size * channels;
        let mut interleaved_input = vec![0.0f32; total_samples];
        let mut interleaved_output = vec![0.0f32; total_samples];

        // Interleave input
        for (channel, input_slice) in input_slices.iter().enumerate().take(channels) {
            for (frame, &sample) in input_slice.iter().enumerate().take(buffer_size) {
                interleaved_input[frame * channels + channel] = sample;
            }
        }

        // Process audio
        if let Ok(mut processor) = self.processor.lock() {
            if let Err(e) = processor.process(&interleaved_input, &mut interleaved_output) {
                // On error, pass through input
                interleaved_output.copy_from_slice(&interleaved_input);
                debug!("JACK processing error: {}", e);
            }
        } else {
            // If can't lock processor, pass through
            interleaved_output.copy_from_slice(&interleaved_input);
        }

        // Deinterleave output
        for (channel, output_slice) in output_slices.iter_mut().enumerate().take(channels) {
            for (frame, sample) in output_slice.iter_mut().enumerate().take(buffer_size) {
                *sample = interleaved_output[frame * channels + channel];
            }
        }

        // Record performance
        let processing_time = start_time.elapsed();
        if let Ok(benchmark) = self.benchmark.lock() {
            benchmark.record_frame_processing(processing_time);
        }

        self.frame_count += 1;

        // Debug stats every 1000 frames
        if self.frame_count % 1000 == 0 {
            if let Ok(benchmark) = self.benchmark.lock() {
                let stats = benchmark.get_stats();
                debug!(
                    "JACK: {} frames processed, {} XRuns",
                    stats.total_frames, stats.xrun_count
                );
            }
        }

        Control::Continue
    }
}

impl JackModule {
    pub fn new(config: Config) -> Result<Self> {
        info!("Initializing JACK module for professional audio workflows");

        let processor = Arc::new(Mutex::new(NoiseProcessor::new(&config.noise_suppression)?));

        let benchmark = Arc::new(Mutex::new(AudioBenchmark::new(
            config.audio.sample_rate,
            config.audio.buffer_size as usize,
        )));

        Ok(Self {
            config,
            client: None,
            processor,
            benchmark,
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn connect_to_jack(&mut self) -> Result<()> {
        info!("Connecting to JACK server...");

        let client_name = "ghostwave";
        let (client, status) = Client::new(client_name, ClientOptions::NO_START_SERVER)
            .context("Failed to create JACK client")?;

        info!("✅ Connected to JACK server");
        info!("Client name: {}", client.name());
        info!("JACK status: {:?}", status);
        info!("Sample rate: {}Hz", client.sample_rate());
        info!("Buffer size: {} frames", client.buffer_size());

        // Verify sample rate compatibility
        let jack_sample_rate = client.sample_rate();
        if jack_sample_rate != self.config.audio.sample_rate as usize {
            warn!(
                "Sample rate mismatch: JACK={}Hz, Config={}Hz",
                jack_sample_rate, self.config.audio.sample_rate
            );
            info!("Using JACK sample rate: {}Hz", jack_sample_rate);
        }

        self.client = Some(client);
        Ok(())
    }

    pub fn create_ports(&mut self) -> Result<()> {
        let client = self
            .client
            .take()
            .ok_or_else(|| anyhow::anyhow!("JACK client not connected"))?;

        info!("Creating JACK audio ports");

        let channels = self.config.audio.channels as usize;

        // Create input ports
        let mut input_ports = Vec::new();
        for i in 0..channels {
            let port_name = format!("input_{}", i + 1);
            let port = client
                .register_port(&port_name, AudioIn::default())
                .with_context(|| format!("Failed to create input port: {}", port_name))?;
            input_ports.push(port);
            info!("Created input port: {}", port_name);
        }

        // Create output ports
        let mut output_ports = Vec::new();
        for i in 0..channels {
            let port_name = format!("output_{}", i + 1);
            let port = client
                .register_port(&port_name, AudioOut::default())
                .with_context(|| format!("Failed to create output port: {}", port_name))?;
            output_ports.push(port);
            info!("Created output port: {}", port_name);
        }

        info!(
            "✅ Created {} input and {} output ports",
            channels, channels
        );

        // Create process handler
        let processor = JackProcessor::new(
            input_ports,
            output_ports,
            self.processor.clone(),
            self.benchmark.clone(),
        );

        // Activate client with process handler
        let active_client = client
            .activate_async((), processor)
            .context("Failed to activate JACK client")?;

        self.running.store(true, Ordering::SeqCst);

        info!("🎵 JACK audio processing activated");
        info!("Processing mode: {}", self.get_processing_mode());

        // Keep the active client alive
        // In a real implementation, you'd store this and manage its lifecycle
        std::mem::forget(active_client);

        Ok(())
    }

    pub fn auto_connect_ports(&self) -> Result<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("JACK client not connected"))?;

        info!("Auto-connecting JACK ports");

        // Get available ports
        let input_ports = client.ports(None, None, jack::PortFlags::IS_OUTPUT);
        let output_ports = client.ports(None, None, jack::PortFlags::IS_INPUT);

        info!("Available input sources: {}", input_ports.len());
        info!("Available output destinations: {}", output_ports.len());

        // Auto-connect to system ports if available
        let system_capture_ports: Vec<_> = input_ports
            .iter()
            .filter(|name| name.starts_with("system:capture_"))
            .collect();

        let system_playback_ports: Vec<_> = output_ports
            .iter()
            .filter(|name| name.starts_with("system:playback_"))
            .collect();

        let channels = self.config.audio.channels as usize;

        // Connect inputs (system output -> our input)
        for i in 0..channels.min(system_capture_ports.len()) {
            let source = system_capture_ports[i];
            let dest = format!("ghostwave:input_{}", i + 1);

            match client.connect_ports_by_name(source, &dest) {
                Ok(_) => info!("Connected {} -> {}", source, dest),
                Err(e) => warn!("Failed to connect {} -> {}: {}", source, dest, e),
            }
        }

        // Connect outputs (our output -> system input)
        for i in 0..channels.min(system_playback_ports.len()) {
            let source = format!("ghostwave:output_{}", i + 1);
            let dest = system_playback_ports[i];

            match client.connect_ports_by_name(&source, dest) {
                Ok(_) => info!("Connected {} -> {}", source, dest),
                Err(e) => warn!("Failed to connect {} -> {}: {}", source, dest, e),
            }
        }

        info!("✅ JACK port auto-connection complete");
        Ok(())
    }

    pub fn get_processing_mode(&self) -> String {
        if let Ok(processor) = self.processor.lock() {
            format!("JACK + {}", processor.get_processing_mode())
        } else {
            "JACK + Unknown".to_string()
        }
    }

    #[allow(dead_code)] // Public API for JACK introspection
    pub fn get_jack_info(&self) -> Option<JackInfo> {
        self.client.as_ref().map(|client| JackInfo {
            client_name: client.name().to_string(),
            sample_rate: client.sample_rate(),
            buffer_size: client.buffer_size(),
            cpu_load: client.cpu_load(),
            is_realtime: true, // Assume JACK is running in realtime mode
        })
    }

    pub fn disconnect(&mut self) -> Result<()> {
        info!("Disconnecting from JACK server");

        self.running.store(false, Ordering::SeqCst);

        if let Some(_client) = self.client.take() {
            // Client will be automatically deactivated when dropped
            info!("JACK client deactivated");
        }

        // Report final stats
        if let Ok(benchmark) = self.benchmark.lock() {
            benchmark.report_stats();
        }

        info!("✅ JACK module disconnected");
        Ok(())
    }
}

impl Drop for JackModule {
    fn drop(&mut self) {
        let _ = self.disconnect();
    }
}

/// JACK server information
#[derive(Debug, Clone)]
#[allow(dead_code)] // Public API for JACK introspection
pub struct JackInfo {
    pub client_name: String,
    pub sample_rate: usize,
    pub buffer_size: u32,
    pub cpu_load: f32,
    pub is_realtime: bool,
}

/// Check if JACK server is running
pub fn check_jack_availability() -> bool {
    match Client::new("ghostwave-test", ClientOptions::NO_START_SERVER) {
        Ok(_) => {
            info!("✅ JACK server is running");
            true
        }
        Err(e) => {
            debug!("JACK server not available: {}", e);
            false
        }
    }
}

/// Get JACK server information
pub fn get_jack_server_info() -> Result<Option<JackInfo>> {
    if !check_jack_availability() {
        return Ok(None);
    }

    let (client, _) = Client::new("ghostwave-info", ClientOptions::NO_START_SERVER)?;

    Ok(Some(JackInfo {
        client_name: client.name().to_string(),
        sample_rate: client.sample_rate(),
        buffer_size: client.buffer_size(),
        cpu_load: client.cpu_load(),
        is_realtime: true, // Assume JACK is running in realtime mode
    }))
}

pub async fn run_jack_mode(config: Config) -> Result<()> {
    info!("🎶 Starting GhostWave in JACK professional audio mode");
    info!("This provides seamless integration with professional audio workflows");

    // Check JACK availability
    if !check_jack_availability() {
        return Err(anyhow::anyhow!(
            "JACK server is not running. Start JACK with: jackd -d alsa -r {} -p {}",
            config.audio.sample_rate,
            config.audio.buffer_size
        ));
    }

    // Show JACK server info
    if let Some(info) = get_jack_server_info()? {
        info!("JACK Server Information:");
        info!("  Sample Rate: {}Hz", info.sample_rate);
        info!("  Buffer Size: {} frames", info.buffer_size);
        info!("  CPU Load: {:.1}%", info.cpu_load * 100.0);
        info!("  Realtime: {}", info.is_realtime);
    }

    // Create and configure JACK module
    let mut jack_module = JackModule::new(config)?;

    jack_module
        .connect_to_jack()
        .context("Failed to connect to JACK server")?;

    jack_module
        .create_ports()
        .context("Failed to create JACK ports")?;

    jack_module
        .auto_connect_ports()
        .context("Failed to auto-connect JACK ports")?;

    info!("🎯 JACK module ready - Professional audio processing active");
    info!("Use qjackctl or jack_connect to manage audio routing");

    // Wait for shutdown signal
    tokio::signal::ctrl_c().await?;
    info!("Received shutdown signal");

    jack_module.disconnect()?;
    info!("JACK module shutdown complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jack_availability() {
        let available = check_jack_availability();
        println!("JACK available: {}", available);
    }

    #[test]
    fn test_jack_server_info() {
        if let Ok(Some(info)) = get_jack_server_info() {
            println!("JACK server info: {:?}", info);
        }
    }
}
