use anyhow::Result;
use cpal::{Device, StreamConfig};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, error};

use crate::config::Config;
use crate::noise_suppression::NoiseProcessor;

pub struct AudioProcessor {
    config: Config,
    noise_processor: Arc<RwLock<NoiseProcessor>>,
}

impl AudioProcessor {
    pub fn new(config: Config) -> Result<Self> {
        let noise_processor = Arc::new(RwLock::new(
            NoiseProcessor::new(&config.noise_suppression)?
        ));

        Ok(Self {
            config,
            noise_processor,
        })
    }

    pub async fn process_audio_buffer(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if !self.config.noise_suppression.enabled {
            output.copy_from_slice(input);
            return Ok(());
        }

        let mut processor = self.noise_processor.write().await;
        processor.process(input, output)?;

        Ok(())
    }

    pub fn get_stream_config(&self) -> StreamConfig {
        StreamConfig {
            channels: self.config.audio.channels.into(),
            sample_rate: cpal::SampleRate(self.config.audio.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.audio.buffer_size),
        }
    }
}

pub async fn run_standalone(config: Config) -> Result<()> {
    info!("Initializing standalone audio processing");

    let host = cpal::default_host();
    let input_device = if let Some(name) = &config.audio.input_device {
        find_device_by_name(&host, name)?
    } else {
        host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device"))?
    };

    let output_device = if let Some(name) = &config.audio.output_device {
        find_device_by_name(&host, name)?
    } else {
        host.default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No default output device"))?
    };

    let processor = Arc::new(AudioProcessor::new(config)?);
    let stream_config = processor.get_stream_config();

    info!("Input device: {:?}", input_device.name());
    info!("Output device: {:?}", output_device.name());
    info!("Stream config: {:?}", stream_config);

    let input_stream = input_device.build_input_stream(
        &stream_config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            debug!("Processing {} samples", data.len());
        },
        move |err| {
            error!("Input stream error: {}", err);
        },
        None,
    )?;

    let output_stream = output_device.build_output_stream(
        &stream_config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            for sample in data.iter_mut() {
                *sample = 0.0;
            }
        },
        move |err| {
            error!("Output stream error: {}", err);
        },
        None,
    )?;

    input_stream.play()?;
    output_stream.play()?;

    info!("🎤 Audio processing started - Press Ctrl+C to stop");

    tokio::signal::ctrl_c().await?;
    info!("Shutting down audio processing");

    drop(input_stream);
    drop(output_stream);

    Ok(())
}

fn find_device_by_name(host: &cpal::Host, name: &str) -> Result<Device> {
    for device in host.input_devices()? {
        if device.name()?.contains(name) {
            return Ok(device);
        }
    }
    for device in host.output_devices()? {
        if device.name()?.contains(name) {
            return Ok(device);
        }
    }
    Err(anyhow::anyhow!("Device '{}' not found", name))
}