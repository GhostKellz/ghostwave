//! # GhostWave VST Plugin
//!
//! VST3/CLAP plugin providing NVIDIA Broadcast-style audio processing.
//!
//! ## Features
//! - **Noise Removal**: RTX-accelerated noise suppression
//! - **Echo Removal**: Acoustic echo cancellation
//! - **De-Esser**: Sibilance reduction (4-10kHz)
//! - **Parametric EQ**: 8-band professional EQ
//! - **Compressor/Limiter**: Dynamics processing with look-ahead
//!
//! ## GPU Acceleration
//! When compiled with `nvidia-rtx` feature and running on RTX 40/50 series,
//! leverages Tensor Cores for low-latency AI denoising.

use nih_plug::prelude::*;
use std::sync::Arc;

mod editor;
mod params;
mod processing;

pub use params::GhostWaveParams;
use processing::GhostWaveProcessing;

/// GhostWave VST Plugin
pub struct GhostWavePlugin {
    params: Arc<GhostWaveParams>,
    processing: GhostWaveProcessing,
    sample_rate: f32,
}

impl Default for GhostWavePlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(GhostWaveParams::default()),
            processing: GhostWaveProcessing::new(48000.0),
            sample_rate: 48000.0,
        }
    }
}

impl Plugin for GhostWavePlugin {
    const NAME: &'static str = "GhostWave";
    const VENDOR: &'static str = "Ghost Ecosystem";
    const URL: &'static str = "https://github.com/ghostkellz/ghostwave";
    const EMAIL: &'static str = "support@ghost-ecosystem.dev";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        // Mono
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
        // Stereo
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
    ];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(self.params.clone(), self.params.editor_state.clone())
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate;
        self.processing = GhostWaveProcessing::new(self.sample_rate);
        true
    }

    fn reset(&mut self) {
        self.processing.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Update processing parameters from plugin params
        self.processing.update_params(&self.params);

        // Process each sample across all channels
        for mut frame in buffer.iter_samples() {
            // Collect samples from all channels
            let num_channels = frame.len();
            let mut samples: Vec<f32> = frame.iter_mut().map(|s| *s).collect();

            // Process in-place (mono sum for now, expand later)
            if num_channels == 1 {
                self.processing.process_block(&mut samples);
            } else {
                // For stereo, process each channel or sum to mono then split
                // For now, process left channel and copy to right
                let mut mono = vec![samples[0]];
                self.processing.process_block(&mut mono);
                samples[0] = mono[0];
                if num_channels > 1 {
                    samples[1] = mono[0]; // Simple mono->stereo for now
                }
            }

            // Write back to frame
            for (i, sample) in frame.iter_mut().enumerate() {
                if i < samples.len() {
                    *sample = samples[i];
                }
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for GhostWavePlugin {
    const CLAP_ID: &'static str = "com.ghost-ecosystem.ghostwave";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("NVIDIA Broadcast-style audio processing for Linux");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Mono,
        ClapFeature::Stereo,
        ClapFeature::Filter,
        ClapFeature::Equalizer,
        ClapFeature::Compressor,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for GhostWavePlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"GhostWaveVST3!!!";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Dynamics,
        Vst3SubCategory::Eq,
    ];
}

nih_export_clap!(GhostWavePlugin);
nih_export_vst3!(GhostWavePlugin);
