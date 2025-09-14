use anyhow::Result;

use crate::config::NoiseSuppressionConfig;

pub struct NoiseProcessor {
    config: NoiseSuppressionConfig,
    gate: NoiseGate,
    spectral_filter: SpectralFilter,
}

impl NoiseProcessor {
    pub fn new(config: &NoiseSuppressionConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            gate: NoiseGate::new(config.gate_threshold, config.release_time),
            spectral_filter: SpectralFilter::new(config.strength),
        })
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if !self.config.enabled || input.len() != output.len() {
            output.copy_from_slice(input);
            return Ok(());
        }

        let mut temp_buffer = vec![0.0f32; input.len()];

        self.spectral_filter.process(input, &mut temp_buffer);

        self.gate.process(&temp_buffer, output);

        Ok(())
    }
}

struct NoiseGate {
    threshold: f32,
    release_time: f32,
    envelope: f32,
    attack_coeff: f32,
    release_coeff: f32,
}

impl NoiseGate {
    fn new(threshold_db: f32, release_time: f32) -> Self {
        let threshold = 10.0_f32.powf(threshold_db / 20.0);
        let sample_rate = 48000.0;
        let attack_time = 0.001;

        Self {
            threshold,
            release_time,
            envelope: 0.0,
            attack_coeff: (-1.0_f32 / (attack_time * sample_rate)).exp(),
            release_coeff: (-1.0_f32 / (release_time * sample_rate)).exp(),
        }
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        for (i, &sample) in input.iter().enumerate() {
            let level = sample.abs();

            if level > self.envelope {
                self.envelope = level + (self.envelope - level) * self.attack_coeff;
            } else {
                self.envelope = level + (self.envelope - level) * self.release_coeff;
            }

            let gate_factor = if self.envelope > self.threshold {
                1.0
            } else {
                (self.envelope / self.threshold).powf(2.0)
            };

            output[i] = sample * gate_factor;
        }
    }
}

struct SpectralFilter {
    strength: f32,
    frame_size: usize,
    hop_size: usize,
    window: Vec<f32>,
    noise_profile: Vec<f32>,
    adaptation_rate: f32,
}

impl SpectralFilter {
    fn new(strength: f32) -> Self {
        let frame_size = 1024;
        let hop_size = frame_size / 4;
        let window = Self::hanning_window(frame_size);

        Self {
            strength,
            frame_size,
            hop_size,
            window,
            noise_profile: vec![0.0; frame_size / 2 + 1],
            adaptation_rate: 0.95,
        }
    }

    fn process(&self, input: &[f32], output: &mut [f32]) {
        output.copy_from_slice(input);

        for i in 0..input.len() {
            let filtered = input[i] * (1.0 - self.strength * 0.5);
            output[i] = filtered;
        }
    }

    fn hanning_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect()
    }
}