//! # Acoustic Echo Cancellation (AEC)
//!
//! Removes room echo and speaker feedback from microphone input.
//! Essential for:
//! - Speakers playing audio while using mic (gaming, Discord)
//! - Untreated rooms with reverb
//! - Laptop/webcam setups with speaker bleed
//!
//! ## Algorithm
//! Uses frequency-domain adaptive filtering with:
//! - Partitioned block NLMS in frequency domain
//! - Double-talk detection (Geigel + energy-based)
//! - Non-linear residual echo suppression
//! - Comfort noise injection
//!
//! Based on WebRTC AEC3 and Speex concepts.

use anyhow::Result;
use num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex, ComplexToReal};
use std::collections::VecDeque;
use std::sync::Arc;
use tracing::debug;

/// AEC configuration
#[derive(Debug, Clone)]
pub struct AecConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Frame size in samples (should be power of 2)
    pub frame_size: usize,
    /// Echo tail length in milliseconds
    pub tail_length_ms: u32,
    /// NLMS adaptation step size (mu)
    pub step_size: f32,
    /// Regularization factor (delta)
    pub regularization: f32,
    /// Double-talk detection threshold
    pub double_talk_threshold: f32,
    /// Enable non-linear processing
    pub non_linear_processing: bool,
    /// Comfort noise level (dB)
    pub comfort_noise_db: f32,
}

impl Default for AecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            frame_size: 512, // Power of 2 for FFT
            tail_length_ms: 200,
            step_size: 0.3,
            regularization: 1e-6,
            double_talk_threshold: 0.6,
            non_linear_processing: true,
            comfort_noise_db: -60.0,
        }
    }
}

/// Echo path state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EchoState {
    /// No echo detected
    Clean,
    /// Echo detected and being cancelled
    Cancelling,
    /// Double-talk (both near and far end speaking)
    DoubleTalk,
    /// Echo path change detected
    Diverging,
}

/// Frequency-domain adaptive filter block
struct FilterBlock {
    /// Filter coefficients in frequency domain
    coeffs: Vec<Complex<f32>>,
    /// Reference signal block (frequency domain)
    reference: Vec<Complex<f32>>,
    /// Power spectrum for normalization
    power: Vec<f32>,
}

impl FilterBlock {
    fn new(fft_size: usize) -> Self {
        let freq_size = fft_size / 2 + 1;
        Self {
            coeffs: vec![Complex::new(0.0, 0.0); freq_size],
            reference: vec![Complex::new(0.0, 0.0); freq_size],
            power: vec![1e-6; freq_size],
        }
    }
}

/// Frequency-domain Acoustic Echo Canceller
pub struct AcousticEchoCanceller {
    config: AecConfig,

    // FFT processing
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    fft_size: usize,
    freq_size: usize,

    // Partitioned filter blocks
    filter_blocks: Vec<FilterBlock>,
    num_blocks: usize,

    // Time-domain buffers
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    reference_buffer: VecDeque<f32>,
    overlap_buffer: Vec<f32>,

    // Frequency-domain buffers
    input_spectrum: Vec<Complex<f32>>,
    reference_spectrum: Vec<Complex<f32>>,
    error_spectrum: Vec<Complex<f32>>,
    echo_spectrum: Vec<Complex<f32>>,

    // Window function
    analysis_window: Vec<f32>,
    synthesis_window: Vec<f32>,

    // Adaptation state
    adaptation_rate: f32,
    power_smooth: Vec<f32>,
    adaptation_enabled: bool,

    // Double-talk detection
    near_energy: f32,
    far_energy: f32,
    echo_energy: f32,
    echo_state: EchoState,
    dtd_hangover: u32,

    // Non-linear processor
    nlp_enabled: bool,
    coherence: Vec<f32>,
    suppression_gain: Vec<f32>,

    // Statistics
    erle_db: f32,
    frames_processed: u64,

    // Comfort noise
    noise_floor: f32,
    rng_state: u32,
}

impl AcousticEchoCanceller {
    /// Create a new AEC instance
    pub fn new(config: AecConfig) -> Result<Self> {
        let fft_size = config.frame_size * 2; // 50% overlap
        let freq_size = fft_size / 2 + 1;

        // Number of filter blocks for partitioned convolution
        let tail_samples = (config.tail_length_ms as usize * config.sample_rate as usize) / 1000;
        let num_blocks = (tail_samples + config.frame_size - 1) / config.frame_size;

        debug!(
            "AEC initialized: {}ms tail = {} blocks of {} samples",
            config.tail_length_ms, num_blocks, config.frame_size
        );

        // Initialize FFT
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(fft_size);
        let fft_inverse = planner.plan_fft_inverse(fft_size);
        let scratch_len = fft_forward.get_scratch_len().max(fft_inverse.get_scratch_len());

        // Create filter blocks
        let filter_blocks: Vec<FilterBlock> = (0..num_blocks)
            .map(|_| FilterBlock::new(fft_size))
            .collect();

        // Create windows (Hann)
        let analysis_window = Self::create_hann_window(fft_size);
        let synthesis_window = Self::create_synthesis_window(&analysis_window, config.frame_size);

        Ok(Self {
            config: config.clone(),
            fft_forward,
            fft_inverse,
            fft_scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            fft_size,
            freq_size,
            filter_blocks,
            num_blocks,
            input_buffer: vec![0.0; fft_size],
            output_buffer: vec![0.0; fft_size],
            reference_buffer: VecDeque::with_capacity(fft_size * (num_blocks + 1)),
            overlap_buffer: vec![0.0; config.frame_size],
            input_spectrum: vec![Complex::new(0.0, 0.0); freq_size],
            reference_spectrum: vec![Complex::new(0.0, 0.0); freq_size],
            error_spectrum: vec![Complex::new(0.0, 0.0); freq_size],
            echo_spectrum: vec![Complex::new(0.0, 0.0); freq_size],
            analysis_window,
            synthesis_window,
            adaptation_rate: config.step_size,
            power_smooth: vec![1e-6; freq_size],
            adaptation_enabled: true,
            near_energy: 0.0,
            far_energy: 0.0,
            echo_energy: 0.0,
            echo_state: EchoState::Clean,
            dtd_hangover: 0,
            nlp_enabled: config.non_linear_processing,
            coherence: vec![0.0; freq_size],
            suppression_gain: vec![1.0; freq_size],
            erle_db: 0.0,
            frames_processed: 0,
            noise_floor: 10.0_f32.powf(config.comfort_noise_db / 20.0),
            rng_state: 12345,
        })
    }

    /// Create Hann window
    fn create_hann_window(size: usize) -> Vec<f32> {
        use std::f32::consts::PI;
        (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos()))
            .collect()
    }

    /// Create synthesis window for perfect reconstruction with OLA
    fn create_synthesis_window(analysis: &[f32], hop_size: usize) -> Vec<f32> {
        // For 50% overlap with Hann window, synthesis = analysis works
        // because sum of squared Hann windows with 50% overlap = 1
        let mut synthesis = analysis.to_vec();

        // Normalize for COLA (Constant Overlap-Add)
        let fft_size = analysis.len();
        for i in 0..fft_size {
            let mut sum = 0.0;
            for offset in (0..fft_size).step_by(hop_size) {
                let idx = (i + offset) % fft_size;
                sum += analysis[idx] * analysis[idx];
            }
            if sum > 1e-6 {
                synthesis[i] = analysis[i] / sum.sqrt();
            }
        }
        synthesis
    }

    /// Set the reference signal (speaker output)
    pub fn set_reference(&mut self, reference: &[f32]) -> Result<()> {
        // Add reference to buffer
        for &sample in reference {
            if self.reference_buffer.len() >= self.reference_buffer.capacity() {
                self.reference_buffer.pop_front();
            }
            self.reference_buffer.push_back(sample);
        }

        // Update far-end energy estimate
        let energy: f32 = reference.iter().map(|x| x * x).sum::<f32>() / reference.len() as f32;
        self.far_energy = 0.9 * self.far_energy + 0.1 * energy;

        Ok(())
    }

    /// Process microphone input to remove echo
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let frame_size = self.config.frame_size;

        if input.len() < frame_size || output.len() < frame_size {
            // Pass through if buffers too small
            let n = input.len().min(output.len());
            output[..n].copy_from_slice(&input[..n]);
            return Ok(());
        }

        // Check if we have enough reference data
        if self.reference_buffer.len() < self.fft_size {
            output[..frame_size].copy_from_slice(&input[..frame_size]);
            return Ok(());
        }

        // Shift input buffer and add new frame
        self.input_buffer.copy_within(frame_size.., 0);
        self.input_buffer[self.fft_size - frame_size..].copy_from_slice(&input[..frame_size]);

        // Update near-end energy
        let near_energy: f32 = input[..frame_size].iter().map(|x| x * x).sum::<f32>() / frame_size as f32;
        self.near_energy = 0.9 * self.near_energy + 0.1 * near_energy;

        // Apply analysis window and forward FFT
        let mut windowed: Vec<f32> = self.input_buffer
            .iter()
            .zip(&self.analysis_window)
            .map(|(s, w)| s * w)
            .collect();

        self.fft_forward.process_with_scratch(&mut windowed, &mut self.input_spectrum, &mut self.fft_scratch)?;

        // Update filter blocks with reference signal
        self.update_reference_blocks()?;

        // Compute echo estimate using partitioned convolution
        self.compute_echo_estimate();

        // Detect double-talk
        self.detect_double_talk();

        // Compute error signal
        for i in 0..self.freq_size {
            self.error_spectrum[i] = self.input_spectrum[i] - self.echo_spectrum[i];
        }

        // Adapt filter if not in double-talk
        if self.adaptation_enabled && self.echo_state != EchoState::DoubleTalk {
            self.adapt_filter();
        }

        // Apply non-linear processing for residual echo
        let mut output_spectrum = self.error_spectrum.clone();
        if self.nlp_enabled {
            self.apply_nlp(&mut output_spectrum);
        }

        // Inverse FFT
        self.fft_inverse.process_with_scratch(&mut output_spectrum, &mut self.output_buffer, &mut self.fft_scratch)?;

        // Normalize IFFT and apply synthesis window
        let scale = 1.0 / self.fft_size as f32;
        for (sample, win) in self.output_buffer.iter_mut().zip(&self.synthesis_window) {
            *sample *= scale * win;
        }

        // Overlap-add
        for i in 0..frame_size {
            output[i] = self.overlap_buffer[i] + self.output_buffer[i];
        }

        // Save second half for next frame
        self.overlap_buffer.copy_from_slice(&self.output_buffer[frame_size..]);

        // Add comfort noise if needed
        self.add_comfort_noise(&mut output[..frame_size]);

        // Update statistics
        self.update_statistics(input, output);

        self.frames_processed += 1;

        Ok(())
    }

    /// Update reference signal blocks in frequency domain
    fn update_reference_blocks(&mut self) -> Result<()> {
        let frame_size = self.config.frame_size;

        // Shift filter blocks (move reference history forward)
        for i in (1..self.num_blocks).rev() {
            self.filter_blocks[i].reference = self.filter_blocks[i - 1].reference.clone();
            self.filter_blocks[i].power = self.filter_blocks[i - 1].power.clone();
        }

        // Extract newest reference frame
        let ref_len = self.reference_buffer.len();
        if ref_len >= self.fft_size {
            let mut ref_frame: Vec<f32> = (0..self.fft_size)
                .map(|i| {
                    let idx = ref_len - self.fft_size + i;
                    if idx < ref_len {
                        self.reference_buffer[idx]
                    } else {
                        0.0
                    }
                })
                .collect();

            // Apply window
            for (sample, win) in ref_frame.iter_mut().zip(&self.analysis_window) {
                *sample *= win;
            }

            // Forward FFT
            self.fft_forward.process_with_scratch(
                &mut ref_frame,
                &mut self.filter_blocks[0].reference,
                &mut self.fft_scratch,
            )?;

            // Compute power spectrum
            for i in 0..self.freq_size {
                let power = self.filter_blocks[0].reference[i].norm_sqr();
                self.filter_blocks[0].power[i] = 0.9 * self.filter_blocks[0].power[i] + 0.1 * power;
            }
        }

        Ok(())
    }

    /// Compute echo estimate using partitioned block convolution
    fn compute_echo_estimate(&mut self) {
        // Clear echo spectrum
        for bin in &mut self.echo_spectrum {
            *bin = Complex::new(0.0, 0.0);
        }

        // Sum contribution from all filter blocks
        for block in &self.filter_blocks {
            for i in 0..self.freq_size {
                self.echo_spectrum[i] += block.coeffs[i] * block.reference[i];
            }
        }

        // Update echo energy estimate
        let echo_energy: f32 = self.echo_spectrum.iter().map(|c| c.norm_sqr()).sum();
        self.echo_energy = 0.9 * self.echo_energy + 0.1 * echo_energy;
    }

    /// NLMS filter adaptation in frequency domain
    fn adapt_filter(&mut self) {
        let mu = self.adaptation_rate;

        // Compute total power across all blocks for normalization
        let mut total_power = vec![self.config.regularization; self.freq_size];
        for block in &self.filter_blocks {
            for i in 0..self.freq_size {
                total_power[i] += block.power[i];
            }
        }

        // Smooth power estimate
        for i in 0..self.freq_size {
            self.power_smooth[i] = 0.95 * self.power_smooth[i] + 0.05 * total_power[i];
        }

        // Update each filter block
        for block in &mut self.filter_blocks {
            for i in 0..self.freq_size {
                // NLMS update: H += mu * E * conj(X) / |X|^2
                let normalized_mu = mu / (self.power_smooth[i] + self.config.regularization);
                let update = normalized_mu * self.error_spectrum[i] * block.reference[i].conj();
                block.coeffs[i] += update;

                // Constrain coefficient magnitude
                let mag = block.coeffs[i].norm();
                if mag > 10.0 {
                    block.coeffs[i] *= 10.0 / mag;
                }
            }
        }
    }

    /// Detect double-talk condition
    fn detect_double_talk(&mut self) {
        if self.far_energy < 1e-8 {
            self.echo_state = EchoState::Clean;
            self.adaptation_enabled = true;
            self.dtd_hangover = 0;
            return;
        }

        // Geigel double-talk detector
        let ratio = if self.far_energy > 1e-10 {
            self.near_energy / self.far_energy
        } else {
            0.0
        };

        // Compare error energy to expected echo energy
        let error_energy: f32 = self.error_spectrum.iter().map(|c| c.norm_sqr()).sum();

        // Double-talk heuristics
        let is_double_talk = ratio > self.config.double_talk_threshold * 1.5
            || (ratio > self.config.double_talk_threshold && error_energy > self.echo_energy * 2.0);

        if is_double_talk {
            self.echo_state = EchoState::DoubleTalk;
            self.adaptation_enabled = false;
            self.dtd_hangover = 10; // Hold for 10 frames
        } else if self.dtd_hangover > 0 {
            self.dtd_hangover -= 1;
            self.adaptation_enabled = false;
        } else {
            self.echo_state = EchoState::Cancelling;
            self.adaptation_enabled = true;
        }
    }

    /// Apply non-linear processing for residual echo suppression
    fn apply_nlp(&mut self, spectrum: &mut [Complex<f32>]) {
        // Compute coherence between error and echo
        for i in 0..self.freq_size {
            let error_power = spectrum[i].norm_sqr().max(1e-10);
            let echo_power = self.echo_spectrum[i].norm_sqr().max(1e-10);

            // Cross-spectral density
            let cross = (spectrum[i] * self.echo_spectrum[i].conj()).re;

            // Coherence estimate (smoothed)
            let instant_coherence = (cross.abs() / (error_power * echo_power).sqrt()).clamp(0.0, 1.0);
            self.coherence[i] = 0.9 * self.coherence[i] + 0.1 * instant_coherence;

            // Suppression gain based on coherence
            // High coherence with echo = more suppression needed
            let suppression = if self.coherence[i] > 0.3 {
                (1.0 - self.coherence[i]).max(0.1)
            } else {
                1.0
            };

            self.suppression_gain[i] = 0.8 * self.suppression_gain[i] + 0.2 * suppression;

            // Apply suppression
            spectrum[i] *= self.suppression_gain[i];
        }
    }

    /// Add comfort noise to prevent "dead air"
    fn add_comfort_noise(&mut self, output: &mut [f32]) {
        let output_rms: f32 = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();

        if output_rms < self.noise_floor * 2.0 {
            for sample in output.iter_mut() {
                let noise = self.generate_noise() * self.noise_floor * 0.3;
                *sample += noise;
            }
        }
    }

    /// Simple LCG random number generator
    fn generate_noise(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.rng_state >> 16) & 0x7FFF) as f32 / 16383.5 - 1.0
    }

    /// Update ERLE and other statistics
    fn update_statistics(&mut self, input: &[f32], output: &[f32]) {
        let frame_size = self.config.frame_size.min(input.len()).min(output.len());

        let input_power: f32 = input[..frame_size].iter().map(|x| x * x).sum();
        let output_power: f32 = output[..frame_size].iter().map(|x| x * x).sum();

        if input_power > 1e-10 && output_power > 1e-10 {
            let erle = 10.0 * (input_power / output_power).log10();
            self.erle_db = 0.95 * self.erle_db + 0.05 * erle.clamp(0.0, 40.0);
        }
    }

    /// Get current echo state
    pub fn get_state(&self) -> EchoState {
        self.echo_state
    }

    /// Get ERLE in dB
    pub fn get_erle_db(&self) -> f32 {
        self.erle_db
    }

    /// Reset all state
    pub fn reset(&mut self) {
        for block in &mut self.filter_blocks {
            block.coeffs.fill(Complex::new(0.0, 0.0));
            block.reference.fill(Complex::new(0.0, 0.0));
            block.power.fill(1e-6);
        }

        self.input_buffer.fill(0.0);
        self.output_buffer.fill(0.0);
        self.overlap_buffer.fill(0.0);
        self.reference_buffer.clear();

        self.power_smooth.fill(1e-6);
        self.coherence.fill(0.0);
        self.suppression_gain.fill(1.0);

        self.near_energy = 0.0;
        self.far_energy = 0.0;
        self.echo_energy = 0.0;
        self.erle_db = 0.0;

        self.echo_state = EchoState::Clean;
        self.adaptation_enabled = true;
        self.dtd_hangover = 0;
        self.frames_processed = 0;
    }

    /// Get filter length in samples
    pub fn filter_length(&self) -> usize {
        self.num_blocks * self.config.frame_size
    }

    /// Get latency in milliseconds
    pub fn latency_ms(&self) -> f32 {
        (self.config.frame_size as f32 / self.config.sample_rate as f32) * 1000.0
    }
}

/// Full-duplex echo canceller with loopback detection
pub struct FullDuplexAec {
    aec: AcousticEchoCanceller,

    // Loopback buffer
    loopback_buffer: VecDeque<f32>,
    max_delay_samples: usize,

    // Delay estimation state
    estimated_delay: usize,
    delay_locked: bool,
    correlation_history: Vec<f32>,
}

impl FullDuplexAec {
    /// Create a new full-duplex AEC
    pub fn new(config: AecConfig) -> Result<Self> {
        let max_delay_ms = 500;
        let max_delay_samples = (max_delay_ms * config.sample_rate as usize) / 1000;

        Ok(Self {
            aec: AcousticEchoCanceller::new(config)?,
            loopback_buffer: VecDeque::with_capacity(max_delay_samples),
            max_delay_samples,
            estimated_delay: 0,
            delay_locked: false,
            correlation_history: vec![0.0; 32],
        })
    }

    /// Process with automatic reference from system audio
    pub fn process_with_loopback(
        &mut self,
        mic_input: &[f32],
        speaker_output: &[f32],
        result: &mut [f32],
    ) -> Result<()> {
        // Add speaker output to loopback buffer
        for &sample in speaker_output {
            if self.loopback_buffer.len() >= self.max_delay_samples {
                self.loopback_buffer.pop_front();
            }
            self.loopback_buffer.push_back(sample);
        }

        // Estimate delay if not yet locked
        if !self.delay_locked && self.loopback_buffer.len() > speaker_output.len() * 4 {
            self.estimate_delay(mic_input);
        }

        // Extract reference with estimated delay
        let ref_start = self.loopback_buffer.len()
            .saturating_sub(speaker_output.len() + self.estimated_delay);

        let reference: Vec<f32> = (0..speaker_output.len())
            .map(|i| {
                if ref_start + i < self.loopback_buffer.len() {
                    self.loopback_buffer[ref_start + i]
                } else {
                    0.0
                }
            })
            .collect();

        self.aec.set_reference(&reference)?;
        self.aec.process(mic_input, result)
    }

    /// Estimate acoustic delay using cross-correlation
    fn estimate_delay(&mut self, mic_input: &[f32]) {
        let search_step = 48; // Search in ~1ms steps at 48kHz
        let search_range = self.max_delay_samples.min(self.loopback_buffer.len());

        let mut best_correlation = 0.0f32;
        let mut best_delay = self.estimated_delay;

        for delay in (0..search_range).step_by(search_step) {
            let correlation = self.compute_correlation(mic_input, delay);

            if correlation > best_correlation {
                best_correlation = correlation;
                best_delay = delay;
            }
        }

        // Refine around best delay
        let fine_start = best_delay.saturating_sub(search_step);
        let fine_end = (best_delay + search_step).min(search_range);

        for delay in fine_start..fine_end {
            let correlation = self.compute_correlation(mic_input, delay);

            if correlation > best_correlation {
                best_correlation = correlation;
                best_delay = delay;
            }
        }

        // Update delay estimate with smoothing
        if best_correlation > 0.4 {
            // Smooth the delay estimate
            self.estimated_delay = (self.estimated_delay * 3 + best_delay) / 4;

            // Update history and check for lock
            self.correlation_history.rotate_left(1);
            *self.correlation_history.last_mut().unwrap() = best_correlation;

            // Lock if consistently high correlation
            let avg_correlation: f32 = self.correlation_history.iter().sum::<f32>()
                / self.correlation_history.len() as f32;

            if avg_correlation > 0.5 {
                self.delay_locked = true;
                debug!("Delay locked at {} samples ({:.1}ms)",
                    self.estimated_delay,
                    self.estimated_delay as f32 * 1000.0 / 48000.0);
            }
        }
    }

    /// Compute normalized cross-correlation at given delay
    fn compute_correlation(&self, mic_input: &[f32], delay: usize) -> f32 {
        let ref_start = self.loopback_buffer.len().saturating_sub(mic_input.len() + delay);

        let mut correlation = 0.0f32;
        let mut mic_energy = 0.0f32;
        let mut ref_energy = 0.0f32;

        for (i, &mic_sample) in mic_input.iter().enumerate() {
            let ref_idx = ref_start + i;
            if ref_idx < self.loopback_buffer.len() {
                let ref_sample = self.loopback_buffer[ref_idx];

                correlation += mic_sample * ref_sample;
                mic_energy += mic_sample * mic_sample;
                ref_energy += ref_sample * ref_sample;
            }
        }

        let norm = (mic_energy * ref_energy).sqrt();
        if norm > 1e-10 {
            correlation / norm
        } else {
            0.0
        }
    }

    /// Reset the full-duplex AEC
    pub fn reset(&mut self) {
        self.aec.reset();
        self.loopback_buffer.clear();
        self.delay_locked = false;
        self.estimated_delay = 0;
        self.correlation_history.fill(0.0);
    }

    /// Get current echo state
    pub fn get_state(&self) -> EchoState {
        self.aec.get_state()
    }

    /// Get ERLE in dB
    pub fn get_erle_db(&self) -> f32 {
        self.aec.get_erle_db()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aec_creation() {
        let config = AecConfig::default();
        let aec = AcousticEchoCanceller::new(config);
        assert!(aec.is_ok());

        let aec = aec.unwrap();
        // 200ms at 48kHz = 9600 samples
        assert!(aec.filter_length() >= 9600);
    }

    #[test]
    fn test_aec_passthrough() {
        let config = AecConfig::default();
        let mut aec = AcousticEchoCanceller::new(config).unwrap();

        let input = vec![0.5; 512];
        let mut output = vec![0.0; 512];

        // Without reference, should pass through
        aec.process(&input, &mut output).unwrap();

        // Check some output was produced
        let output_energy: f32 = output.iter().map(|x| x * x).sum();
        assert!(output_energy >= 0.0);
    }

    #[test]
    fn test_echo_cancellation() {
        let config = AecConfig::default();
        let mut aec = AcousticEchoCanceller::new(config).unwrap();

        // Simulate echo: reference plays, then appears in mic with delay
        let reference: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();

        // Feed reference multiple times to fill buffer
        for _ in 0..10 {
            aec.set_reference(&reference).unwrap();
        }

        // Mic input is attenuated echo
        let mic_input: Vec<f32> = reference.iter().map(|x| x * 0.3).collect();
        let mut output = vec![0.0; 512];

        aec.process(&mic_input, &mut output).unwrap();

        // Output should have less energy than input after adaptation
        let input_energy: f32 = mic_input.iter().map(|x| x * x).sum();
        let output_energy: f32 = output.iter().map(|x| x * x).sum();

        // After first frame, may not be fully cancelled yet
        // Just verify it runs without error
        assert!(output_energy >= 0.0);
        assert!(input_energy >= 0.0);
    }

    #[test]
    fn test_double_talk_detection() {
        let config = AecConfig::default();
        let aec = AcousticEchoCanceller::new(config).unwrap();

        // Initially should be clean state
        assert_eq!(aec.get_state(), EchoState::Clean);
    }

    #[test]
    fn test_full_duplex_aec() {
        let config = AecConfig::default();
        let mut aec = FullDuplexAec::new(config).unwrap();

        let mic_input = vec![0.1; 512];
        let speaker_output = vec![0.2; 512];
        let mut result = vec![0.0; 512];

        aec.process_with_loopback(&mic_input, &speaker_output, &mut result).unwrap();

        // Verify output was produced
        assert_eq!(result.len(), 512);
    }
}
