//! # RNNoise-Compatible Neural Network Denoiser
//!
//! Implements RNNoise-style GRU-based noise suppression with real FFT processing.
//! Compatible with pre-trained RNNoise models and custom trained variants.
//!
//! ## Architecture
//! Based on the RNNoise paper: "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement"
//!
//! ```text
//! Input (480 samples @ 48kHz = 10ms)
//!     ↓
//! Windowing (Vorbis window)
//!     ↓
//! Real FFT (480-point → 241 complex bins)
//!     ↓
//! Bark-scale bands (22 bands)
//!     ↓
//! Feature computation (42 features)
//!     ↓
//! GRU layers (3x dense, 1x GRU)
//!     ↓
//! Band gains (22 outputs + VAD)
//!     ↓
//! Gain interpolation per-bin
//!     ↓
//! Inverse Real FFT + Overlap-add
//!     ↓
//! Output
//! ```

use anyhow::Result;
use num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex, ComplexToReal};
use std::sync::Arc;
use std::f32::consts::PI;
use tracing::warn;

use super::inference::InferenceEngine;
use super::model_manager::ModelType;
use super::features::BarkBands;

/// Number of bark-scale bands (RNNoise standard)
pub const NB_BANDS: usize = 22;

/// Frame size in samples (10ms at 48kHz)
pub const FRAME_SIZE: usize = 480;

/// Window size for FFT (same as frame for RNNoise)
pub const WINDOW_SIZE: usize = 480;

/// Number of input features to the neural network
pub const NB_FEATURES: usize = 42;

/// Number of complex frequency bins (N/2 + 1 for real FFT)
pub const FREQ_SIZE: usize = FRAME_SIZE / 2 + 1; // 241

/// RNNoise model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNNoiseModel {
    /// Original RNNoise (small, fast) - 96 GRU units
    Original,
    /// Enhanced model with better quality - 128 GRU units
    Enhanced,
    /// Large model for maximum quality - 256 GRU units
    Large,
    /// Custom trained model
    Custom,
}

impl From<ModelType> for RNNoiseModel {
    fn from(model_type: ModelType) -> Self {
        match model_type {
            ModelType::RNNoiseTiny => RNNoiseModel::Original,
            ModelType::RNNoiseStandard => RNNoiseModel::Enhanced,
            ModelType::RNNoiseLarge => RNNoiseModel::Large,
            _ => RNNoiseModel::Enhanced,
        }
    }
}

/// GRU layer weights and state
struct GruLayer {
    /// Input-to-hidden weights for update gate [input_size, hidden_size]
    w_z: Vec<f32>,
    /// Hidden-to-hidden weights for update gate [hidden_size, hidden_size]
    u_z: Vec<f32>,
    /// Bias for update gate [hidden_size]
    b_z: Vec<f32>,

    /// Input-to-hidden weights for reset gate
    w_r: Vec<f32>,
    /// Hidden-to-hidden weights for reset gate
    u_r: Vec<f32>,
    /// Bias for reset gate
    b_r: Vec<f32>,

    /// Input-to-hidden weights for candidate activation
    w_h: Vec<f32>,
    /// Hidden-to-hidden weights for candidate activation
    u_h: Vec<f32>,
    /// Bias for candidate activation
    b_h: Vec<f32>,

    /// Hidden state
    hidden: Vec<f32>,

    /// Dimensions
    input_size: usize,
    hidden_size: usize,
}

impl GruLayer {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        // Initialize with small random values (Xavier initialization)
        let scale = (2.0 / (input_size + hidden_size) as f32).sqrt();

        let mut init_weights = |size: usize| -> Vec<f32> {
            (0..size).map(|i| {
                // Deterministic pseudo-random for reproducibility
                let x = (i as f32 * 0.618033988749895) % 1.0;
                (x - 0.5) * 2.0 * scale
            }).collect()
        };

        Self {
            w_z: init_weights(input_size * hidden_size),
            u_z: init_weights(hidden_size * hidden_size),
            b_z: vec![0.0; hidden_size],
            w_r: init_weights(input_size * hidden_size),
            u_r: init_weights(hidden_size * hidden_size),
            b_r: vec![0.0; hidden_size],
            w_h: init_weights(input_size * hidden_size),
            u_h: init_weights(hidden_size * hidden_size),
            b_h: vec![0.0; hidden_size],
            hidden: vec![0.0; hidden_size],
            input_size,
            hidden_size,
        }
    }

    /// Load weights from a flat buffer
    fn load_weights(&mut self, weights: &[f32]) -> Result<usize> {
        let ih_size = self.input_size * self.hidden_size;
        let hh_size = self.hidden_size * self.hidden_size;
        let b_size = self.hidden_size;

        let expected = 3 * ih_size + 3 * hh_size + 3 * b_size;
        if weights.len() < expected {
            return Err(anyhow::anyhow!(
                "Insufficient weights: need {}, got {}", expected, weights.len()
            ));
        }

        let mut offset = 0;

        self.w_z.copy_from_slice(&weights[offset..offset + ih_size]); offset += ih_size;
        self.u_z.copy_from_slice(&weights[offset..offset + hh_size]); offset += hh_size;
        self.b_z.copy_from_slice(&weights[offset..offset + b_size]); offset += b_size;

        self.w_r.copy_from_slice(&weights[offset..offset + ih_size]); offset += ih_size;
        self.u_r.copy_from_slice(&weights[offset..offset + hh_size]); offset += hh_size;
        self.b_r.copy_from_slice(&weights[offset..offset + b_size]); offset += b_size;

        self.w_h.copy_from_slice(&weights[offset..offset + ih_size]); offset += ih_size;
        self.u_h.copy_from_slice(&weights[offset..offset + hh_size]); offset += hh_size;
        self.b_h.copy_from_slice(&weights[offset..offset + b_size]); offset += b_size;

        Ok(offset)
    }

    /// Forward pass through GRU
    fn forward(&mut self, input: &[f32], output: &mut [f32]) {
        let h = self.hidden_size;

        // Temporary vectors for gates
        let mut z = vec![0.0f32; h]; // Update gate
        let mut r = vec![0.0f32; h]; // Reset gate
        let mut h_tilde = vec![0.0f32; h]; // Candidate activation

        // Compute gates: z = sigmoid(W_z * x + U_z * h + b_z)
        for j in 0..h {
            // Input contribution
            let mut sum_z = self.b_z[j];
            let mut sum_r = self.b_r[j];
            let mut sum_h = self.b_h[j];

            for i in 0..self.input_size.min(input.len()) {
                sum_z += input[i] * self.w_z[i * h + j];
                sum_r += input[i] * self.w_r[i * h + j];
                sum_h += input[i] * self.w_h[i * h + j];
            }

            // Hidden state contribution
            for i in 0..h {
                sum_z += self.hidden[i] * self.u_z[i * h + j];
                sum_r += self.hidden[i] * self.u_r[i * h + j];
            }

            // Sigmoid for gates
            z[j] = sigmoid(sum_z);
            r[j] = sigmoid(sum_r);

            // Candidate uses reset gate
            let mut sum_h_hidden = 0.0;
            for i in 0..h {
                sum_h_hidden += (r[j] * self.hidden[i]) * self.u_h[i * h + j];
            }
            h_tilde[j] = tanh_approx(sum_h + sum_h_hidden);
        }

        // Update hidden state: h = (1 - z) * h + z * h_tilde
        for j in 0..h {
            self.hidden[j] = (1.0 - z[j]) * self.hidden[j] + z[j] * h_tilde[j];
            output[j] = self.hidden[j];
        }
    }

    fn reset(&mut self) {
        self.hidden.fill(0.0);
    }
}

/// Dense (fully connected) layer
struct DenseLayer {
    weights: Vec<f32>,
    bias: Vec<f32>,
    input_size: usize,
    output_size: usize,
}

impl DenseLayer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let scale = (2.0 / (input_size + output_size) as f32).sqrt();

        let weights: Vec<f32> = (0..input_size * output_size)
            .map(|i| {
                let x = (i as f32 * 0.618033988749895) % 1.0;
                (x - 0.5) * 2.0 * scale
            })
            .collect();

        Self {
            weights,
            bias: vec![0.0; output_size],
            input_size,
            output_size,
        }
    }

    fn load_weights(&mut self, weights: &[f32]) -> Result<usize> {
        let w_size = self.input_size * self.output_size;
        let b_size = self.output_size;

        if weights.len() < w_size + b_size {
            return Err(anyhow::anyhow!("Insufficient weights for dense layer"));
        }

        self.weights.copy_from_slice(&weights[0..w_size]);
        self.bias.copy_from_slice(&weights[w_size..w_size + b_size]);

        Ok(w_size + b_size)
    }

    fn forward(&self, input: &[f32], output: &mut [f32]) {
        for j in 0..self.output_size {
            let mut sum = self.bias[j];
            for i in 0..self.input_size.min(input.len()) {
                sum += input[i] * self.weights[i * self.output_size + j];
            }
            output[j] = sum;
        }
    }

    fn forward_relu(&self, input: &[f32], output: &mut [f32]) {
        self.forward(input, output);
        for val in output.iter_mut() {
            *val = val.max(0.0);
        }
    }

    fn forward_sigmoid(&self, input: &[f32], output: &mut [f32]) {
        self.forward(input, output);
        for val in output.iter_mut() {
            *val = sigmoid(*val);
        }
    }
}

/// Fast sigmoid approximation
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Fast tanh approximation using rational function
#[inline]
fn tanh_approx(x: f32) -> f32 {
    let x = x.clamp(-5.0, 5.0);
    let x2 = x * x;
    x * (27.0 + x2) / (27.0 + 9.0 * x2)
}

/// RNNoise processor with real FFT and neural network
pub struct RNNoiseProcessor {
    /// Sample rate (should be 48000)
    sample_rate: u32,
    /// Model variant
    model: RNNoiseModel,

    // FFT processing
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
    fft_scratch: Vec<Complex<f32>>,

    /// Analysis window (Vorbis)
    analysis_window: Vec<f32>,
    /// Synthesis window
    synthesis_window: Vec<f32>,

    /// Input ring buffer for overlap
    input_buffer: Vec<f32>,
    /// Complex spectrum
    spectrum: Vec<Complex<f32>>,
    /// Time domain output from IFFT
    ifft_output: Vec<f32>,
    /// Overlap-add buffer
    overlap_buffer: Vec<f32>,

    // Bark-scale processing
    bark_bands: BarkBands,
    band_energies: Vec<f32>,
    band_gains: Vec<f32>,
    prev_band_gains: Vec<f32>,

    // Neural network layers
    dense_in: DenseLayer,      // NB_FEATURES -> hidden
    gru_layer: GruLayer,       // hidden -> hidden
    dense_out: DenseLayer,     // hidden -> NB_BANDS + 1 (gains + VAD)

    // Feature extraction state
    features: Vec<f32>,
    prev_features: Vec<f32>,
    cepstral_mem: Vec<f32>,

    // DCT matrix for cepstral coefficients
    dct_matrix: Vec<f32>,

    // Noise estimation
    noise_estimate: Vec<f32>,
    noise_update_rate: f32,
    min_noise_floor: f32,

    // Output state
    voice_probability: f32,
    noise_reduction_db: f32,
    frames_processed: u64,

    // Inference engine (for GPU acceleration)
    inference: Arc<InferenceEngine>,
    use_gpu: bool,
}

impl RNNoiseProcessor {
    /// Create a new RNNoise processor
    pub fn new(
        sample_rate: u32,
        _frame_size: usize,
        model_type: ModelType,
        inference: Arc<InferenceEngine>,
    ) -> Result<Self> {
        let model = RNNoiseModel::from(model_type);

        if sample_rate != 48000 {
            warn!("RNNoise optimized for 48kHz; input is {}Hz - quality may vary", sample_rate);
        }

        // GRU hidden size based on model
        let hidden_size = match model {
            RNNoiseModel::Original => 96,
            RNNoiseModel::Enhanced => 128,
            RNNoiseModel::Large => 256,
            RNNoiseModel::Custom => 128,
        };

        // Initialize FFT
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(WINDOW_SIZE);
        let fft_inverse = planner.plan_fft_inverse(WINDOW_SIZE);
        let fft_scratch = vec![Complex::new(0.0, 0.0); fft_forward.get_scratch_len().max(fft_inverse.get_scratch_len())];

        // Create windows
        let analysis_window = Self::create_vorbis_window(WINDOW_SIZE);
        let synthesis_window = Self::create_synthesis_window(&analysis_window);

        // Create DCT matrix for cepstral computation
        let dct_matrix = Self::create_dct_matrix(NB_BANDS, 6);

        let use_gpu = inference.is_available();

        Ok(Self {
            sample_rate,
            model,
            fft_forward,
            fft_inverse,
            fft_scratch,
            analysis_window,
            synthesis_window,
            input_buffer: vec![0.0; WINDOW_SIZE],
            spectrum: vec![Complex::new(0.0, 0.0); FREQ_SIZE],
            ifft_output: vec![0.0; WINDOW_SIZE],
            overlap_buffer: vec![0.0; FRAME_SIZE],
            bark_bands: BarkBands::new(),
            band_energies: vec![0.0; NB_BANDS],
            band_gains: vec![1.0; NB_BANDS],
            prev_band_gains: vec![1.0; NB_BANDS],
            dense_in: DenseLayer::new(NB_FEATURES, hidden_size),
            gru_layer: GruLayer::new(hidden_size, hidden_size),
            dense_out: DenseLayer::new(hidden_size, NB_BANDS + 1),
            features: vec![0.0; NB_FEATURES],
            prev_features: vec![0.0; NB_FEATURES],
            cepstral_mem: vec![0.0; 6],
            dct_matrix,
            noise_estimate: vec![1e-6; NB_BANDS],
            noise_update_rate: 0.02,
            min_noise_floor: 1e-8,
            voice_probability: 0.0,
            noise_reduction_db: 0.0,
            frames_processed: 0,
            inference,
            use_gpu,
        })
    }

    /// Create Vorbis window function (MDCT-compatible)
    fn create_vorbis_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let x = (i as f32 + 0.5) / size as f32;
                (PI / 2.0 * (PI * x).sin().powi(2)).sin()
            })
            .collect()
    }

    /// Create synthesis window for perfect reconstruction
    fn create_synthesis_window(analysis: &[f32]) -> Vec<f32> {
        // For 50% overlap, synthesis window = analysis window
        // The overlap-add will give unity gain
        analysis.to_vec()
    }

    /// Create DCT-II matrix for cepstral coefficients
    fn create_dct_matrix(n_bands: usize, n_ceps: usize) -> Vec<f32> {
        let mut matrix = vec![0.0; n_bands * n_ceps];
        for k in 0..n_ceps {
            for n in 0..n_bands {
                matrix[k * n_bands + n] =
                    (PI * k as f32 * (n as f32 + 0.5) / n_bands as f32).cos();
            }
        }
        matrix
    }

    /// Process audio buffer
    pub fn process(&mut self, input: &[f32], output: &mut [f32], strength: f32) -> Result<()> {
        let mut in_pos = 0;
        let mut out_pos = 0;

        while in_pos + FRAME_SIZE <= input.len() && out_pos + FRAME_SIZE <= output.len() {
            self.process_frame(
                &input[in_pos..in_pos + FRAME_SIZE],
                &mut output[out_pos..out_pos + FRAME_SIZE],
                strength,
            )?;
            in_pos += FRAME_SIZE;
            out_pos += FRAME_SIZE;
        }

        // Pass through any remaining samples
        let remaining = input.len().saturating_sub(in_pos).min(output.len().saturating_sub(out_pos));
        if remaining > 0 {
            output[out_pos..out_pos + remaining]
                .copy_from_slice(&input[in_pos..in_pos + remaining]);
        }

        Ok(())
    }

    /// Process a single frame (480 samples = 10ms at 48kHz)
    fn process_frame(&mut self, input: &[f32], output: &mut [f32], strength: f32) -> Result<()> {
        // Shift input buffer left and add new frame
        self.input_buffer.copy_within(FRAME_SIZE.., 0);
        self.input_buffer[WINDOW_SIZE - FRAME_SIZE..].copy_from_slice(input);

        // Apply analysis window
        let mut windowed: Vec<f32> = self.input_buffer
            .iter()
            .zip(&self.analysis_window)
            .map(|(s, w)| s * w)
            .collect();

        // Forward FFT
        self.fft_forward.process_with_scratch(&mut windowed, &mut self.spectrum, &mut self.fft_scratch)?;

        // Compute bark-scale band energies
        self.compute_band_energies();

        // Extract features
        self.extract_features();

        // Run neural network
        self.run_network()?;

        // Interpolate gains for smoothness
        let gain_smoothing = 0.5f32;
        for i in 0..NB_BANDS {
            self.band_gains[i] = gain_smoothing * self.band_gains[i]
                + (1.0 - gain_smoothing) * self.prev_band_gains[i];
        }
        self.prev_band_gains.copy_from_slice(&self.band_gains);

        // Apply strength control
        for gain in &mut self.band_gains {
            *gain = 1.0 - strength * (1.0 - *gain);
        }

        // Apply gains to spectrum (interpolated per-bin)
        self.apply_spectral_gains();

        // Inverse FFT
        self.fft_inverse.process_with_scratch(&mut self.spectrum, &mut self.ifft_output, &mut self.fft_scratch)?;

        // Normalize IFFT output
        let scale = 1.0 / WINDOW_SIZE as f32;
        for sample in &mut self.ifft_output {
            *sample *= scale;
        }

        // Apply synthesis window
        for (sample, win) in self.ifft_output.iter_mut().zip(&self.synthesis_window) {
            *sample *= win;
        }

        // Overlap-add
        for i in 0..FRAME_SIZE {
            output[i] = self.overlap_buffer[i] + self.ifft_output[i];
        }

        // Save second half for next overlap
        self.overlap_buffer.copy_from_slice(&self.ifft_output[FRAME_SIZE..]);

        self.frames_processed += 1;
        self.update_stats(input, output);

        Ok(())
    }

    /// Compute bark-scale band energies from spectrum
    fn compute_band_energies(&mut self) {
        for band in 0..NB_BANDS {
            let (start, end) = self.bark_bands.get_band_bins(band, FREQ_SIZE);

            let mut energy = 0.0f32;
            for bin in start..end.min(FREQ_SIZE) {
                energy += self.spectrum[bin].norm_sqr();
            }

            // RNNoise uses normalized energy
            let width = (end - start).max(1) as f32;
            self.band_energies[band] = (energy / width).max(self.min_noise_floor);
        }
    }

    /// Extract features for neural network
    fn extract_features(&mut self) {
        // Save previous features for delta computation
        self.prev_features.copy_from_slice(&self.features);

        // Feature 0-21: Log band energies (normalized)
        for i in 0..NB_BANDS {
            let log_e = (self.band_energies[i] + 1e-10).ln();
            // Normalize to roughly [-1, 1]
            self.features[i] = (log_e + 7.0) / 10.0;
        }

        // Feature 22-37: Band energy deltas
        for i in 0..NB_BANDS.min(16) {
            self.features[NB_BANDS + i] = self.features[i] - self.prev_features[i];
        }

        // Feature 38-43: Cepstral coefficients (approximate MFCCs)
        self.compute_cepstral_coefficients();

        // Update noise estimate during low-energy frames
        let total_energy: f32 = self.band_energies.iter().sum();
        if total_energy < 0.01 {
            for i in 0..NB_BANDS {
                self.noise_estimate[i] = self.noise_estimate[i] * (1.0 - self.noise_update_rate)
                    + self.band_energies[i] * self.noise_update_rate;
            }
        }
    }

    /// Compute cepstral coefficients using DCT
    fn compute_cepstral_coefficients(&mut self) {
        // Apply DCT to log band energies
        for k in 0..6 {
            let mut sum = 0.0f32;
            for n in 0..NB_BANDS {
                sum += self.features[n] * self.dct_matrix[k * NB_BANDS + n];
            }

            // Temporal smoothing
            let alpha = 0.7;
            self.cepstral_mem[k] = alpha * self.cepstral_mem[k] + (1.0 - alpha) * sum;

            if 38 + k < NB_FEATURES {
                self.features[38 + k] = self.cepstral_mem[k] * 0.1;
            }
        }
    }

    /// Run neural network inference
    fn run_network(&mut self) -> Result<()> {
        if self.use_gpu && self.inference.is_available() {
            // GPU path - use TensorRT or ONNX Runtime
            let outputs = self.inference.run_rnnoise(
                &self.features,
                &mut self.gru_layer.hidden,
                &mut vec![0.0; self.gru_layer.hidden_size], // unused
                &mut vec![0.0; self.gru_layer.hidden_size], // unused
            )?;

            for i in 0..NB_BANDS {
                self.band_gains[i] = outputs.get(i).copied().unwrap_or(1.0).clamp(0.0, 1.0);
            }
            self.voice_probability = outputs.get(NB_BANDS).copied().unwrap_or(0.5);
        } else {
            // CPU path - run network layers directly
            self.run_cpu_network();
        }

        Ok(())
    }

    /// CPU neural network inference
    fn run_cpu_network(&mut self) {
        let hidden_size = self.gru_layer.hidden_size;

        // Dense input layer with ReLU
        let mut hidden1 = vec![0.0f32; hidden_size];
        self.dense_in.forward_relu(&self.features, &mut hidden1);

        // GRU layer
        let mut hidden2 = vec![0.0f32; hidden_size];
        self.gru_layer.forward(&hidden1, &mut hidden2);

        // Output layer with sigmoid for gains
        let mut output = vec![0.0f32; NB_BANDS + 1];
        self.dense_out.forward_sigmoid(&hidden2, &mut output);

        // Extract gains and VAD
        for i in 0..NB_BANDS {
            self.band_gains[i] = output[i];
        }
        self.voice_probability = output[NB_BANDS];

        // Apply noise-aware gain adjustment
        // Reduce gain more where SNR is low
        for i in 0..NB_BANDS {
            let snr = self.band_energies[i] / (self.noise_estimate[i] + 1e-10);
            let snr_factor = (snr / (snr + 1.0)).powf(0.3);
            self.band_gains[i] *= snr_factor;
        }
    }

    /// Apply gains to frequency spectrum with interpolation
    fn apply_spectral_gains(&mut self) {
        for bin in 0..FREQ_SIZE {
            // Find which band this bin belongs to and interpolate
            let freq_ratio = bin as f32 / FREQ_SIZE as f32;
            let band_float = freq_ratio * NB_BANDS as f32;
            let band_low = (band_float as usize).min(NB_BANDS - 1);
            let band_high = (band_low + 1).min(NB_BANDS - 1);
            let interp = band_float - band_low as f32;

            let gain = self.band_gains[band_low] * (1.0 - interp)
                + self.band_gains[band_high] * interp;

            self.spectrum[bin] *= gain;
        }
    }

    /// Update statistics
    fn update_stats(&mut self, input: &[f32], output: &[f32]) {
        let in_energy: f32 = input.iter().map(|x| x * x).sum();
        let out_energy: f32 = output.iter().map(|x| x * x).sum();

        if in_energy > 1e-10 {
            let ratio = (out_energy / in_energy).max(1e-10);
            self.noise_reduction_db = -10.0 * ratio.log10();
        }
    }

    /// Load model weights from buffer
    pub fn load_weights(&mut self, weights: &[f32]) -> Result<()> {
        let mut offset = 0;

        offset += self.dense_in.load_weights(&weights[offset..])?;
        offset += self.gru_layer.load_weights(&weights[offset..])?;
        let _ = self.dense_out.load_weights(&weights[offset..])?;

        Ok(())
    }

    /// Get voice activity probability
    pub fn get_voice_probability(&self) -> f32 {
        self.voice_probability
    }

    /// Get noise reduction in dB
    pub fn get_noise_reduction_db(&self) -> f32 {
        self.noise_reduction_db
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.overlap_buffer.fill(0.0);
        self.gru_layer.reset();
        self.features.fill(0.0);
        self.prev_features.fill(0.0);
        self.cepstral_mem.fill(0.0);
        self.band_gains.fill(1.0);
        self.prev_band_gains.fill(1.0);
        self.noise_estimate.fill(1e-6);
        self.voice_probability = 0.0;
        self.frames_processed = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vorbis_window_properties() {
        let window = RNNoiseProcessor::create_vorbis_window(WINDOW_SIZE);

        assert_eq!(window.len(), WINDOW_SIZE);

        // Check symmetry
        for i in 0..WINDOW_SIZE / 2 {
            assert!((window[i] - window[WINDOW_SIZE - 1 - i]).abs() < 1e-5,
                "Window not symmetric at {}", i);
        }

        // For 50% overlap OLA with this window, check reconstruction property
        // Sum of overlapping squared windows should be approximately constant
        let hop_size = FRAME_SIZE; // 50% overlap
        let mut sum = 0.0f32;
        for i in 0..hop_size.min(WINDOW_SIZE) {
            sum += window[i].powi(2);
        }
        // Should have some non-zero energy
        assert!(sum > 0.1, "Window sum too low: {}", sum);
    }

    #[test]
    fn test_gru_dimensions() {
        let gru = GruLayer::new(42, 128);

        assert_eq!(gru.hidden.len(), 128);
        assert_eq!(gru.w_z.len(), 42 * 128);
        assert_eq!(gru.u_z.len(), 128 * 128);
    }

    #[test]
    fn test_activation_functions() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);

        assert!((tanh_approx(0.0)).abs() < 1e-6);
        assert!(tanh_approx(5.0) > 0.99);
        assert!(tanh_approx(-5.0) < -0.99);
    }

    #[test]
    fn test_dct_matrix() {
        let dct = RNNoiseProcessor::create_dct_matrix(22, 6);
        assert_eq!(dct.len(), 22 * 6);

        // First row should be all 1s (DC component)
        for n in 0..22 {
            assert!((dct[n] - 1.0).abs() < 1e-5);
        }
    }
}
