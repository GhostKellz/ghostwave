//! # Audio Feature Extraction
//!
//! Provides feature extraction for neural network input:
//! - Bark-scale frequency bands
//! - Spectral features (energy, flux, centroid)
//! - Cepstral coefficients
//! - Pitch estimation

use std::f32::consts::PI;

/// Number of bark-scale bands
pub const NB_BANDS: usize = 22;

/// Bark-scale frequency band mapping
///
/// Maps FFT bins to perceptually-spaced bark bands.
/// Based on the bark scale which approximates human auditory perception.
pub struct BarkBands {
    /// Center frequencies for each band (Hz)
    center_freqs: [f32; NB_BANDS],
    /// Bandwidth of each band (Hz)
    bandwidths: [f32; NB_BANDS],
}

impl BarkBands {
    /// Create new bark band mapper
    pub fn new() -> Self {
        // Bark scale band center frequencies (Hz)
        // These are standard RNNoise bark bands
        let center_freqs = [
            50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 510.0, 630.0,
            770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0,
            2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0,
        ];

        // Approximate bandwidths (Hz)
        let bandwidths = [
            100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 120.0, 140.0,
            150.0, 160.0, 190.0, 210.0, 240.0, 280.0, 320.0, 380.0,
            450.0, 550.0, 700.0, 900.0, 1100.0, 2200.0,
        ];

        Self {
            center_freqs,
            bandwidths,
        }
    }

    /// Get the FFT bin range for a bark band
    pub fn get_band_bins(&self, band: usize, fft_size: usize) -> (usize, usize) {
        if band >= NB_BANDS {
            return (0, 0);
        }

        let sample_rate = 48000.0;
        let bin_hz = sample_rate / (2.0 * fft_size as f32);

        let center = self.center_freqs[band];
        let bandwidth = self.bandwidths[band];

        let start_freq = (center - bandwidth / 2.0).max(0.0);
        let end_freq = center + bandwidth / 2.0;

        let start_bin = (start_freq / bin_hz).floor() as usize;
        let end_bin = (end_freq / bin_hz).ceil() as usize;

        (start_bin.min(fft_size - 1), end_bin.min(fft_size))
    }

    /// Map an FFT bin to its bark band
    pub fn bin_to_band(&self, bin: usize, fft_size: usize) -> usize {
        let sample_rate = 48000.0;
        let bin_hz = sample_rate / (2.0 * fft_size as f32);
        let freq = bin as f32 * bin_hz;

        // Find the band that contains this frequency
        for band in 0..NB_BANDS {
            let center = self.center_freqs[band];
            let bandwidth = self.bandwidths[band];
            let low = center - bandwidth / 2.0;
            let high = center + bandwidth / 2.0;

            if freq >= low && freq < high {
                return band;
            }
        }

        // Default to highest band
        NB_BANDS - 1
    }

    /// Convert frequency to bark scale
    pub fn hz_to_bark(freq: f32) -> f32 {
        // Traunmüller formula
        ((freq / 600.0) + 1.0).ln() * 6.0
    }

    /// Convert bark to frequency
    pub fn bark_to_hz(bark: f32) -> f32 {
        600.0 * ((bark / 6.0).exp() - 1.0)
    }
}

impl Default for BarkBands {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature extractor for neural network input
#[allow(dead_code)] // Public API - feature extraction state
pub struct FeatureExtractor {
    sample_rate: u32,
    fft_size: usize,

    // DCT matrix for cepstral coefficients
    dct_matrix: Vec<Vec<f32>>,

    // Previous frame state for delta features
    prev_energies: Vec<f32>,
    prev_prev_energies: Vec<f32>,

    // Pitch estimation state
    autocorr_buffer: Vec<f32>,
}

impl FeatureExtractor {
    /// Create new feature extractor
    pub fn new(sample_rate: u32, fft_size: usize) -> Self {
        // Pre-compute DCT-II matrix for cepstral coefficients
        let n_ceps = 13;
        let n_bands = NB_BANDS;
        let mut dct_matrix = vec![vec![0.0; n_bands]; n_ceps];

        for k in 0..n_ceps {
            for n in 0..n_bands {
                dct_matrix[k][n] = (PI * k as f32 * (n as f32 + 0.5) / n_bands as f32).cos();
            }
        }

        Self {
            sample_rate,
            fft_size,
            dct_matrix,
            prev_energies: vec![0.0; NB_BANDS],
            prev_prev_energies: vec![0.0; NB_BANDS],
            autocorr_buffer: vec![0.0; fft_size],
        }
    }

    /// Extract all features from a frame
    pub fn extract_features(
        &mut self,
        band_energies: &[f32],
        fft_real: &[f32],
        fft_imag: &[f32],
    ) -> Vec<f32> {
        let mut features = Vec::with_capacity(64);

        // 1. Log band energies (22 features)
        let log_energies: Vec<f32> = band_energies
            .iter()
            .map(|&e| (e.max(1e-10)).ln())
            .collect();
        features.extend(&log_energies);

        // 2. Delta band energies (22 features)
        let deltas: Vec<f32> = log_energies
            .iter()
            .zip(self.prev_energies.iter())
            .map(|(curr, prev)| curr - prev)
            .collect();
        features.extend(&deltas);

        // 3. Delta-delta energies (22 features, optional)
        // Skipped for lower latency

        // 4. Cepstral coefficients (13 features)
        let mfccs = self.compute_mfcc(&log_energies);
        features.extend(&mfccs);

        // 5. Spectral features (4 features)
        let spectral_centroid = self.compute_spectral_centroid(fft_real, fft_imag);
        let spectral_flux = self.compute_spectral_flux(band_energies);
        let spectral_rolloff = self.compute_spectral_rolloff(band_energies);
        let spectral_flatness = self.compute_spectral_flatness(band_energies);

        features.push(spectral_centroid);
        features.push(spectral_flux);
        features.push(spectral_rolloff);
        features.push(spectral_flatness);

        // Update state
        self.prev_prev_energies.copy_from_slice(&self.prev_energies);
        self.prev_energies.copy_from_slice(&log_energies);

        features
    }

    /// Compute MFCC (Mel-Frequency Cepstral Coefficients)
    fn compute_mfcc(&self, log_energies: &[f32]) -> Vec<f32> {
        let mut mfccs = vec![0.0; self.dct_matrix.len()];

        for (k, mfcc) in mfccs.iter_mut().enumerate() {
            for (n, &energy) in log_energies.iter().enumerate() {
                if n < self.dct_matrix[k].len() {
                    *mfcc += energy * self.dct_matrix[k][n];
                }
            }
        }

        mfccs
    }

    /// Compute spectral centroid (center of mass of spectrum)
    fn compute_spectral_centroid(&self, fft_real: &[f32], fft_imag: &[f32]) -> f32 {
        let mut weighted_sum = 0.0_f32;
        let mut magnitude_sum = 0.0_f32;

        for (i, (real, imag)) in fft_real.iter().zip(fft_imag.iter()).enumerate() {
            let magnitude = (real * real + imag * imag).sqrt();
            let freq = i as f32 * self.sample_rate as f32 / (2.0 * fft_real.len() as f32);

            weighted_sum += freq * magnitude;
            magnitude_sum += magnitude;
        }

        if magnitude_sum > 1e-10 {
            weighted_sum / magnitude_sum / self.sample_rate as f32 // Normalize
        } else {
            0.0
        }
    }

    /// Compute spectral flux (change from previous frame)
    fn compute_spectral_flux(&self, band_energies: &[f32]) -> f32 {
        let flux: f32 = band_energies
            .iter()
            .zip(self.prev_energies.iter())
            .map(|(&curr, &prev)| {
                let log_curr = (curr.max(1e-10)).ln();
                let diff = log_curr - prev;
                diff * diff
            })
            .sum();

        (flux / NB_BANDS as f32).sqrt()
    }

    /// Compute spectral rolloff (frequency below which X% of energy is contained)
    fn compute_spectral_rolloff(&self, band_energies: &[f32]) -> f32 {
        let total_energy: f32 = band_energies.iter().sum();
        let threshold = total_energy * 0.85;

        let mut cumulative = 0.0_f32;
        for (i, &energy) in band_energies.iter().enumerate() {
            cumulative += energy;
            if cumulative >= threshold {
                return i as f32 / NB_BANDS as f32;
            }
        }

        1.0
    }

    /// Compute spectral flatness (tonality measure)
    fn compute_spectral_flatness(&self, band_energies: &[f32]) -> f32 {
        let n = band_energies.len() as f32;

        // Geometric mean
        let log_sum: f32 = band_energies
            .iter()
            .map(|&e| (e.max(1e-10)).ln())
            .sum();
        let geometric_mean = (log_sum / n).exp();

        // Arithmetic mean
        let arithmetic_mean: f32 = band_energies.iter().sum::<f32>() / n;

        if arithmetic_mean > 1e-10 {
            (geometric_mean / arithmetic_mean).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Estimate pitch using autocorrelation
    pub fn estimate_pitch(&mut self, samples: &[f32]) -> Option<f32> {
        let min_period = (self.sample_rate as f32 / 400.0) as usize; // Max 400 Hz
        let max_period = (self.sample_rate as f32 / 60.0) as usize;  // Min 60 Hz

        if samples.len() < max_period * 2 {
            return None;
        }

        // Compute autocorrelation
        let mut best_period = 0;
        let mut best_correlation = 0.0_f32;

        for period in min_period..max_period.min(samples.len() / 2) {
            let mut correlation = 0.0_f32;
            let mut energy1 = 0.0_f32;
            let mut energy2 = 0.0_f32;

            for i in 0..samples.len() - period {
                correlation += samples[i] * samples[i + period];
                energy1 += samples[i] * samples[i];
                energy2 += samples[i + period] * samples[i + period];
            }

            // Normalized correlation
            let norm = (energy1 * energy2).sqrt();
            if norm > 1e-10 {
                correlation /= norm;
            }

            if correlation > best_correlation {
                best_correlation = correlation;
                best_period = period;
            }
        }

        // Require minimum correlation for valid pitch
        if best_correlation > 0.5 && best_period > 0 {
            Some(self.sample_rate as f32 / best_period as f32)
        } else {
            None
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.prev_energies.fill(0.0);
        self.prev_prev_energies.fill(0.0);
    }
}

/// Voice Activity Detection features
pub struct VadFeatures {
    /// Short-term energy
    pub energy: f32,
    /// Zero crossing rate
    pub zcr: f32,
    /// Spectral entropy
    pub spectral_entropy: f32,
    /// High-frequency ratio
    pub hf_ratio: f32,
    /// Pitch confidence
    pub pitch_confidence: f32,
}

impl VadFeatures {
    /// Extract VAD-specific features from audio
    pub fn extract(samples: &[f32], band_energies: &[f32]) -> Self {
        // Energy
        let energy = samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32;

        // Zero crossing rate
        let mut zcr = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                zcr += 1;
            }
        }
        let zcr = zcr as f32 / samples.len() as f32;

        // Spectral entropy
        let total_energy: f32 = band_energies.iter().sum::<f32>().max(1e-10);
        let entropy: f32 = band_energies
            .iter()
            .map(|&e| {
                let p = e / total_energy;
                if p > 1e-10 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();
        let spectral_entropy = entropy / (NB_BANDS as f32).ln(); // Normalize

        // High frequency ratio (useful for detecting fricatives vs silence)
        let low_energy: f32 = band_energies[0..8].iter().sum();
        let high_energy: f32 = band_energies[8..].iter().sum();
        let hf_ratio = if low_energy > 1e-10 {
            high_energy / low_energy
        } else {
            0.0
        };

        Self {
            energy,
            zcr,
            spectral_entropy,
            hf_ratio,
            pitch_confidence: 0.0, // Computed separately
        }
    }

    /// Simple VAD decision
    pub fn is_voice(&self) -> bool {
        // Heuristic thresholds
        let energy_threshold = 1e-4;
        let zcr_threshold = 0.3;
        let entropy_threshold = 0.5;

        self.energy > energy_threshold
            && self.zcr < zcr_threshold
            && self.spectral_entropy > entropy_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bark_scale() {
        // Test Hz to bark conversion
        let bark_1000 = BarkBands::hz_to_bark(1000.0);
        assert!(bark_1000 > 0.0 && bark_1000 < 24.0);

        // Test round-trip
        let hz_back = BarkBands::bark_to_hz(bark_1000);
        assert!((hz_back - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_band_bins() {
        let bands = BarkBands::new();

        // Test first band
        let (start, end) = bands.get_band_bins(0, 241);
        assert!(start < end);

        // Test last band
        let (start, end) = bands.get_band_bins(NB_BANDS - 1, 241);
        assert!(start < end);
    }

    #[test]
    fn test_feature_extractor() {
        let mut extractor = FeatureExtractor::new(48000, 480);

        let band_energies = vec![0.1; NB_BANDS];
        let fft_real = vec![0.1; 241];
        let fft_imag = vec![0.0; 241];

        let features = extractor.extract_features(&band_energies, &fft_real, &fft_imag);

        // Should have at least bark bands + deltas + mfcc + spectral features
        assert!(features.len() >= NB_BANDS * 2 + 13 + 4);
    }

    #[test]
    fn test_vad_features() {
        // Silence
        let silence = vec![0.0; 480];
        let silent_energies = vec![0.0; NB_BANDS];
        let vad = VadFeatures::extract(&silence, &silent_energies);
        assert!(!vad.is_voice());

        // Loud signal
        let loud: Vec<f32> = (0..480).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let loud_energies = vec![0.1; NB_BANDS];
        let _vad = VadFeatures::extract(&loud, &loud_energies);
        // This might or might not be detected as voice depending on the signal
    }
}
