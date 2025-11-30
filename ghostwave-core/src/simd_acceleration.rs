//! # SIMD/AVX2 CPU Acceleration
//!
//! Provides vectorized operations for audio processing using SIMD instructions.
//! Falls back gracefully to scalar implementations on unsupported hardware.

use anyhow::Result;
use std::arch::x86_64::*;
use tracing::{info, debug, warn};

/// CPU feature detection and SIMD capabilities
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub has_sse: bool,
    pub has_sse2: bool,
    pub has_sse3: bool,
    pub has_ssse3: bool,
    pub has_sse41: bool,
    pub has_sse42: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_fma: bool,
    pub has_avx512f: bool,
}

impl SimdCapabilities {
    /// Detect CPU SIMD capabilities at runtime
    pub fn detect() -> Self {
        Self {
            has_sse: is_x86_feature_detected!("sse"),
            has_sse2: is_x86_feature_detected!("sse2"),
            has_sse3: is_x86_feature_detected!("sse3"),
            has_ssse3: is_x86_feature_detected!("ssse3"),
            has_sse41: is_x86_feature_detected!("sse4.1"),
            has_sse42: is_x86_feature_detected!("sse4.2"),
            has_avx: is_x86_feature_detected!("avx"),
            has_avx2: is_x86_feature_detected!("avx2"),
            has_fma: is_x86_feature_detected!("fma"),
            has_avx512f: is_x86_feature_detected!("avx512f"),
        }
    }

    /// Get the best available instruction set
    pub fn best_instruction_set(&self) -> InstructionSet {
        if self.has_avx512f {
            InstructionSet::AVX512
        } else if self.has_avx2 && self.has_fma {
            InstructionSet::Avx2Fma
        } else if self.has_avx2 {
            InstructionSet::AVX2
        } else if self.has_avx {
            InstructionSet::AVX
        } else if self.has_sse42 {
            InstructionSet::SSE42
        } else if self.has_sse2 {
            InstructionSet::SSE2
        } else {
            InstructionSet::Scalar
        }
    }

    /// Get optimal vector width in samples
    pub fn optimal_vector_width(&self) -> usize {
        match self.best_instruction_set() {
            InstructionSet::AVX512 => 16, // 512 bits / 32 bits per f32
            InstructionSet::Avx2Fma | InstructionSet::AVX2 | InstructionSet::AVX => 8, // 256 bits / 32 bits
            InstructionSet::SSE42 | InstructionSet::SSE2 => 4, // 128 bits / 32 bits
            InstructionSet::Scalar => 1,
        }
    }

    pub fn report(&self) {
        let best = self.best_instruction_set();
        let width = self.optimal_vector_width();

        info!("🚀 SIMD Capabilities Detected:");
        info!("  Best instruction set: {:?}", best);
        info!("  Optimal vector width: {} samples", width);
        info!("  SSE2: {}, AVX: {}, AVX2: {}, FMA: {}, AVX512F: {}",
              self.has_sse2, self.has_avx, self.has_avx2, self.has_fma, self.has_avx512f);
    }
}

/// Supported instruction sets in order of preference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum InstructionSet {
    #[default]
    Scalar,
    SSE2,
    SSE42,
    AVX,
    AVX2,
    Avx2Fma,
    AVX512,
}

impl std::fmt::Display for InstructionSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstructionSet::Scalar => write!(f, "Scalar (no SIMD)"),
            InstructionSet::SSE2 => write!(f, "SSE2"),
            InstructionSet::SSE42 => write!(f, "SSE4.2"),
            InstructionSet::AVX => write!(f, "AVX"),
            InstructionSet::AVX2 => write!(f, "AVX2"),
            InstructionSet::Avx2Fma => write!(f, "AVX2+FMA"),
            InstructionSet::AVX512 => write!(f, "AVX-512"),
        }
    }
}

/// SIMD-accelerated audio processing operations
pub struct SimdProcessor {
    capabilities: SimdCapabilities,
    instruction_set: InstructionSet,
    vector_width: usize,
}

impl SimdProcessor {
    pub fn new() -> Self {
        let capabilities = SimdCapabilities::detect();
        let instruction_set = capabilities.best_instruction_set();
        let vector_width = capabilities.optimal_vector_width();

        capabilities.report();

        Self {
            capabilities,
            instruction_set,
            vector_width,
        }
    }

    /// Vectorized audio buffer operations
    pub fn process_buffer(&self, input: &[f32], output: &mut [f32], operation: BufferOperation) -> Result<()> {
        if input.len() != output.len() {
            return Err(anyhow::anyhow!("Input and output buffer length mismatch"));
        }

        match self.instruction_set {
            InstructionSet::Avx2Fma | InstructionSet::AVX2 => {
                // SAFETY: process_avx2 requires AVX2 target feature which we've detected
                unsafe { self.process_avx2(input, output, operation) }
            }
            InstructionSet::AVX => {
                // SAFETY: process_avx requires AVX target feature which we've detected
                unsafe { self.process_avx(input, output, operation) }
            }
            InstructionSet::SSE42 | InstructionSet::SSE2 => {
                // SAFETY: process_sse2 requires SSE2 target feature which we've detected
                unsafe { self.process_sse2(input, output, operation) }
            }
            InstructionSet::Scalar | InstructionSet::AVX512 => {
                // AVX-512 fallback to scalar for now (complex to implement)
                self.process_scalar(input, output, operation)
            }
        }
    }

    /// AVX2 vectorized processing (8 f32s at once)
    #[target_feature(enable = "avx2")]
    unsafe fn process_avx2(&self, input: &[f32], output: &mut [f32], operation: BufferOperation) -> Result<()> {
        let len = input.len();
        let vector_len = len - (len % 8);

        match operation {
            BufferOperation::Copy => {
                for i in (0..vector_len).step_by(8) {
                    unsafe {
                        let a = _mm256_loadu_ps(input.as_ptr().add(i));
                        _mm256_storeu_ps(output.as_mut_ptr().add(i), a);
                    }
                }
            }
            BufferOperation::Gain(gain) => {
                for i in (0..vector_len).step_by(8) {
                    unsafe {
                        let gain_vec = _mm256_set1_ps(gain);
                        let a = _mm256_loadu_ps(input.as_ptr().add(i));
                        let result = _mm256_mul_ps(a, gain_vec);
                        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
                    }
                }
            }
            BufferOperation::Add => {
                for i in (0..vector_len).step_by(8) {
                    unsafe {
                        let a = _mm256_loadu_ps(input.as_ptr().add(i));
                        let b = _mm256_loadu_ps(output.as_ptr().add(i));
                        let result = _mm256_add_ps(a, b);
                        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
                    }
                }
            }
            BufferOperation::Multiply => {
                for i in (0..vector_len).step_by(8) {
                    unsafe {
                        let a = _mm256_loadu_ps(input.as_ptr().add(i));
                        let b = _mm256_loadu_ps(output.as_ptr().add(i));
                        let result = _mm256_mul_ps(a, b);
                        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
                    }
                }
            }
            BufferOperation::SoftClip(threshold) => {
                for i in (0..vector_len).step_by(8) {
                    unsafe {
                        let threshold_vec = _mm256_set1_ps(threshold);
                        let neg_threshold_vec = _mm256_set1_ps(-threshold);
                        let a = _mm256_loadu_ps(input.as_ptr().add(i));
                        // Soft clipping using tanh approximation
                        let clamped = _mm256_max_ps(_mm256_min_ps(a, threshold_vec), neg_threshold_vec);
                        let result = _mm256_div_ps(clamped, _mm256_set1_ps(1.0 + threshold));
                        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
                    }
                }
            }
            BufferOperation::HighPass { cutoff_norm, state } => {
                unsafe {
                    // Simple one-pole high-pass filter
                    let alpha = _mm256_set1_ps(cutoff_norm);
                    let one_minus_alpha = _mm256_set1_ps(1.0 - cutoff_norm);
                    let mut prev = _mm256_set1_ps(*state);

                    for i in (0..vector_len).step_by(8) {
                        let input_vec = _mm256_loadu_ps(input.as_ptr().add(i));
                        let filtered = _mm256_add_ps(
                            _mm256_mul_ps(alpha, input_vec),
                            _mm256_mul_ps(one_minus_alpha, prev)
                        );
                        let output_vec = _mm256_sub_ps(input_vec, filtered);
                        _mm256_storeu_ps(output.as_mut_ptr().add(i), output_vec);
                        prev = filtered;
                    }

                    // Update state with last value
                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), prev);
                    *state = temp[7];
                }
            }
        }

        // Handle remaining samples with scalar operations
        for i in vector_len..len {
            self.process_sample_scalar(input[i], &mut output[i], operation)?;
        }

        Ok(())
    }

    /// AVX vectorized processing (8 f32s at once)
    #[target_feature(enable = "avx")]
    unsafe fn process_avx(&self, input: &[f32], output: &mut [f32], operation: BufferOperation) -> Result<()> {
        let len = input.len();
        let vector_len = len - (len % 8);

        match operation {
            BufferOperation::Copy => {
                for i in (0..vector_len).step_by(8) {
                    unsafe {
                        let a = _mm256_loadu_ps(input.as_ptr().add(i));
                        _mm256_storeu_ps(output.as_mut_ptr().add(i), a);
                    }
                }
            }
            BufferOperation::Gain(gain) => {
                for i in (0..vector_len).step_by(8) {
                    unsafe {
                        let gain_vec = _mm256_set1_ps(gain);
                        let a = _mm256_loadu_ps(input.as_ptr().add(i));
                        let result = _mm256_mul_ps(a, gain_vec);
                        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
                    }
                }
            }
            _ => {
                // Fall back to scalar for complex operations in AVX mode
                return self.process_scalar(input, output, operation);
            }
        }

        // Handle remaining samples
        for i in vector_len..len {
            self.process_sample_scalar(input[i], &mut output[i], operation)?;
        }

        Ok(())
    }

    /// SSE2 vectorized processing (4 f32s at once)
    #[target_feature(enable = "sse2")]
    unsafe fn process_sse2(&self, input: &[f32], output: &mut [f32], operation: BufferOperation) -> Result<()> {
        let len = input.len();
        let vector_len = len - (len % 4);

        match operation {
            BufferOperation::Copy => {
                for i in (0..vector_len).step_by(4) {
                    unsafe {
                        let a = _mm_loadu_ps(input.as_ptr().add(i));
                        _mm_storeu_ps(output.as_mut_ptr().add(i), a);
                    }
                }
            }
            BufferOperation::Gain(gain) => {
                for i in (0..vector_len).step_by(4) {
                    unsafe {
                        let gain_vec = _mm_set1_ps(gain);
                        let a = _mm_loadu_ps(input.as_ptr().add(i));
                        let result = _mm_mul_ps(a, gain_vec);
                        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
                    }
                }
            }
            BufferOperation::Add => {
                for i in (0..vector_len).step_by(4) {
                    unsafe {
                        let a = _mm_loadu_ps(input.as_ptr().add(i));
                        let b = _mm_loadu_ps(output.as_ptr().add(i));
                        let result = _mm_add_ps(a, b);
                        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
                    }
                }
            }
            _ => {
                // Fall back to scalar for complex operations
                return self.process_scalar(input, output, operation);
            }
        }

        // Handle remaining samples
        for i in vector_len..len {
            self.process_sample_scalar(input[i], &mut output[i], operation)?;
        }

        Ok(())
    }

    /// Scalar fallback processing
    fn process_scalar(&self, input: &[f32], output: &mut [f32], operation: BufferOperation) -> Result<()> {
        for i in 0..input.len() {
            self.process_sample_scalar(input[i], &mut output[i], operation)?;
        }
        Ok(())
    }

    /// Process a single sample (scalar)
    fn process_sample_scalar(&self, input: f32, output: &mut f32, operation: BufferOperation) -> Result<()> {
        match operation {
            BufferOperation::Copy => {
                *output = input;
            }
            BufferOperation::Gain(gain) => {
                *output = input * gain;
            }
            BufferOperation::Add => {
                *output += input;
            }
            BufferOperation::Multiply => {
                *output *= input;
            }
            BufferOperation::SoftClip(threshold) => {
                *output = if input.abs() <= threshold {
                    input
                } else {
                    threshold * input.signum() * (1.0 - (-2.0 * input.abs() / threshold).exp())
                };
            }
            BufferOperation::HighPass { cutoff_norm, state } => {
                // SAFETY: state pointer is valid for the duration of the operation
                unsafe {
                    let filtered = cutoff_norm * input + (1.0 - cutoff_norm) * *state;
                    *output = input - filtered;
                    *state = filtered;
                }
            }
        }
        Ok(())
    }

    /// Vectorized convolution for FIR filters
    pub fn convolve(&self, input: &[f32], impulse: &[f32], output: &mut [f32]) -> Result<()> {
        if output.len() != input.len() + impulse.len() - 1 {
            return Err(anyhow::anyhow!("Output buffer size incorrect for convolution"));
        }

        match self.instruction_set {
            InstructionSet::Avx2Fma | InstructionSet::AVX2 => {
                // SAFETY: convolve_avx2 requires AVX2+FMA target features which we've detected
                unsafe { self.convolve_avx2(input, impulse, output) }
            }
            _ => {
                self.convolve_scalar(input, impulse, output)
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn convolve_avx2(&self, input: &[f32], impulse: &[f32], output: &mut [f32]) -> Result<()> {
        output.fill(0.0);

        for i in 0..input.len() {
            let impulse_len = impulse.len();
            let vector_len = impulse_len - (impulse_len % 8);

            // Vectorized multiply-accumulate
            for j in (0..vector_len).step_by(8) {
                unsafe {
                    let input_val = _mm256_set1_ps(input[i]);
                    let impulse_vec = _mm256_loadu_ps(impulse.as_ptr().add(j));
                    let output_vec = _mm256_loadu_ps(output.as_ptr().add(i + j));
                    let result = _mm256_fmadd_ps(input_val, impulse_vec, output_vec);
                    _mm256_storeu_ps(output.as_mut_ptr().add(i + j), result);
                }
            }

            // Handle remaining impulse samples
            for j in vector_len..impulse_len {
                output[i + j] += input[i] * impulse[j];
            }
        }

        Ok(())
    }

    fn convolve_scalar(&self, input: &[f32], impulse: &[f32], output: &mut [f32]) -> Result<()> {
        output.fill(0.0);

        for i in 0..input.len() {
            for j in 0..impulse.len() {
                output[i + j] += input[i] * impulse[j];
            }
        }

        Ok(())
    }

    /// Get processor capabilities
    pub fn capabilities(&self) -> &SimdCapabilities {
        &self.capabilities
    }

    /// Get current instruction set
    pub fn instruction_set(&self) -> InstructionSet {
        self.instruction_set
    }

    /// Get vector width
    pub fn vector_width(&self) -> usize {
        self.vector_width
    }

    /// Benchmark SIMD performance
    pub fn benchmark(&self) -> Result<SimdBenchmarkResults> {
        use std::time::Instant;

        const TEST_SIZE: usize = 4096;
        const ITERATIONS: usize = 1000;

        let input: Vec<f32> = (0..TEST_SIZE).map(|i| (i as f32) * 0.001).collect();
        let mut output = vec![0.0f32; TEST_SIZE];

        // Benchmark different operations
        let mut results = SimdBenchmarkResults::default();

        // Gain operation benchmark
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            self.process_buffer(&input, &mut output, BufferOperation::Gain(0.5))?;
        }
        results.gain_throughput_msamples_per_sec =
            (TEST_SIZE * ITERATIONS) as f64 / start.elapsed().as_secs_f64() / 1_000_000.0;

        // Copy operation benchmark
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            self.process_buffer(&input, &mut output, BufferOperation::Copy)?;
        }
        results.copy_throughput_msamples_per_sec =
            (TEST_SIZE * ITERATIONS) as f64 / start.elapsed().as_secs_f64() / 1_000_000.0;

        // Add operation benchmark
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            self.process_buffer(&input, &mut output, BufferOperation::Add)?;
        }
        results.add_throughput_msamples_per_sec =
            (TEST_SIZE * ITERATIONS) as f64 / start.elapsed().as_secs_f64() / 1_000_000.0;

        results.instruction_set = self.instruction_set;
        results.vector_width = self.vector_width;

        Ok(results)
    }
}

/// Supported buffer operations
#[derive(Debug, Clone, Copy)]
pub enum BufferOperation {
    Copy,
    Gain(f32),
    Add,
    Multiply,
    SoftClip(f32),
    HighPass { cutoff_norm: f32, state: *mut f32 },
}

unsafe impl Send for BufferOperation {}
unsafe impl Sync for BufferOperation {}

/// SIMD benchmark results
#[derive(Debug, Clone, Default)]
pub struct SimdBenchmarkResults {
    pub instruction_set: InstructionSet,
    pub vector_width: usize,
    pub gain_throughput_msamples_per_sec: f64,
    pub copy_throughput_msamples_per_sec: f64,
    pub add_throughput_msamples_per_sec: f64,
}

impl SimdBenchmarkResults {
    pub fn report(&self) {
        info!("🏁 SIMD Performance Benchmark Results:");
        info!("  Instruction Set: {}", self.instruction_set);
        info!("  Vector Width: {} samples", self.vector_width);
        info!("  Gain Throughput: {:.1} MSamples/sec", self.gain_throughput_msamples_per_sec);
        info!("  Copy Throughput: {:.1} MSamples/sec", self.copy_throughput_msamples_per_sec);
        info!("  Add Throughput: {:.1} MSamples/sec", self.add_throughput_msamples_per_sec);

        if self.gain_throughput_msamples_per_sec > 100.0 {
            info!("  ✅ Excellent SIMD performance");
        } else if self.gain_throughput_msamples_per_sec > 50.0 {
            info!("  ✅ Good SIMD performance");
        } else {
            warn!("  ⚠️ Poor SIMD performance, check CPU capabilities");
        }
    }
}

/// Global SIMD processor instance using OnceLock for Rust 2024 safety
static GLOBAL_SIMD_PROCESSOR: std::sync::OnceLock<SimdProcessor> = std::sync::OnceLock::new();

/// Initialize global SIMD processor
pub fn init_global_simd() -> &'static SimdProcessor {
    GLOBAL_SIMD_PROCESSOR.get_or_init(SimdProcessor::new)
}

/// Get global SIMD processor
pub fn global_simd() -> Option<&'static SimdProcessor> {
    GLOBAL_SIMD_PROCESSOR.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let caps = SimdCapabilities::detect();
        println!("SIMD capabilities: {:?}", caps);

        let best = caps.best_instruction_set();
        println!("Best instruction set: {:?}", best);

        assert!(!matches!(best, InstructionSet::Scalar) || !caps.has_sse2);
    }

    #[test]
    fn test_simd_processor() {
        let processor = SimdProcessor::new();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 8];

        let result = processor.process_buffer(&input, &mut output, BufferOperation::Gain(2.0));
        assert!(result.is_ok());

        // Check that gain was applied
        for (i, &val) in output.iter().enumerate() {
            assert!((val - input[i] * 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_simd_benchmark() {
        let processor = SimdProcessor::new();
        let results = processor.benchmark().unwrap();

        results.report();

        assert!(results.gain_throughput_msamples_per_sec > 0.0);
        assert!(results.copy_throughput_msamples_per_sec > 0.0);
    }

    #[test]
    fn test_convolution() {
        let processor = SimdProcessor::new();
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let impulse = vec![0.5, 0.25, 0.125];
        let mut output = vec![0.0; input.len() + impulse.len() - 1];

        let result = processor.convolve(&input, &impulse, &mut output);
        assert!(result.is_ok());

        // Check impulse response
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - 0.25).abs() < 1e-6);
        assert!((output[2] - 0.125).abs() < 1e-6);
    }
}