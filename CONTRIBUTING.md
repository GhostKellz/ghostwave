# Contributing to GhostWave

Thank you for your interest in contributing to GhostWave! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

### Prerequisites

- Rust 1.75 or later
- PipeWire development libraries
- ALSA development libraries
- CUDA Toolkit 12.0+ (optional, for RTX acceleration)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/ghostkellz/ghostwave.git
cd ghostwave

# Install dependencies (Arch Linux)
sudo pacman -S pipewire pipewire-pulse pipewire-alsa pipewire-jack alsa-lib

# Build
cargo build

# Run tests
cargo test --all-features

# Run with debug logging
RUST_LOG=debug cargo run -- --doctor
```

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - GhostWave version (`ghostwave --version`)
   - Operating system and version
   - Audio hardware details
   - Steps to reproduce
   - Expected vs actual behavior
   - Relevant logs (`RUST_LOG=debug ghostwave`)

### Suggesting Features

1. Check existing feature requests
2. Open an issue with the feature request template
3. Describe the use case and expected behavior

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`cargo test --all-features`)
6. Run clippy (`cargo clippy --all-features`)
7. Format code (`cargo fmt`)
8. Commit with clear messages
9. Push and create a Pull Request

### Commit Messages

Follow conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Build/tooling changes

Example: `feat(denoise): add voice isolation mode`

## Code Guidelines

### Rust Style

- Follow the Rust API guidelines
- Use `cargo fmt` for formatting
- Address all `cargo clippy` warnings
- Document public APIs with doc comments
- Include examples in documentation

### Audio Processing

- Maintain real-time safety in audio callbacks
- Avoid allocations in the audio thread
- Use lock-free data structures where needed
- Test latency impact of changes

### GPU Acceleration

- Ensure CPU fallback for all GPU features
- Handle CUDA initialization failures gracefully
- Test on systems without NVIDIA GPUs

## Testing

### Unit Tests

```bash
cargo test --package ghostwave-core
```

### Integration Tests

```bash
cargo test --test '*'
```

### Benchmarks

```bash
cargo run -- --bench
```

## Project Structure

```
ghostwave/
├── src/              # CLI application
├── ghostwave-core/   # Core audio processing library
│   ├── src/
│   │   ├── lib.rs           # Public API
│   │   ├── processor.rs     # Main processor
│   │   ├── ai_denoise/      # AI noise suppression
│   │   ├── dsp/             # DSP components
│   │   └── backends/        # Audio backends
│   └── tests/               # Integration tests
└── ghostwave-vst/    # VST3/CLAP plugin
```

## Release Process

1. Update version in all Cargo.toml files
2. Update CHANGELOG.md
3. Create git tag (`git tag v0.3.0`)
4. Push tag to trigger CI release

## Getting Help

- Open an issue for questions
- Check the documentation in `/docs`
- Review existing issues and PRs

## License

By contributing, you agree that your contributions will be licensed under the MIT OR Apache-2.0 license.
