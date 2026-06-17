# GhostWave Documentation

## Backends
Audio system integration guides.

- [PipeWire](backends/pipewire.md) — PipeWire filter node setup and configuration
- [ALSA](backends/alsa.md) — Direct ALSA hardware access

## GPU
NVIDIA GPU acceleration and optimization.

- [NVIDIA](gpu/nvidia.md) — CUDA setup and GPU acceleration
- [RTX 5090](gpu/rtx-5090.md) — Blackwell architecture optimizations

## Integration
Embedding and connecting with other applications.

- [Crate Usage](integration/crate-usage.md) — Using ghostwave-core as a Rust dependency
- [PhantomLink](integration/phantomlink.md) — Professional mixing integration
- [NVControl](integration/nvcontrol.md) — NVIDIA GPU management integration

## Development
Building, testing, and contributing.

- [Architecture](development/architecture.md) — System design overview
- [Building](development/building.md) — Build instructions and development setup
- [API Reference](development/api-reference.md) — Library API documentation
- [Deployment](development/deployment.md) — Packaging and distribution
- [Performance](development/performance.md) — Benchmarking and tuning

## Security
Dependency advisory tracking, kept in sync with `deny.toml`.

- [Accepted Advisories](advisories/accepted.md) — Knowingly accepted advisories (`atty` via the VST framework)
- [Resolved Advisories](advisories/resolved.md) — Advisories cleared by dependency updates

## Reference
- [Known Gaps](known-gaps.md) — Current limitations and planned work
- [Troubleshooting](troubleshooting.md) — Common issues and solutions
