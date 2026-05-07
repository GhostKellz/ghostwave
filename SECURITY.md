# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Email security concerns to: ckelley@ghostkellz.sh
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 60 days

### Scope

Security issues we care about:
- Memory safety issues in audio processing
- Buffer overflows or underflows
- Privilege escalation via IPC
- Path traversal in configuration loading
- Denial of service in real-time audio path
- Information disclosure
- CUDA/GPU memory safety issues

### Out of Scope

- Issues requiring physical access
- Social engineering attacks
- Vulnerabilities in dependencies (report to upstream)
- Issues in unsupported versions

## Security Considerations

### Real-Time Audio Safety

GhostWave processes audio in real-time with low latency requirements. The codebase:
- Avoids allocations in audio callbacks
- Uses lock-free queues for thread communication
- Validates buffer sizes before processing

### IPC Security

The JSON-RPC IPC server:
- Binds to Unix domain sockets with user permissions
- Does not accept network connections by default
- Validates all incoming commands

### GPU Memory

When using CUDA acceleration:
- GPU memory is allocated with bounds checking
- Failed allocations fall back to CPU processing
- No sensitive data is stored in GPU memory

### File Access

- Configuration files are read from user-owned directories
- Model files are verified before loading
- No arbitrary file operations are performed

## Acknowledgments

We appreciate security researchers who help keep GhostWave secure. Contributors will be acknowledged (with permission) in release notes.
