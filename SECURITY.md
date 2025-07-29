# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please:

1. **Do not** open a public issue
2. Email security concerns to: [maintainer-email]
3. Include detailed information about the vulnerability
4. Allow reasonable time for response before public disclosure

## Security Considerations

### Input Validation
- All optimization functions validate input parameters
- Gradient computations include numerical stability checks
- Memory usage is bounded to prevent DoS attacks

### Dependencies
- Regular dependency updates via Dependabot
- Security scanning in CI/CD pipeline
- Minimal dependency footprint

### Container Security
- Non-root user in Docker containers
- Minimal base images
- Regular base image updates

## Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Varies by severity

Thank you for helping keep our project secure!
