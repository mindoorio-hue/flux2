# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please:

1. **DO NOT** open a public issue
2. Email the maintainers privately (or use GitHub Security Advisories)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Security Considerations

### API Keys and Tokens
- Never commit `.env` files with real credentials
- Use environment variables for sensitive data
- Rotate API keys regularly

### Model Access
- FLUX.1-dev requires HuggingFace token (gated model)
- Ensure tokens have appropriate permissions
- Don't share tokens in logs or error messages

### Input Validation
- All user inputs are validated before processing
- Base64 image decoding includes error handling
- Dimension and parameter limits are enforced

### Docker Security
- Base images are from official sources
- Dependencies are pinned in requirements.txt
- No unnecessary packages in production image

### RunPod Deployment
- Use RunPod's authentication mechanisms
- Set appropriate endpoint permissions
- Monitor for unusual usage patterns

## Best Practices

When deploying:
1. Use HTTPS for all API calls
2. Implement rate limiting
3. Monitor resource usage
4. Keep dependencies updated
5. Use least-privilege access
6. Enable logging for security events

## Known Limitations

- Base64 decoding may be vulnerable to resource exhaustion with extremely large images
- No built-in authentication (relies on RunPod's system)
- VRAM usage not strictly limited (can cause OOM)

## Updates

Security updates will be released as patches. Subscribe to repository notifications for alerts.
