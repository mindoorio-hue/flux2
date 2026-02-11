# Contributing to Flux2 Endpoint

Thank you for considering contributing to this project!

## How to Contribute

### Reporting Bugs
- Use the GitHub Issues tab
- Use the bug report template
- Include detailed reproduction steps
- Provide error logs and environment details

### Suggesting Features
- Use the GitHub Issues tab
- Use the feature request template
- Describe the use case clearly
- Consider performance and compatibility implications

### Code Contributions

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/flux2-endpoint.git
   cd flux2-endpoint
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation if needed

4. **Test your changes**
   ```bash
   # Run quick tests
   python quick_test.py

   # Run pytest
   pytest tests/ -v

   # Test with actual model (if possible)
   python local_test.py
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   Use conventional commits:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for tests
   - `refactor:` for code refactoring

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and modular

## Testing Guidelines

- Add unit tests for new features
- Ensure all existing tests still pass
- Test with both FLUX.1-dev and FLUX.1-schnell
- Test all three workflows if applicable

## Documentation

When adding features, update:
- `README.md` - User-facing documentation
- `CLAUDE.md` - Development guide
- `WORKFLOWS.md` - Workflow-specific details
- Docstrings in code

## Performance Considerations

- Consider VRAM usage for GPU features
- Test with different image sizes
- Measure inference time impact
- Document performance characteristics

## Questions?

Feel free to open an issue for any questions about contributing!
