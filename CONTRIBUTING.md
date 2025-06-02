# Contributing to ThinkThread

Thank you for your interest in contributing to ThinkThread! We welcome contributions from the community.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/thinkthread.git
   cd thinkthread
   ```
3. Install dependencies:
   ```bash
   poetry install
   ```

## Development Setup

1. Set up your environment variables:
   ```bash
   export OPENAI_API_KEY=your_key_here  # or ANTHROPIC_API_KEY
   ```

2. Run tests:
   ```bash
   poetry run pytest
   ```

3. Run the CLI in test mode:
   ```bash
   poetry run think --test "Your question here"
   ```

## Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test them:
   ```bash
   poetry run pytest tests/
   poetry run think --test "Test your changes"
   ```

3. Format your code:
   ```bash
   poetry run black .
   poetry run ruff .
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

## Pull Request Guidelines

1. **Keep it focused**: One feature or fix per PR
2. **Write tests**: Add tests for new functionality
3. **Update docs**: Update README.md if needed
4. **Follow conventions**: Use conventional commits (feat:, fix:, docs:, etc.)

## Code Style

- Use Black for formatting
- Follow PEP 8
- Keep functions small and focused
- Add type hints where beneficial

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Use `--test` mode for CLI testing without API calls

## Reporting Issues

- Use GitHub Issues
- Include minimal reproduction code
- Specify your Python version and OS
- Include full error messages

## Questions?

Feel free to open an issue for any questions about contributing.

Thank you for helping make ThinkThread better!