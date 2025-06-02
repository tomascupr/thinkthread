# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install dependencies
poetry install

# Set up API keys (choose one)
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='...'
```

### Running Tests
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_dummy_llm.py -xvs

# Run with coverage
poetry run pytest --cov=thinkthread

# Run tests matching pattern
poetry run pytest -k "test_reason"
```

### Code Quality
```bash
# Format code
poetry run black .

# Lint code
poetry run ruff .
```

### CLI Commands
```bash
# Run in test mode (no API calls)
poetry run think --test "Your question"

# Run with specific provider
poetry run think --provider anthropic "Question"

# Debug mode
poetry run think --debug "Question"
```

## Architecture

ThinkThread uses a **hybrid architecture** that combines a new simplified API with a robust old SDK implementation:

### Layer Structure
1. **New API** (`thinkthread/`): Simple functions like `reason()`, `explore()`, `solve()`
2. **Adapter** (`thinkthread/core/adapter.py`): Bridges new API to old SDK
3. **Old SDK** (`thinkthread/core/`): Battle-tested Chain-of-Recursive-Thoughts implementation
4. **LLM Clients** (`thinkthread/llm/`): Provider implementations with retry logic and caching

### Key Integration Points

**Test Mode Switching**: When `test_mode=True` is passed, the adapter reinitializes the LLM client to use `DummyLLMClient`. This was a critical fix to prevent hanging in test mode.

**Mode Auto-Detection**: The adapter automatically detects which reasoning mode to use based on keywords in the prompt:
- "explore", "brainstorm" → TreeOfThoughts
- "debate", "pros and cons" → Multi-perspective debate
- "solve", "fix", "problem" → Solution-focused reasoning
- Default → Refinement mode

**Result Conversion**: Old SDK results are converted to the new structured format with fields like `answer`, `confidence`, `cost`, `mode`.

### Important Files

- `thinkthread/reason.py`: Main API entry point, handles test_mode switching
- `thinkthread/core/adapter.py`: Critical bridge between new/old, contains mode routing
- `thinkthread/core/session.py`: Chain-of-Recursive-Thoughts implementation
- `thinkthread/llm/base.py`: Base LLM client with caching and retry logic
- `thinkthread/cli.py`: New simplified CLI implementation

## Common Tasks

### Adding a New Reasoning Mode
1. Add method to `SDKAdapter` in `core/adapter.py`
2. Expose in `reason.py` as a new function
3. Add CLI command in `cli.py` if needed
4. Update mode detection in `_detect_mode()`

### Debugging Integration Issues
1. Check if test_mode is properly switching the LLM client
2. Verify mode detection is working correctly
3. Ensure result conversion maintains all metadata
4. Check that the adapter is creating sessions with correct config

### Testing Changes
Always test both modes:
- With API: `poetry run think "question"`
- Without API: `poetry run think --test "question"`

## Recent Changes

The codebase recently underwent a major integration where:
1. New simplified API was created for better developer experience
2. Old SDK was preserved for its production robustness
3. Adapter pattern was implemented to bridge both systems
4. Test mode hanging issue was fixed by properly reinitializing LLM clients
5. All debug print statements were cleaned up