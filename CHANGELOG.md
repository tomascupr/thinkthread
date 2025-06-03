# Changelog

All notable changes to ThinkThread will be documented in this file.

## [0.8.1] - 2025-01-06

### Added
- **Comprehensive SDK Documentation**: Three new documentation files for advanced users:
  - `SDK_GUIDE.md`: Complete guide to SDK architecture, configuration, and advanced features
  - `API_REFERENCE.md`: Detailed parameter documentation for all functions and classes
  - `EXAMPLES.md`: Practical usage patterns and production-ready code examples
- Documentation for all 30+ configuration parameters that were previously undocumented
- Examples for custom evaluators, scoring functions, and production patterns (caching, monitoring, rate limiting)

### Changed
- **CLI Simplification**: Changed from `think` to `thinkthread` as the main command
  - Commands now use: `thinkthread default/explore/solve/debate/refine`
  - Removed the ambiguous `think` command to avoid CLI parsing issues
- Updated README to link to new SDK documentation

## [0.8.0] - 2025-06-03

### Added
- **Simple API**: Just 5 functions - `reason()`, `explore()`, `solve()`, `debate()`, `refine()`
- **Test Mode**: Develop without API calls using `test_mode=True`
- **Clean CLI**: Simple commands like `think "question"` and subcommands like `think solve "problem"`
- **Zero Config**: Works out of the box with just an API key
- **Production Ready**: All functions tested with real API calls

### Changed
- **Package Rename**: From `thinkthread_sdk` to `thinkthread` for cleaner imports
- **API First**: Beautiful simple functions that wrap the proven SDK
- **Simplified CLI**: From `thinkthread run` to just `think` with intuitive subcommands
- **Minimal Repo**: Removed non-essential files, keeping only core package and documentation

### Fixed
- **TreeThinker Performance**: Optimized `explore()` function with reduced defaults for faster response times
- **CLI Function Conflicts**: Resolved import naming conflicts in CLI commands
- **Code Quality**: Fixed all linting issues and applied consistent formatting
- **Package Structure**: Ensured all template files (.j2) are properly included in distribution

### Technical Details
- Minimal wrapper approach - only ~150 lines of new API code
- Preserved all SDK robustness (retries, caching, evaluation, monitoring)
- No breaking changes for advanced users - can still access `ThinkThreadSession` directly
- Comprehensive testing: all 5 API functions verified with real OpenAI API calls
- Optimized TreeThinker defaults: beam_width=1, max_iterations=1, depth=2 for speed
- Clean package structure with only essential files

## [0.7.1] - 2025-05-31

### Fixed
- Fixed CLI structure to properly register `tot` and `think` commands
- Resolved "No such command" errors for unified CLI commands
- Fixed command registration conflicts between imported modules

### Added
- Added high-level API abstractions through new `ThinkThreadUtils` class with simplified one-liner methods:
  - `self_refine`: Refine an answer through multiple rounds of critique and revision
  - `self_refine_async`: Async version of self_refine
  - `n_best_brainstorm`: Generate multiple answers and select the best one
  - `n_best_brainstorm_async`: Async version of n_best_brainstorm
- Added CLI commands to expose the new functionality:
  - `thinkthread refine`: Refine an initial answer
  - `thinkthread brainstorm`: Generate multiple answers and select the best
- Updated README with examples of the new utilities, including metadata return options

## [0.7.0] - 2025-05-13

### Added
- Introduced TreeThinker module for Tree-of-Thoughts reasoning
- Added comprehensive documentation for Tree-of-Thoughts in docs/tree_thinker.md
- Created practical examples demonstrating TreeThinker usage
- Implemented CLI command `thinkthread tot` for tree-based reasoning
- Added beam search pruning for efficient exploration of reasoning paths
- Implemented sophisticated scoring algorithm for evaluating thought branches
- Added asynchronous API for non-blocking tree-based reasoning
- Created visualization mechanism for the tree of thoughts

## [0.6.0] - 2025-05-13

### Fixed
- Fixed missing `_generate_uncached` abstract method implementation in all LLM clients
- Fixed type errors in cache handling to ensure consistent return types
- Fixed unused variables in test files

### Changed
- Improved README documentation with better organization and clarity
- Simplified mermaid diagram for better readability
- Enhanced docs README with more context about the SDK

### Added
- Created test scripts for OpenAI and Anthropic API integration

## [0.5.0] - 2025-05-12

### Added
- Comprehensive documentation including README, usage guide, developer guide, configuration reference, and CLI reference
- Prompt templating using Jinja2 for all prompts
- Recursive reasoning with configurable rounds and alternatives
- Self-evaluation capability for answer quality assessment
- Pairwise evaluation for comparing answers head-to-head
- Asynchronous API support for non-blocking operation
- Streaming response capability for real-time output
- Multiple LLM provider support (OpenAI, Anthropic, HuggingFace)
- Extensible architecture for adding new providers and evaluation strategies
- Configuration system using Pydantic with environment variable support
- Command-line interface with comprehensive options

### Changed
- Improved error handling and logging throughout the codebase
- Enhanced docstrings for better code documentation
