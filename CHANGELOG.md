# Changelog

All notable changes to the ThinkThread SDK will be documented in this file.

## [Unreleased]

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
