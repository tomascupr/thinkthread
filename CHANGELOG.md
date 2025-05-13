# Changelog

All notable changes to the ThinkThread SDK will be documented in this file.

## [0.5.2] - 2025-05-13

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
- Added comprehensive formatting and testing report

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
