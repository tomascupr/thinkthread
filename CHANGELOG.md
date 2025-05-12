# Changelog

All notable changes to the CoRT SDK will be documented in this file.

## [Unreleased]

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
