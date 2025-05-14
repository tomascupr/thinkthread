# ThinkThread SDK Documentation

This directory contains comprehensive documentation for the ThinkThread SDK, a powerful framework for improving LLM responses through advanced reasoning techniques. The SDK enables developers to integrate sophisticated reasoning capabilities into their LLM-based applications.

## Reasoning Approaches

ThinkThread SDK supports two complementary reasoning approaches:

1. **Chain-of-Recursive-Thoughts (CoRT)**: A linear refinement process that iteratively improves answers
2. **Tree-of-Thoughts (ToT)**: A tree-based search approach that explores multiple reasoning paths in parallel

## Documentation Structure

- `usage_guide.md`: Detailed guide on using the SDK for different scenarios
- `developer_guide.md`: Architecture explanation and extension guide
- `configuration_reference.md`: Complete reference of all configuration options
- `cli_reference.md`: Detailed documentation of CLI commands and options
- `tree_thinker.md`: Guide to using the Tree-of-Thoughts reasoning approach
- `comparison.md`: Comparison of different reasoning approaches and when to use each

## Building Documentation

We use MkDocs for building the documentation site. To set up and build locally:

```bash
# Install MkDocs and required plugins
pip install mkdocs mkdocs-material

# Build the documentation
mkdocs build

# Serve the documentation locally
mkdocs serve
```

Then visit `http://localhost:8000` to view the documentation.

The generated site will be in the `site/` directory (which is gitignored).
