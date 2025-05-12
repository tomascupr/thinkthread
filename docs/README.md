# CoRT SDK Documentation

This directory contains comprehensive documentation for the CoRT SDK.

## Documentation Structure

- `usage_guide.md`: Detailed guide on using the SDK for different scenarios
- `developer_guide.md`: Architecture explanation and extension guide
- `configuration_reference.md`: Complete reference of all configuration options
- `cli_reference.md`: Detailed documentation of CLI commands and options

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
