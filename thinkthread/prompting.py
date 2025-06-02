"""Template management for ThinkThread prompts.

This module provides utilities for loading and rendering Jinja2 templates
used for prompting language models in the ThinkThread process.
"""

import os
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape


class TemplateManager:
    """Manages Jinja2 templates for ThinkThread prompts.

    This class is responsible for loading templates from a directory and
    rendering them with the provided context data.
    """

    DEFAULT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "prompts")

    def __init__(self, template_dir: Optional[str] = None) -> None:
        """Initialize the TemplateManager with a template directory.

        Args:
            template_dir: Path to the directory containing template files.
                        If None, uses the default template directory.

        """
        self.template_dir = template_dir or self.DEFAULT_TEMPLATE_DIR

        self.env = Environment(
            loader=FileSystemLoader(self.template_dir), autoescape=select_autoescape()
        )

    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a template with the given context.

        Args:
            template_name: Name of the template file (e.g., "initial_prompt.j2")
            context: Dictionary of variables to pass to the template

        Returns:
            The rendered template string

        Raises:
            FileNotFoundError: If the template file does not exist
            jinja2.exceptions.TemplateError: If there is an error rendering the template

        """
        template = self.env.get_template(template_name)
        return template.render(**context)
