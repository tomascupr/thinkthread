import os
import pytest
from tempfile import TemporaryDirectory

from thinkthread_sdk.prompting import TemplateManager


def test_template_manager_default_dir():
    """Test that TemplateManager loads templates from the default directory."""
    manager = TemplateManager()

    assert os.path.basename(manager.template_dir) == "prompts"
    assert os.path.exists(manager.template_dir)


def test_template_manager_custom_dir():
    """Test that TemplateManager loads templates from a custom directory."""
    with TemporaryDirectory() as temp_dir:
        manager = TemplateManager(template_dir=temp_dir)
        assert manager.template_dir == temp_dir


def test_render_template():
    """Test that TemplateManager renders a template with context."""
    with TemporaryDirectory() as temp_dir:
        template_path = os.path.join(temp_dir, "test_template.j2")
        with open(template_path, "w") as f:
            f.write("Hello, {{ name }}!")

        manager = TemplateManager(template_dir=temp_dir)
        result = manager.render_template("test_template.j2", {"name": "World"})

        assert result == "Hello, World!"


def test_render_template_not_found():
    """Test that TemplateManager raises an error when a template is not found."""
    manager = TemplateManager()

    with pytest.raises(Exception):
        manager.render_template("non_existent_template.j2", {})


def test_render_template_with_cort_context():
    """Test that TemplateManager renders CoRT-specific templates correctly."""
    with TemporaryDirectory() as temp_dir:
        initial_path = os.path.join(temp_dir, "initial_prompt.j2")
        with open(initial_path, "w") as f:
            f.write("Question: {{ question }}\n\nAnswer:")

        alternative_path = os.path.join(temp_dir, "alternative_prompt.j2")
        with open(alternative_path, "w") as f:
            f.write(
                "Question: {{ question }}\n\nCurrent: {{ current_answer }}\n\nAlternative:"
            )

        evaluation_path = os.path.join(temp_dir, "evaluation_prompt.j2")
        with open(evaluation_path, "w") as f:
            f.write(
                "Question: {{ question }}\n\n{{ formatted_answers }}\n\nWhich is best (1-{{ num_answers }})?"
            )

        manager = TemplateManager(template_dir=temp_dir)

        initial = manager.render_template(
            "initial_prompt.j2", {"question": "What is CoRT?"}
        )
        assert "Question: What is CoRT?" in initial

        alternative = manager.render_template(
            "alternative_prompt.j2",
            {"question": "What is CoRT?", "current_answer": "Chain-of-Thought"},
        )
        assert "Question: What is CoRT?" in alternative
        assert "Current: Chain-of-Thought" in alternative

        evaluation = manager.render_template(
            "evaluation_prompt.j2",
            {
                "question": "What is CoRT?",
                "formatted_answers": "Answer 1: Chain-of-Thought\nAnswer 2: Chain-of-Reasoning",
                "num_answers": 2,
            },
        )
        assert "Question: What is CoRT?" in evaluation
        assert "Answer 1: Chain-of-Thought" in evaluation
        assert "Which is best (1-2)?" in evaluation
