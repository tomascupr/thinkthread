import os
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from cort_sdk.prompting import TemplateManager
from cort_sdk.config import create_config


@pytest.fixture
def template_names():
    """Return a list of template names that should exist."""
    return [
        "initial_prompt.j2",
        "alternative_prompt.j2",
        "evaluation_prompt.j2",
        "pairwise_prompt.j2",
        "final_answer.j2",
    ]


def test_default_templates_exist(template_names):
    """Test that all required templates exist in the default template directory."""
    manager = TemplateManager()
    template_dir = manager.template_dir

    for template_name in template_names:
        template_path = os.path.join(template_dir, template_name)
        assert os.path.exists(template_path), f"Template {template_name} not found"


@pytest.mark.parametrize(
    "template_name,context,expected_substring",
    [
        ("initial_prompt.j2", {"question": "Test question"}, "Test question"),
        (
            "alternative_prompt.j2",
            {"question": "Test question", "current_answer": "Current answer"},
            "Current answer",
        ),
        (
            "evaluation_prompt.j2",
            {
                "question": "Test question",
                "formatted_answers": "Answer 1: Test\nAnswer 2: Test",
                "num_answers": 2,
            },
            "Answer 1: Test",
        ),
        (
            "pairwise_prompt.j2",
            {
                "question": "Test question",
                "prev_answer": "Previous",
                "new_answer": "New",
            },
            "Previous",
        ),
    ],
)
def test_template_rendering(template_name, context, expected_substring):
    """Test that templates render correctly with the provided context."""
    manager = TemplateManager()
    rendered = manager.render_template(template_name, context)

    assert (
        expected_substring in rendered
    ), f"Expected '{expected_substring}' in rendered template"


def test_custom_template_directory():
    """Test that custom templates can be loaded from a specified directory."""
    with TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, "custom_template.j2"), "w") as f:
            f.write("Custom template with {{ variable }}")

        manager = TemplateManager(template_dir=temp_dir)
        rendered = manager.render_template(
            "custom_template.j2", {"variable": "test value"}
        )

        assert "Custom template with test value" == rendered


def test_all_template_contexts():
    """Test that all templates can be rendered with their expected contexts."""
    manager = TemplateManager()

    initial = manager.render_template(
        "initial_prompt.j2", {"question": "Test question"}
    )
    assert "Test question" in initial

    alternative = manager.render_template(
        "alternative_prompt.j2",
        {"question": "Test question", "current_answer": "Current answer"},
    )
    assert "Test question" in alternative
    assert "Current answer" in alternative

    evaluation = manager.render_template(
        "evaluation_prompt.j2",
        {
            "question": "Test question",
            "formatted_answers": "Answer 1: First\nAnswer 2: Second",
            "num_answers": 2,
        },
    )
    assert "Test question" in evaluation
    assert "Answer 1: First" in evaluation

    pairwise = manager.render_template(
        "pairwise_prompt.j2",
        {
            "question": "Test question",
            "prev_answer": "Previous answer",
            "new_answer": "New answer",
        },
    )
    assert "Test question" in pairwise
    assert "Previous answer" in pairwise
    assert "New answer" in pairwise
