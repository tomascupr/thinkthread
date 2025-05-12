"""Integration tests for template customization and usage."""
import os
import pytest
from tempfile import TemporaryDirectory

from cort_sdk.cort_session import CoRTSession
from cort_sdk.llm.dummy import DummyLLMClient
from cort_sdk.prompting import TemplateManager


def test_template_customization():
    """Test that customizing templates changes the prompts used."""
    with TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, "initial_prompt.j2"), "w") as f:
            f.write("CUSTOM TEMPLATE: Please answer: {{ question }}")

        with open(os.path.join(temp_dir, "alternative_prompt.j2"), "w") as f:
            f.write(
                "CUSTOM ALTERNATIVE: {{ question }} (Current: {{ current_answer }})"
            )

        with open(os.path.join(temp_dir, "evaluation_prompt.j2"), "w") as f:
            f.write("CUSTOM EVALUATION: {{ question }}\n{{ formatted_answers }}\nBest?")

        with open(os.path.join(temp_dir, "pairwise_prompt.j2"), "w") as f:
            f.write(
                "CUSTOM PAIRWISE: {{ question }}\nPrev: {{ prev_answer }}\nNew: {{ new_answer }}"
            )

        prompts_received = []

        class RecordingClient(DummyLLMClient):
            def generate(self, prompt, **kwargs):
                prompts_received.append(prompt)
                return super().generate(prompt, **kwargs)

        client = RecordingClient(responses=["Answer"] * 10)

        template_manager = TemplateManager(template_dir=temp_dir)

        from cort_sdk.config import CoRTConfig

        config = CoRTConfig()
        config.use_pairwise_evaluation = False

        session = CoRTSession(
            llm_client=client,
            template_manager=template_manager,
            rounds=1,
            alternatives=1,
            config=config,
        )

        session.run("Test question")

        assert "CUSTOM TEMPLATE" in prompts_received[0]
        assert "CUSTOM ALTERNATIVE" in prompts_received[1]
        assert "CUSTOM EVALUATION" in prompts_received[2]
