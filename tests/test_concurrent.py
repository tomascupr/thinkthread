"""Tests for concurrent usage of the ThinkThread SDK."""

import pytest
import asyncio

from thinkthread_sdk.llm.dummy import DummyLLMClient
from thinkthread_sdk.cort_session import CoRTSession
from thinkthread_sdk.prompting import TemplateManager
from thinkthread_sdk.config import CoRTConfig


@pytest.mark.asyncio
async def test_concurrent_sessions():
    """Test that multiple CoRT sessions can run concurrently without issues."""
    template_manager = TemplateManager()

    clients = [DummyLLMClient(responses=[f"Answer from client {i}"]) for i in range(5)]

    config = CoRTConfig(use_pairwise_evaluation=False)

    sessions = [
        CoRTSession(
            llm_client=clients[i],
            template_manager=template_manager,  # Shared template manager
            max_rounds=1,
            config=config,
        )
        for i in range(5)
    ]

    questions = [f"Test question {i}" for i in range(5)]

    tasks = [
        session.run_async(question) for session, question in zip(sessions, questions)
    ]

    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        assert f"Answer from client {i}" in result


@pytest.mark.asyncio
async def test_template_manager_thread_safety():
    """Test that TemplateManager is thread-safe when used concurrently."""
    template_manager = TemplateManager()

    async def render_task(i: int):
        await asyncio.sleep(0.01 * (i % 5))
        result = template_manager.render_template(
            "initial_prompt.j2", {"question": f"Question {i}"}
        )
        return result

    tasks = [render_task(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        assert f"Question {i}" in result
