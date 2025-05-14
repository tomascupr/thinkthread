"""ThinkThread reasoning session implementation.

This module contains the ThinkThreadSession class that orchestrates the multi-round
questioning and refinement process using LLMs.
"""

from typing import List, Optional
import asyncio

from thinkthread_sdk.llm import LLMClient
from thinkthread_sdk.prompting import TemplateManager
from thinkthread_sdk.config import ThinkThreadConfig, create_config
from thinkthread_sdk.evaluation import (
    EvaluationStrategy,
    DefaultEvaluationStrategy,
    Evaluator,
    ModelEvaluator,
)
from thinkthread_sdk.monitoring import GLOBAL_MONITOR, timed
from thinkthread_sdk.base_reasoner import BaseReasoner
from thinkthread_sdk.reasoning_utils import calculate_similarity


class ThinkThreadSession(BaseReasoner):
    """ThinkThread session.

    This class orchestrates a multi-round questioning and refinement process
    using an LLM to generate increasingly better answers to a question.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        alternatives: int = 3,
        rounds: int = 2,
        max_rounds: Optional[int] = None,
        template_manager: Optional[TemplateManager] = None,
        evaluation_strategy: Optional[EvaluationStrategy] = None,
        evaluator: Optional[Evaluator] = None,
        config: Optional[ThinkThreadConfig] = None,
    ) -> None:
        """Initialize a ThinkThread session.

        Args:
            llm_client: LLM client to use for generating and evaluating answers
            alternatives: Number of alternative answers to generate per round
            rounds: Number of refinement rounds to perform (for backward compatibility)
            max_rounds: Maximum number of refinement rounds (overrides rounds if set)
            template_manager: Optional template manager for prompt templates
            evaluation_strategy: Optional strategy for evaluating answers
            evaluator: Optional evaluator for pairwise comparison of answers
            config: Optional configuration object

        """
        super().__init__(llm_client, template_manager, config)
        self.alternatives = alternatives
        self.rounds = rounds
        self.max_rounds = max_rounds if max_rounds is not None else self.rounds
        self.evaluation_strategy = evaluation_strategy or DefaultEvaluationStrategy()
        self.evaluator = evaluator or ModelEvaluator()
        self.use_pairwise_evaluation = self.config.use_pairwise_evaluation
        self.use_self_evaluation = self.config.use_self_evaluation
        self._round_similarities = []

    @timed("run")
    def run(self, question: str) -> str:
        """Execute the ThinkThread process on a question.

        The ThinkThread algorithm improves answer quality through
        multiple rounds of refinement. The process involves:

        1. Generating an initial answer with moderate temperature (0.7) for creativity
        2. For each refinement round:
           a. Generating alternative answers with higher temperature (0.9) to explore diverse solutions
           b. Evaluating all answers using one of three strategies:
              - Self-evaluation: Compare each alternative against the current best answer
              - Pairwise evaluation: Similar to self-evaluation but with different implementation
              - Default strategy: Evaluate all answers together to select the best one
           c. Using the best answer as the current answer for the next round
        3. Returning the final best answer after all refinement rounds

        The evaluation strategy is determined by configuration settings:
        - use_self_evaluation: Uses the evaluator to compare alternatives one by one
        - use_pairwise_evaluation: Similar approach with different implementation details
        - Default: Uses the evaluation_strategy to rank all answers at once

        Args:
            question: The question to answer

        Returns:
            The final best answer after all refinement rounds

        """
        if hasattr(self.config, "enable_monitoring"):
            GLOBAL_MONITOR.enable(self.config.enable_monitoring)

        if hasattr(self.config, "use_caching") and hasattr(
            self.llm_client, "enable_cache"
        ):
            self.llm_client.enable_cache(self.config.use_caching)

        if hasattr(self.config, "concurrency_limit") and hasattr(
            self.llm_client, "set_concurrency_limit"
        ):
            self.llm_client.set_concurrency_limit(self.config.concurrency_limit)

        GLOBAL_MONITOR.start("initial_generation")
        initial_prompt = self.template_manager.render_template(
            "initial_prompt.j2", {"question": question}
        )

        initial_temperature = 0.7
        if (
            hasattr(self.config, "use_adaptive_temperature")
            and self.config.use_adaptive_temperature
        ):
            initial_temperature = getattr(self.config, "initial_temperature", 0.7)

        current_answer = self.llm_client.generate(
            initial_prompt, temperature=initial_temperature
        )
        GLOBAL_MONITOR.end("initial_generation")

        self._round_similarities = []

        if self.max_rounds <= 0:
            return current_answer

        previous_answer = ""

        for round_num in range(1, self.max_rounds + 1):
            if round_num > 1:
                similarity = self._calculate_similarity(current_answer, previous_answer)
                self._round_similarities.append(similarity)

                if (
                    self.config.early_termination
                    and similarity >= self.config.early_termination_threshold
                ):
                    break

            previous_answer = current_answer
            alternatives = self._generate_alternatives(question, current_answer)

            if self.use_self_evaluation:
                best_answer = current_answer

                for alternative in alternatives:
                    if self.evaluator.evaluate(
                        question,
                        best_answer,
                        alternative,
                        self.llm_client,
                        self.template_manager,
                    ):
                        best_answer = alternative

                current_answer = best_answer
            elif self.use_pairwise_evaluation:
                best_answer = current_answer

                for alternative in alternatives:
                    if self.evaluator.evaluate(
                        question,
                        best_answer,
                        alternative,
                        self.llm_client,
                        self.template_manager,
                    ):
                        best_answer = alternative

                current_answer = best_answer
            else:
                all_answers = [current_answer] + alternatives

                best_index = self.evaluation_strategy.evaluate(
                    question, all_answers, self.llm_client, self.template_manager
                )

                current_answer = all_answers[best_index]

        return current_answer

    @timed("generate_alternatives")
    def _generate_alternatives(self, question: str, current_answer: str) -> List[str]:
        """Generate alternative answers to the question.

        Args:
            question: The original question
            current_answer: The current best answer

        Returns:
            List of alternative answers

        """
        generation_temperature = 0.9
        if (
            hasattr(self.config, "use_adaptive_temperature")
            and self.config.use_adaptive_temperature
        ):
            base_temp = getattr(self.config, "generation_temperature", 0.9)
            min_temp = getattr(self.config, "min_generation_temperature", 0.5)
            decay_rate = getattr(self.config, "temperature_decay_rate", 0.8)
            round_num = len(self._round_similarities) + 1

            generation_temperature = max(min_temp, base_temp * (decay_rate**round_num))

            if self._round_similarities:
                avg_similarity = sum(self._round_similarities) / len(
                    self._round_similarities
                )
                if avg_similarity > 0.8:
                    generation_temperature *= 0.9  # Reduce by 10%
                elif avg_similarity < 0.4:
                    generation_temperature = min(
                        base_temp, generation_temperature * 1.1
                    )  # Increase by 10%

        alternatives = []
        for i in range(self.alternatives):
            GLOBAL_MONITOR.start(f"alternative_generation_{i}")
            prompt = self.template_manager.render_template(
                "alternative_prompt.j2",
                {"question": question, "current_answer": current_answer},
            )

            alternative = self.llm_client.generate(
                prompt, temperature=generation_temperature
            )
            alternatives.append(alternative)
            GLOBAL_MONITOR.end(f"alternative_generation_{i}")

        return alternatives

    @timed("run_async")
    async def run_async(self, question: str) -> str:
        """Execute the ThinkThread process asynchronously on a question.

        This method provides a non-blocking way to run the ThinkThread reasoning process,
        making it suitable for use in async applications like web servers, GUI
        applications, or any context where you don't want to block the main thread.

        The async implementation follows the same logical flow as the synchronous
        version but uses async LLM calls throughout the process. This provides
        several benefits:

        1. Improved responsiveness in interactive applications
        2. Better resource utilization in server environments
        3. Ability to handle multiple reasoning sessions concurrently
        4. Integration with other async frameworks and libraries

        The implementation awaits each LLM call and uses helper methods that are
        also async to maintain the non-blocking nature throughout the entire
        reasoning process.

        Args:
            question: The question to answer

        Returns:
            The final best answer after all refinement rounds

        Note:
            This method is safe to call concurrently from multiple tasks, as
            the state is maintained within the method's execution context.

        """
        if hasattr(self.config, "enable_monitoring"):
            GLOBAL_MONITOR.enable(self.config.enable_monitoring)

        if hasattr(self.config, "use_caching") and hasattr(
            self.llm_client, "enable_cache"
        ):
            self.llm_client.enable_cache(self.config.use_caching)

        if hasattr(self.config, "concurrency_limit") and hasattr(
            self.llm_client, "set_concurrency_limit"
        ):
            self.llm_client.set_concurrency_limit(self.config.concurrency_limit)

        GLOBAL_MONITOR.start("initial_generation_async")
        initial_prompt = self.template_manager.render_template(
            "initial_prompt.j2", {"question": question}
        )

        initial_temperature = 0.7
        if (
            hasattr(self.config, "use_adaptive_temperature")
            and self.config.use_adaptive_temperature
        ):
            initial_temperature = getattr(self.config, "initial_temperature", 0.7)

        current_answer = await self.llm_client.acomplete(
            initial_prompt, temperature=initial_temperature
        )
        GLOBAL_MONITOR.end("initial_generation_async")

        self._round_similarities = []

        if self.max_rounds <= 0:
            return current_answer

        previous_answer = ""

        for round_num in range(1, self.max_rounds + 1):
            if round_num > 1:
                similarity = self._calculate_similarity(current_answer, previous_answer)
                self._round_similarities.append(similarity)

                if (
                    self.config.early_termination
                    and similarity >= self.config.early_termination_threshold
                ):
                    break

            previous_answer = current_answer
            alternatives = await self._generate_alternatives_async(
                question, current_answer
            )

            if self.use_self_evaluation or self.use_pairwise_evaluation:
                best_answer = current_answer

                if not self.config.parallel_evaluation:
                    # Sequential evaluation
                    for alternative in alternatives:
                        is_better = await self._evaluate_async(
                            question, best_answer, alternative
                        )
                        if is_better:
                            best_answer = alternative
                else:
                    # Parallel evaluation
                    async def evaluate_alternative(alt):
                        is_better = await self._run_with_semaphore(
                            self._evaluate_async, question, best_answer, alt
                        )
                        return alt, is_better

                    tasks = [evaluate_alternative(alt) for alt in alternatives]
                    results = await asyncio.gather(*tasks)

                    for alt, is_better in results:
                        if is_better:
                            best_answer = alt

                current_answer = best_answer
            else:
                all_answers = [current_answer] + alternatives

                best_index = await self._evaluate_all_async(question, all_answers)

                current_answer = all_answers[best_index]

        return current_answer

    async def _run_with_semaphore(self, func, *args, **kwargs):
        """Run a function with a semaphore for rate limiting.

        Args:
            func: The function to run
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function
        """
        if (
            hasattr(self.llm_client, "_semaphore")
            and self.llm_client._semaphore is not None
        ):
            async with self.llm_client._semaphore:
                return await func(*args, **kwargs)
        return await func(*args, **kwargs)

    async def _generate_alternatives_async(
        self, question: str, current_answer: str
    ) -> List[str]:
        """Asynchronously generate alternative answers to the question.

        This method creates multiple alternative answers to the given question based on
        the current best answer. When parallel_alternatives is enabled, it uses asyncio.gather
        to generate alternatives concurrently, potentially providing significant performance
        improvements.

        Args:
            question: The original question
            current_answer: The current best answer

        Returns:
            List of alternative answers
        """
        GLOBAL_MONITOR.start("generate_alternatives_async")
        try:
            generation_temperature = 0.9
            if (
                hasattr(self.config, "use_adaptive_temperature")
                and self.config.use_adaptive_temperature
            ):
                base_temp = getattr(self.config, "generation_temperature", 0.9)
                min_temp = getattr(self.config, "min_generation_temperature", 0.5)
                decay_rate = getattr(self.config, "temperature_decay_rate", 0.8)
                round_num = len(self._round_similarities) + 1

                # Exponential decay formula: temp = base_temp * (decay_rate^round_num)
                generation_temperature = max(
                    min_temp, base_temp * (decay_rate**round_num)
                )

                if self._round_similarities:
                    avg_similarity = sum(self._round_similarities) / len(
                        self._round_similarities
                    )
                    if avg_similarity > 0.8:
                        generation_temperature *= 0.9  # Reduce by 10%
                    # If similarity is low (diverging), increase temperature
                    elif avg_similarity < 0.4:
                        generation_temperature = min(
                            base_temp, generation_temperature * 1.1
                        )  # Increase by 10%

            if not self.config.parallel_alternatives:
                # Original sequential implementation
                alternatives = []
                for i in range(self.alternatives):
                    GLOBAL_MONITOR.start(f"alternative_generation_async_{i}")
                    prompt = self.template_manager.render_template(
                        "alternative_prompt.j2",
                        {"question": question, "current_answer": current_answer},
                    )
                    alternative = await self.llm_client.acomplete(
                        prompt, temperature=generation_temperature
                    )
                    alternatives.append(alternative)
                    GLOBAL_MONITOR.end(f"alternative_generation_async_{i}")
                return alternatives

            if (
                hasattr(self.config, "use_batched_requests")
                and self.config.use_batched_requests
                and hasattr(self.llm_client, "acomplete_batch")
            ):
                prompts = []
                for i in range(self.alternatives):
                    GLOBAL_MONITOR.start(f"alternative_generation_async_{i}")
                    prompt = self.template_manager.render_template(
                        "alternative_prompt.j2",
                        {"question": question, "current_answer": current_answer},
                    )
                    prompts.append(prompt)

                results = await self._run_with_semaphore(
                    self.llm_client.acomplete_batch,
                    prompts,
                    temperature=generation_temperature,
                )

                for i in range(self.alternatives):
                    GLOBAL_MONITOR.end(f"alternative_generation_async_{i}")

                return results

            # Original parallel implementation
            async def generate_alternative(i):
                GLOBAL_MONITOR.start(f"alternative_generation_async_{i}")
                try:
                    prompt = self.template_manager.render_template(
                        "alternative_prompt.j2",
                        {"question": question, "current_answer": current_answer},
                    )
                    return await self._run_with_semaphore(
                        self.llm_client.acomplete,
                        prompt,
                        temperature=generation_temperature,
                    )
                finally:
                    GLOBAL_MONITOR.end(f"alternative_generation_async_{i}")

            tasks = [generate_alternative(i) for i in range(self.alternatives)]
            alternatives = await asyncio.gather(*tasks)
            return alternatives
        finally:
            GLOBAL_MONITOR.end("generate_alternatives_async")

    async def _evaluate_async(self, question: str, answer1: str, answer2: str) -> bool:
        """Asynchronously evaluate whether answer2 is better than answer1.

        This method is the async counterpart to the evaluator's evaluate method.
        It uses asyncio.to_thread to run the synchronous evaluation in a separate
        thread without blocking the event loop, allowing other async tasks to
        continue running during the evaluation process.

        Using a thread-based approach for evaluation is appropriate here because:
        1. The evaluation logic is CPU-bound rather than I/O-bound
        2. The existing evaluator interface is synchronous
        3. It avoids duplicating complex evaluation logic in an async version

        This approach maintains the non-blocking nature of the async reasoning
        loop while reusing the existing evaluation logic.

        Args:
            question: The original question
            answer1: The first answer
            answer2: The second answer

        Returns:
            True if answer2 is better than answer1, False otherwise

        Note:
            This method is thread-safe and can be called concurrently from
            multiple tasks.

        """
        GLOBAL_MONITOR.start("evaluate_async")
        try:
            return await asyncio.to_thread(
                self.evaluator.evaluate,
                question,
                answer1,
                answer2,
                self.llm_client,
                self.template_manager,
            )
        finally:
            GLOBAL_MONITOR.end("evaluate_async")

    async def _evaluate_all_async(self, question: str, answers: List[str]) -> int:
        """Asynchronously evaluate all answers and return the index of the best one.

        This method is the async counterpart to the evaluation strategy's evaluate
        method. Similar to _evaluate_async, it uses asyncio.to_thread to run the
        synchronous evaluation in a separate thread without blocking the event loop.

        The evaluation strategy compares all answers simultaneously and returns
        the index of the best one. This is more efficient than pairwise comparison
        when evaluating multiple alternatives, as it requires only a single LLM call
        rather than multiple comparisons.

        Using a thread-based approach for evaluation maintains the non-blocking
        nature of the async reasoning loop while reusing the existing evaluation
        logic, which is particularly important for this potentially complex
        multi-answer evaluation.

        Args:
            question: The original question
            answers: List of answers to evaluate

        Returns:
            Index of the best answer in the list

        Note:
            This method is thread-safe and can be called concurrently from
            multiple tasks.

        """
        GLOBAL_MONITOR.start("evaluate_all_async")
        try:
            return await asyncio.to_thread(
                self.evaluation_strategy.evaluate,
                question,
                answers,
                self.llm_client,
                self.template_manager,
            )
        finally:
            GLOBAL_MONITOR.end("evaluate_all_async")

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate the similarity between two strings.

        Args:
            str1: First string
            str2: Second string

        Returns:
            A similarity score between 0.0 and 1.0
        """
        use_fast = (
            hasattr(self.config, "use_fast_similarity")
            and self.config.use_fast_similarity
        )
        return calculate_similarity(str1, str2, fast=use_fast)
