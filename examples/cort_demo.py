import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkthread_sdk.session import ThinkThreadSession
from thinkthread_sdk.llm.dummy import DummyLLMClient
from thinkthread_sdk.evaluation import EvaluationStrategy
from typing import List


class VerboseEvaluationStrategy(EvaluationStrategy):
    def evaluate(self, question, answers, llm_client, template_manager):
        print("Evaluating answers:")
        for i, answer in enumerate(answers):
            print(f"Answer {i + 1}:\n{answer}\n")

        selected = len(answers) - 1
        print(f"Selected answer {selected + 1}\n")
        return selected


responses = [
    "Initial answer: AI is a field of computer science.",
    "Alternative 1: AI involves creating systems that can perform tasks normally requiring human intelligence.",
    "Alternative 2: AI refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.",
    "Alternative 3: AI encompasses various technologies including machine learning, natural language processing, and computer vision.",
]

client = DummyLLMClient(responses=responses)

session = ThinkThreadSession(
    llm_client=client,
    max_rounds=1,
    alternatives=3,
    evaluation_strategy=VerboseEvaluationStrategy(),
)

os.makedirs("examples", exist_ok=True)

question = "What is artificial intelligence?"
print(f"Question: {question}\n")

answer = session.run(question)

print(f"Final answer: {answer}")

print("\n--- With Pairwise Evaluation ---\n")

pairwise_responses = [
    "Initial answer: AI is a field of computer science.",
    "Alternative 1: AI involves creating systems that can perform tasks normally requiring human intelligence.",
    "Alternative 2: AI refers to the simulation of human intelligence in machines that are programmed to think and learn like humans.",
    "Alternative 3: AI encompasses various technologies including machine learning, natural language processing, and computer vision.",
    "The new answer is better.",
    "The new answer is better.",
    "The new answer is better.",
]

pairwise_client = DummyLLMClient(responses=pairwise_responses)

from thinkthread_sdk.config import ThinkThreadConfig

pairwise_config = ThinkThreadConfig()
pairwise_config.use_pairwise_evaluation = True

pairwise_session = ThinkThreadSession(
    llm_client=pairwise_client, max_rounds=1, alternatives=3, config=pairwise_config
)

pairwise_answer = pairwise_session.run(question)

print(f"Final answer with pairwise evaluation: {pairwise_answer}")
