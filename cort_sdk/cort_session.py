from typing import List, Optional

from cort_sdk.llm import LLMClient


class CoRTSession:
    """
    Chain-of-Recursive-Thoughts (CoRT) session.
    
    This class orchestrates a multi-round questioning and refinement process
    using an LLM to generate increasingly better answers to a question.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        alternatives: int = 3,
        rounds: int = 2,
    ):
        """
        Initialize a CoRT session.
        
        Args:
            llm_client: LLM client to use for generating and evaluating answers
            alternatives: Number of alternative answers to generate per round
            rounds: Number of refinement rounds to perform
        """
        self.llm_client = llm_client
        self.alternatives = alternatives
        self.rounds = rounds
    
    def run(self, question: str) -> str:
        """
        Execute the Chain-of-Recursive-Thoughts process on a question.
        
        The process involves:
        1. Generating an initial answer
        2. For each round:
           a. Generating alternative answers
           b. Evaluating all answers to select the best one
           c. Using the best answer as the current answer for the next round
        3. Returning the final best answer
        
        Args:
            question: The question to answer
            
        Returns:
            The final best answer after all refinement rounds
        """
        initial_prompt = f"Answer the following question thoughtfully and accurately:\n\nQuestion: {question}\n\nAnswer:"
        current_answer = self.llm_client.generate(initial_prompt, temperature=0.7)
        
        for round_num in range(1, self.rounds + 1):
            alternatives = self._generate_alternatives(question, current_answer)
            
            all_answers = [current_answer] + alternatives
            
            best_index = self._evaluate_answers(question, all_answers)
            
            current_answer = all_answers[best_index]
        
        return current_answer
    
    def _generate_alternatives(self, question: str, current_answer: str) -> List[str]:
        """
        Generate alternative answers to the question.
        
        Args:
            question: The original question
            current_answer: The current best answer
            
        Returns:
            List of alternative answers
        """
        alternatives = []
        
        for i in range(self.alternatives):
            prompt = (
                f"Question: {question}\n\n"
                f"Current answer: {current_answer}\n\n"
                f"Generate a different possible answer to the question that explores "
                f"a different approach or perspective than the current answer. "
                f"Your answer should be thoughtful, accurate, and distinct from the current answer."
            )
            
            alternative = self.llm_client.generate(prompt, temperature=0.9)
            alternatives.append(alternative)
        
        return alternatives
    
    def _evaluate_answers(self, question: str, answers: List[str]) -> int:
        """
        Evaluate multiple answers and select the best one.
        
        Args:
            question: The original question
            answers: List of answers to evaluate
            
        Returns:
            Index of the best answer in the answers list
        """
        formatted_answers = "\n\n".join([
            f"Answer {i+1}:\n{answer}" for i, answer in enumerate(answers)
        ])
        
        prompt = (
            f"Question: {question}\n\n"
            f"{formatted_answers}\n\n"
            f"Evaluate each of the above answers to the question. "
            f"Consider accuracy, completeness, clarity, and depth of reasoning. "
            f"Which answer is the best? Provide your analysis and then clearly indicate "
            f"the best answer by stating 'The best answer is Answer X' where X is the number "
            f"of the best answer (1 to {len(answers)})."
        )
        
        evaluation = self.llm_client.generate(prompt, temperature=0.2)
        
        best_index = self._parse_evaluation(evaluation, len(answers))
        
        return best_index
    
    def _parse_evaluation(self, evaluation: str, num_answers: int) -> int:
        """
        Parse the evaluation text to determine which answer was selected as best.
        
        Args:
            evaluation: The evaluation text from the LLM
            num_answers: The number of answers that were evaluated
            
        Returns:
            Index of the best answer (0 to num_answers-1)
        """
        for i in range(1, num_answers + 1):
            if f"best answer is Answer {i}" in evaluation or f"Best answer is Answer {i}" in evaluation:
                return i - 1  # Convert to 0-based index
        
        for i in range(1, num_answers + 1):
            indicators = [
                f"Answer {i} is the best",
                f"select Answer {i}",
                f"choose Answer {i}",
                f"prefer Answer {i}",
            ]
            for indicator in indicators:
                if indicator in evaluation:
                    return i - 1
        
        return 0
