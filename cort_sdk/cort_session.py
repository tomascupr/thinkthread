from typing import List, Optional

from cort_sdk.llm import LLMClient
from cort_sdk.prompting import TemplateManager
from cort_sdk.config import CoRTConfig, create_config
from cort_sdk.evaluation import EvaluationStrategy, DefaultEvaluationStrategy, Evaluator, ModelEvaluator


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
        max_rounds: Optional[int] = None,
        template_manager: Optional[TemplateManager] = None,
        evaluation_strategy: Optional[EvaluationStrategy] = None,
        evaluator: Optional[Evaluator] = None,
        config: Optional[CoRTConfig] = None,
    ):
        """
        Initialize a CoRT session.
        
        Args:
            llm_client: LLM client to use for generating and evaluating answers
            alternatives: Number of alternative answers to generate per round
            rounds: Number of refinement rounds to perform (for backward compatibility)
            max_rounds: Maximum number of refinement rounds (overrides rounds if set)
            template_manager: Optional template manager for prompt templates
            evaluation_strategy: Optional strategy for evaluating answers
            config: Optional configuration object
        """
        self.llm_client = llm_client
        self.alternatives = alternatives
        self.rounds = rounds
        
        # Initialize configuration and template manager
        self.config = config or create_config()
        self.template_manager = template_manager or TemplateManager(self.config.prompt_dir)
        
        self.max_rounds = max_rounds if max_rounds is not None else self.rounds
        
        # Initialize evaluation strategy
        self.evaluation_strategy = evaluation_strategy or DefaultEvaluationStrategy()
        
        # Initialize evaluator for pairwise comparison
        self.evaluator = evaluator or ModelEvaluator()
        self.use_pairwise_evaluation = self.config.use_pairwise_evaluation
        self.use_self_evaluation = self.config.use_self_evaluation
    
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
        initial_prompt = self.template_manager.render_template(
            "initial_prompt.j2", {"question": question}
        )
        current_answer = self.llm_client.generate(initial_prompt, temperature=0.7)
        
        if self.max_rounds <= 0:
            return current_answer
        
        for round_num in range(1, self.max_rounds + 1):
            alternatives = self._generate_alternatives(question, current_answer)
            
            if self.use_self_evaluation:
                best_answer = current_answer
                
                for alternative in alternatives:
                    # Use the evaluator to decide if the alternative is better
                    if self.evaluator.evaluate(
                        question, best_answer, alternative, self.llm_client, self.template_manager
                    ):
                        best_answer = alternative
                
                current_answer = best_answer
            elif self.use_pairwise_evaluation:
                best_answer = current_answer
                
                for alternative in alternatives:
                    if self.evaluator.evaluate(
                        question, best_answer, alternative, self.llm_client, self.template_manager
                    ):
                        best_answer = alternative
                
                current_answer = best_answer
            else:
                all_answers = [current_answer] + alternatives
                
                # Use the evaluation strategy to select the best answer
                best_index = self.evaluation_strategy.evaluate(
                    question, all_answers, self.llm_client, self.template_manager
                )
                
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
            prompt = self.template_manager.render_template(
                "alternative_prompt.j2",
                {
                    "question": question,
                    "current_answer": current_answer
                }
            )
            
            alternative = self.llm_client.generate(prompt, temperature=0.9)
            alternatives.append(alternative)
        
        return alternatives
    
    # They are now implemented in DefaultEvaluationStrategy
