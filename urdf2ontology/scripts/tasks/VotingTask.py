from abc import ABC, abstractmethod
from typing import Any, List

class VotingTask(ABC):
    def __init__(self, nb_runs, keyword):
        self.nb_runs = nb_runs
        self.keyword = keyword
       
    # @abstractmethod
    # def prompt(self, user_input: str) -> str:
    #     """
    #     Build the prompt for this task.
    #     """
    #     pass
    
    @abstractmethod
    def system_prompt(self) -> str:
        """
        Build the system prompt for this task.
        """
        pass
    
    @abstractmethod
    def user_prompt(self) -> str:
        """
        Build the user prompt for this task.
        """
        pass

    @abstractmethod
    def interpret(self, output: dict) -> Any:
        """
        Convert model JSON output into a vote. Transforms a task-specific JSON output into a standardized 
        into a comparable value that the voting system can reason about.
        """
        pass
    
    @abstractmethod
    def aggregate(self, votes: List[Any]) -> Any:
        """
        Aggregate interpreted votes.
        """
        pass
