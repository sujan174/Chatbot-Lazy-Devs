from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseAgent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.initialized = False
    
    @abstractmethod
    async def initialize(self):
        """
        Initialize the agent (connect to services, load tools, etc.)
        This is called once when the agent is discovered.
        """
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> List[str]:
        """
        Return a list of capabilities this agent provides.
        These will be shown to the orchestrator to help it decide when to use this agent.
        
        Example:
        return [
            "Create and update Jira tickets",
            "Search Jira issues",
            "Add comments to issues"
        ]
        """
        pass
    
    @abstractmethod
    async def execute(self, instruction: str) -> str:
        """
        Execute a task based on the instruction.
        
        Args:
            instruction: Natural language instruction describing what to do
            
        Returns:
            String result of the execution
        """
        pass
    
    async def cleanup(self):
        """
        Cleanup resources (disconnect from services, etc.)
        Override if needed.
        """
        pass
    
    def _format_error(self, error: Exception) -> str:
        """Helper to format error messages"""
        return f"Error in {self.name}: {str(error)}"