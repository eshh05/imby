"""
Base agent class for the research paper summarization system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class AgentResult:
    """Standard result format for all agents."""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"agent.{name}")
        
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute the agent's main functionality."""
        pass
    
    def validate_input(self, input_data: Dict[str, Any], required_keys: List[str]) -> bool:
        """Validate that input contains all required keys."""
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            self.logger.error(f"Missing required keys: {missing_keys}")
            return False
        return True
    
    def log_execution(self, input_data: Dict[str, Any], result: AgentResult):
        """Log agent execution details."""
        self.logger.info(f"Agent {self.name} executed with success: {result.success}")
        if not result.success and result.error:
            self.logger.error(f"Agent {self.name} error: {result.error}")
