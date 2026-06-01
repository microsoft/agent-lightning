import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TrajectoryStep(BaseModel):
    """A single step in the agent trajectory."""
    thought: str = ""
    reasoning_content: str = ""
    action: Dict[str, Any] = Field(default_factory=dict)  # Can be dict or str
    observation: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought,
            "reasoning_content": self.reasoning_content,
            "action": self.action,
            "observation": self.observation,
            "metadata": self.metadata,
        }


class Trajectory(BaseModel):
    """Complete trajectory of an agent run."""
    problem_statement: str
    steps: List[TrajectoryStep] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_statement": self.problem_statement,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
        }
    reward_calc_time: Optional[float] = None
    test_output: Optional[str] = None
