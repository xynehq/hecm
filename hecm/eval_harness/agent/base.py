from abc import ABC, abstractmethod

from pydantic import BaseModel

from hecm.dataset_generation.schemas import CodingAgentDataPoint


class AgentResponse(BaseModel):
    patch: str
    files_changed: list[str]
    success: bool
    execution_time: float
    log_file: str
    stdout: str
    stderr: str
    exit_code: int


class BaseAgent(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_agent_response(self, data_point: CodingAgentDataPoint) -> AgentResponse:
        raise NotImplementedError(
            "Subclasses to `BaseAgent` must implement the `get_agent_response` method."
        )
