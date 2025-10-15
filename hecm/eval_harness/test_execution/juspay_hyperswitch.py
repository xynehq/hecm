import os
from typing import Dict, List

from hecm.dataset_generation.schemas import CodingAgentDataPoint
from hecm.eval_harness.test_execution.base import BaseTestExecutor


class JuspayHyperswitchTestExecutor(BaseTestExecutor):
    """Executes tests for Juspay Hyperswitch in a Docker container with Rust.

    !!! example
        ```python
        from datasets import load_dataset
        from hecm.dataset_generation.schemas import CodingAgentDataPoint
        from hecm.eval_harness.test_execution import JuspayHyperswitchTestExecutor

        dataset = load_dataset("geekyrakshit/rust-dev", split="train")
        data_point = CodingAgentDataPoint.model_validate(dataset[0])
        executor = JuspayHyperswitchTestExecutor()
        results = executor.execute(data_point)
        ```


    Args:
        image (str): Docker image to use (default: rust:latest)
        working_dir (str): Working directory inside the container (default: /workspace)
        environment (Dict[str, str]): Environment variables to set in the container (default: None)
    """

    def __init__(
        self,
        image: str = "rust:latest",
        working_dir: str = "/workspace",
        environment: Dict[str, str] = None,
    ):
        super().__init__(image=image, working_dir=working_dir, environment=environment)

    def get_commands(self, data_point: CodingAgentDataPoint) -> List[str]:
        repo_dir = os.path.join(self.working_dir, "repo")

        # Prepare commands for test execution
        commands = [
            # Clone the repository
            f"git clone https://github.com/{data_point.repo}.git {repo_dir}",
            # Checkout the base commit
            f"cd {repo_dir} && git checkout {data_point.base_commit}",
            # Apply the test patch
            # f"cd {repo_dir} && echo '{data_point.patch}' | git apply -",
            # Show the git diff
            f"cd {repo_dir} && git diff",
        ]
        return commands
