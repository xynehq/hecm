import os
from typing import Dict, List

from hecm.dataset_generation.schemas import CodingAgentDataPoint
from hecm.eval_harness.test_execution.base import (
    BaseLocalExecutor,
    BaseSandboxedExecutor,
)


class JuspayHyperswitchSandboxedTestExecutor(BaseSandboxedExecutor):
    """Executes tests for Juspay Hyperswitch in a Docker container with Rust.

    !!! example
        ```python
        import weave
        from datasets import load_dataset

        from hecm.dataset_generation.schemas import CodingAgentDataPoint
        from hecm.eval_harness.test_execution import JuspayHyperswitchSandboxedTestExecutor


        @weave.op
        def test_cypress_execution():
            dataset = load_dataset("geekyrakshit/rust-dev", split="train")
            data_point = CodingAgentDataPoint.model_validate(dataset[0])
            executor = JuspayHyperswitchSandboxedTestExecutor(show_output_logs=True)
            results = executor.execute(data_point)
            executor.cleanup()
            return results


        weave.init(project_name="hyperswitch")
        test_cypress_execution()
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
        cypress_test_suffix: str = ":payments",
    ):
        super().__init__(image=image, working_dir=working_dir, environment=environment)
        self.cypress_test_suffix = cypress_test_suffix

    def get_patch_commands(self, patch: str, repo_dir: os.PathLike) -> List[str]:
        patch_file = os.path.join(repo_dir, "changes.patch")
        patch_file_generation_command = f"""cat > {patch_file} << 'EOF'\n{patch}\nEOF"""
        return [
            # Generate the patch file
            patch_file_generation_command,
            # Check if the patch is valid
            f"cd {repo_dir} && git apply --check {patch_file}",
            # Apply the patch
            f"cd {repo_dir} && git apply changes.patch",
        ]

    def get_cypress_test_commands(self, repo_dir: os.PathLike) -> List[str]:
        test_dir = os.path.join(repo_dir, "cypress-tests-v2")
        return [
            # install nodejs and npm
            "apt-get update",
            "apt-get install -y nodejs npm",
            # install cypress dependencies
            f"cd {test_dir} && npm install",
            # run all tests in a headless manner
            f"cd {test_dir} && npm run cypress{self.cypress_test_suffix}",
        ]

    def get_commands(self, data_point: CodingAgentDataPoint) -> List[str]:
        repo_dir = os.path.join(self.working_dir, "repo")

        # Prepare commands for test execution
        commands = [
            # Clone the repository
            f"git clone https://github.com/{data_point.repo}.git {repo_dir}",
            # Checkout the base commit
            f"cd {repo_dir} && git checkout {data_point.base_commit}",
            # Apply the test patch
            *self.get_patch_commands(patch=data_point.patch, repo_dir=repo_dir),
            # Show the git diff
            f"cd {repo_dir} && git diff",
            # execute cypress tests
            *self.get_cypress_test_commands(repo_dir=repo_dir),
        ]
        return commands


class JuspayHyperswitchLocalTestExecutor(BaseLocalExecutor):
    """Executes tests for Juspay Hyperswitch in the local environment.

    !!! example
        ```python
        import weave
        from datasets import load_dataset

        from hecm.dataset_generation.schemas import CodingAgentDataPoint
        from hecm.eval_harness.test_execution import JuspayHyperswitchLocalTestExecutor


        @weave.op
        def test_cypress_execution():
            dataset = load_dataset("geekyrakshit/rust-dev", split="train")
            data_point = CodingAgentDataPoint.model_validate(dataset[0])
            executor = JuspayHyperswitchLocalTestExecutor(show_output_logs=True)
            results = executor.execute(data_point)
            executor.cleanup()
            return results


        weave.init(project_name="hyperswitch")
        test_cypress_execution()
        ```

    Args:
        environment (Dict[str, str]): Environment variables to set (default: None)
        show_output_logs (bool): Whether to show output logs (default: True)
    """

    def __init__(
        self, environment: Dict[str, str] = None, show_output_logs: bool = True
    ):
        super().__init__(environment=environment, show_output_logs=show_output_logs)

    def get_patch_commands(self, patch: str, repo_dir: os.PathLike) -> List[str]:
        patch_file = os.path.join(repo_dir, "changes.patch")
        patch_file_generation_command = f"""cat > {patch_file} << 'EOF'\n{patch}\nEOF"""
        return [
            # Generate the patch file
            patch_file_generation_command,
            # Check if the patch is valid
            f"cd {repo_dir} && git apply --check {patch_file}",
            # Apply the patch
            f"cd {repo_dir} && git apply changes.patch",
        ]

    def get_cypress_test_commands(self, repo_dir: os.PathLike) -> List[str]:
        test_dir = os.path.join(repo_dir, "cypress-tests-v2")
        return [
            # install nodejs and npm
            "apt-get update",
            "apt-get install -y nodejs npm",
            # install cypress dependencies
            f"cd {test_dir} && npm install",
            # run all tests in a headless manner
            f"cd {test_dir} && npm run cypress{self.cypress_test_suffix}",
        ]

    def get_commands(self, data_point: CodingAgentDataPoint) -> List[str]:
        repo_dir = os.path.join(self.working_dir.name, "repo")

        # Prepare commands for test execution
        commands = [
            # Clone the repository
            f"git clone https://github.com/{data_point.repo}.git {repo_dir}",
            # Checkout the base commit
            f"cd {repo_dir} && git checkout {data_point.base_commit}",
            # Apply the test patch
            *self.get_patch_commands(patch=data_point.patch, repo_dir=repo_dir),
            # Show the git diff
            f"cd {repo_dir} && git diff",
            # execute cypress tests
            # *self.get_cypress_test_commands(repo_dir=repo_dir),
        ]
        return commands
