import os
import subprocess
import time
from typing import Dict, List, Optional

import rich

from hecm.dataset_generation.schemas import CodingAgentDataPoint
from hecm.eval_harness.test_execution.legacy.base import (
    BaseLocalExecutor,
    BaseSandboxedExecutor,
    CommandExecutionResult,
    DataPointExecutionSummary,
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

    def get_cypress_test_commands(self) -> List[str]:
        test_dir = os.path.join(self.repo_dir, "cypress-tests-v2")
        return [
            # install nodejs and npm
            "apt-get update",
            "apt-get install -y nodejs npm",
            # install cypress dependencies
            f"cd {test_dir} && npm install",
            # run all tests in a headless manner
            f"cd {test_dir} && npm run cypress{self.cypress_test_suffix}",
        ]

    def get_commands(
        self, data_point: CodingAgentDataPoint, predicted_patch: Optional[str] = None
    ) -> List[str]:
        """
        Get commands for executing tests for a given data point.

        Args:
            data_point: The data point to execute tests for
            predicted_patch: The predicted patch to apply to the repository
        """
        # Prepare commands for test execution
        commands = [
            # Clone the repository
            f"git clone https://github.com/{data_point.repo}.git {self.repo_dir}",
            # Checkout the base commit
            f"cd {self.repo_dir} && git checkout {data_point.base_commit}",
            # Apply the test patch
            *self.get_patch_commands(
                patch=data_point.patch if predicted_patch is None else predicted_patch,
                repo_dir=self.repo_dir,
            ),
            # Show the git diff
            f"cd {self.repo_dir} && git diff",
            # execute cypress tests
            *self.get_cypress_test_commands(repo_dir=self.repo_dir),
        ]
        return commands


class JuspayHyperswitchLocalTestExecutor(BaseLocalExecutor):
    """Executes tests for Juspay Hyperswitch in the local environment.

    ```
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
            cypress_test_suffix (str): Suffix to add to the cypress test command (default: ":payments")
    """

    def __init__(
        self,
        environment: Dict[str, str] = None,
        show_output_logs: bool = True,
        cypress_test_suffix: str = ":payments",
        health_check_url: str = "http://localhost:8080/health",
        health_check_timeout: int = 3000,
        health_check_interval: int = 5,
    ):
        super().__init__(environment=environment, show_output_logs=show_output_logs)
        self.cypress_test_suffix = cypress_test_suffix
        self.health_check_url = health_check_url
        self.health_check_timeout = health_check_timeout
        self.health_check_interval = health_check_interval

    def poll_hyperswitch(self) -> bool:
        start_time = time.time()
        print(f"Polling {self.health_check_url} for health check...")
        while time.time() - start_time < self.health_check_timeout:
            try:
                result = subprocess.run(
                    ["curl", "--head", "--request", "GET", self.health_check_url],
                    capture_output=True,
                    timeout=10,
                )
                if self.show_output_logs:
                    print(f"Health check attempt: status code {result.returncode}")
                if result.returncode == 0:
                    if self.show_output_logs:
                        print(f"Hyperswitch is ready at {self.health_check_url}")
                    return True
            except subprocess.TimeoutExpired:
                if self.show_output_logs:
                    print("Health check timed out")
            time.sleep(self.health_check_interval)

        raise TimeoutError(
            f"Hyperswitch health check failed after {self.health_check_timeout} seconds"
        )

    def get_patch_commands(self, patch: str) -> List[str]:
        patch_file = os.path.join(self.repo_dir, "changes.patch")
        patch_file_generation_command = f"""cat > {patch_file} << 'EOF'\n{patch}\nEOF"""
        return [
            # Generate the patch file
            patch_file_generation_command,
            # Check if the patch is valid
            f"cd {self.repo_dir} && git apply --check {patch_file}",
            # Apply the patch
            f"cd {self.repo_dir} && git apply changes.patch",
        ]

    def get_cypress_test_commands(self) -> List[str]:
        test_dir = os.path.join(self.repo_dir, "cypress-tests-v2")
        return [
            # install nodejs and npm
            "apt-get update",
            "apt-get install -y nodejs npm",
            # install cypress dependencies
            f"cd {test_dir} && npm install",
            # run all tests in a headless manner
            f"cd {test_dir} && npx cypress run",
        ]

    def get_commands(self, data_point: CodingAgentDataPoint) -> List[str]:
        # Prepare commands for test execution
        commands = [
            # Clone the repository
            f"git clone https://github.com/{data_point.repo}.git {self.repo_dir}",
            # Checkout the base commit
            f"cd {self.repo_dir} && git checkout {data_point.base_commit}",
            # Apply the test patch
            *self.get_patch_commands(patch=data_point.patch),
            # Show the git diff
            f"cd {self.repo_dir} && git diff",
            # set up hyperswitch development environment
            f"cd {self.repo_dir} && docker compose --file docker-compose-development.yml up -d",
            # execute cypress tests
            *self.get_cypress_test_commands(),
        ]
        return commands

    def execute(
        self, data_point: CodingAgentDataPoint, predicted_patch: Optional[str] = None
    ):
        """
        Execute tests for a given data point.

        Args:
            data_point: The data point to execute tests for
            predicted_patch: The predicted patch to apply to the repository
        """
        commands_before_health_check = [
            f"git clone https://github.com/{data_point.repo}.git {self.repo_dir}",
            f"cd {self.repo_dir} && git checkout {data_point.base_commit}",
            *self.get_patch_commands(
                patch=data_point.patch if predicted_patch is None else predicted_patch,
            ),
            f"cd {self.repo_dir} && git diff",
            f"cd {self.repo_dir} && docker compose --file docker-compose-development.yml up -d",
        ]

        for cmd in commands_before_health_check:
            subprocess.run(cmd, shell=True, check=True, env=self.environment)

        self.poll_hyperswitch()
        env = self.environment.copy() if self.environment else {}
        env["CYPRESS_BASEURL"] = "http://localhost:8080"

        commands_after_health_check = self.get_cypress_test_commands(
            repo_dir=self.repo_dir
        )
        for cmd in commands_after_health_check:
            subprocess.run(cmd, shell=True, check=True, env=self.environment)

        command_execution_results = []
        for cmd in commands_before_health_check:
            command_execution_results.append(
                CommandExecutionResult(
                    command=cmd,
                    exit_code=0,
                    output="",
                )
            )

        return DataPointExecutionSummary(
            instance_id=data_point.instance_id,
            repo=data_point.repo,
            base_commit=data_point.base_commit,
            all_commands_executed_successfully=True,
            command_results=command_execution_results,
        )

    def cleanup(self):
        try:
            subprocess.run(
                f"cd {self.repo_dir} && docker compose --file docker-compose-development.yml down",
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            rich.print(
                f"[red]Error during cleanup: {subprocess.CalledProcessError}[/red]"
            )
        super().cleanup()
