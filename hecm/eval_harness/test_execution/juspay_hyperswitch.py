import json
import os
import subprocess
import time
from tempfile import TemporaryDirectory

import rich

from hecm.dataset_generation.schemas import CodingAgentDataPoint
from hecm.eval_harness.test_execution.base import (
    BaseLocalExecutor,
    CommandExecutionResult,
    EvaluationResult,
    execute_multiple_commands,
)


class JuspayHyperswitchLocalTestExecutor(BaseLocalExecutor):
    """Executor for Juspay Hyperswitch in the local environment.

    !!! example
        ```python
        from datasets import load_dataset
        from hecm.dataset_generation.schemas import CodingAgentDataPoint
        from hecm.eval_harness.agent import ClaudeCodeProxyAgent
        from hecm.eval_harness.test_execution import JuspayHyperswitchLocalTestExecutor


        dataset = load_dataset("juspay/hyperswitch", split="train")
        data_point = CodingAgentDataPoint.model_validate(dataset[0])
        agent = ClaudeCodeProxyAgent()
        response = agent.get_agent_response(data_point, start_proxy=True, stop_proxy=True)
        test_executor = JuspayHyperswitchLocalTestExecutor(
            environment={
                "CYPRESS_CONNECTOR": "connector_id",
                "CYPRESS_BASEURL": "http://localhost:8080",
                "DEBUG": "cypress:cli",
                "CYPRESS_ADMINAPIKEY": "admin_api_key",
                "CYPRESS_CONNECTOR_AUTH_FILE_PATH": "/Users/geekyrakshit/Workspace/athena/hecm/creds.json",
            },
        )
        test_executor.execute(
            data_point,
            predicted_patch=response.patch,
            result_save_path="results.json",
        )
        ```
    !!! note
        This executor uses the local environment to execute the commands.
        It is not recommended to use this executor in a production environment.
        It is only recommended to use this executor for testing purposes.
        Right now, we're using `JuspayHyperswitchLocalTestExecutor` to avoid the complexity of running docker-in-docker containers.

    Args:
        environment (dict[str, str]): The environment variables to use for the commands.
        cypress_test_suffix (str): The suffix to add to the cypress test command.
        health_check_url (str): The URL to check the health of the Hyperswitch.
        health_check_timeout (int): The timeout for the health check.
        health_check_interval (int): The interval for the health check.
    """

    def __init__(
        self,
        environment: dict[str, str] = None,
        cypress_test_suffix: str = ":payments",
        health_check_url: str = "http://localhost:8080/health",
        health_check_timeout: int = 720,
        health_check_interval: int = 5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.environment = environment
        self.cypress_test_suffix = cypress_test_suffix
        self.health_check_url = health_check_url
        self.health_check_timeout = health_check_timeout
        self.health_check_interval = health_check_interval
        self.command_results: list[CommandExecutionResult] = []

    def clone_repository(self, data_point: CodingAgentDataPoint, repo_dir: str):
        """Clone the repository for the given data point.

        Args:
            data_point (CodingAgentDataPoint): The data point to clone the repository for.
            repo_dir (str): The directory to clone the repository to.

        Returns:
            list[CommandExecutionResult]: The results of executing the commands.

        Raises:
            Exception: If the repository cannot be cloned.
        """
        command_results = execute_multiple_commands(
            [
                # Clone the repository
                f"git clone https://github.com/{data_point.repo}.git {repo_dir}",
                # Checkout the base commit
                f"cd {repo_dir} && git switch --detach {data_point.base_commit}",
            ],
            self.environment,
        )
        self.command_results.extend(command_results)

    def apply_patch(
        self,
        data_point: CodingAgentDataPoint,
        repo_dir: str,
        predicted_patch: str | None = None,
    ):
        """Apply the patch for the given data point.

        Args:
            data_point (CodingAgentDataPoint): The data point to apply the patch for.
            repo_dir (str): The directory to apply the patch to.
            predicted_patch (str | None): The predicted patch to apply.

        Returns:
            list[CommandExecutionResult]: The results of executing the commands.

        Raises:
            Exception: If the patch cannot be applied.
        """
        patch = data_point.patch if predicted_patch is None else predicted_patch
        command_results = execute_multiple_commands(
            [
                f"cat > /tmp/change.patch <<'PATCH'\n{patch}\nPATCH",
                f"cd {repo_dir} && git stash && git apply /tmp/change.patch",
                f"rm /tmp/change.patch",
            ],
            self.environment,
        )
        self.command_results.extend(command_results)

    def docker_compose_up(self, repo_dir: str):
        """Start the docker compose for the given repository.

        Args:
            repo_dir (str): The directory to start the docker compose for.

        Returns:
            list[CommandExecutionResult]: The results of executing the commands.

        Raises:
            Exception: If the docker compose cannot be started.
        """
        command_results = execute_multiple_commands(
            [
                f"cd {repo_dir} && sudo  docker compose --file docker-compose-development.yml up -d"
            ],
            self.environment,
        )
        self.command_results.extend(command_results)

        # Poll for the health check
        start_time = time.time()
        while time.time() - start_time < self.health_check_timeout:
            rich.print(
                f"[cyan]Polling {self.health_check_url} for health check...[/cyan]"
            )
            try:
                result = subprocess.run(
                    ["curl", "--head", "--request", "GET", self.health_check_url],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    rich.print(
                        f"[green]Hyperswitch is ready at {self.health_check_url}[/green]"
                    )
                    return True
            except subprocess.TimeoutExpired:
                rich.print("[red]Health check timed out[/red]")
                raise TimeoutError(
                    f"Hyperswitch health check failed after {self.health_check_timeout} seconds"
                )
            time.sleep(self.health_check_interval)

    def docker_compose_down(self, repo_dir: str):
        """Stop the docker compose for the given repository.

        Args:
            repo_dir (str): The directory to stop the docker compose for.

        Returns:
            list[CommandExecutionResult]: The results of executing the commands.

        Raises:
            Exception: If the docker compose cannot be stopped.
        """
        result = execute_multiple_commands(
            [
                f"cd {repo_dir} && sudo docker compose --file docker-compose-development.yml down",
                f"cd {repo_dir} && sudo docker system prune -f",
                f"sudo rm -rf /var/lib/docker/volumes/repo_router_build_cache/_data",
                f"sudo rm -rf /var/lib/docker/volumes/workspace_router_build_cache/_data^",
            ],
            self.environment,
        )
        self.command_results.extend(result)

    def execute_cypress_tests(self, repo_dir: str):
        """Execute the cypress tests for the given repository.

        Args:
            repo_dir (str): The directory to execute the cypress tests for.

        Returns:
            list[CommandExecutionResult]: The results of executing the commands.

        Raises:
            Exception: If the cypress tests cannot be executed.
        """
        test_dir = os.path.join(repo_dir, "cypress-tests-v2")
        command_results = execute_multiple_commands(
            [
                # install nodejs and npm
                "sudo apt-get update",
                "sudo apt-get install -y nodejs npm",
                # install cypress dependencies
                f"cd {test_dir} && npm install",
                # run all tests in a headless manner
                f"cd {test_dir} && npx cypress run",
            ],
            self.environment,
        )
        self.command_results.extend(command_results)

    def execute_cargo_test(self, repo_dir: str):
        """Execute the cargo test for the given repository.

        Args:
            repo_dir (str): The directory to execute the cargo test for.

        Returns:
            list[CommandExecutionResult]: The results of executing the commands.

        Raises:
            Exception: If the cargo test cannot be executed.
        """
        command_results = execute_multiple_commands(
            [f"cd {repo_dir} && cargo test"], self.environment
        )
        self.command_results.extend(command_results)

    def execute_commands(
        self,
        data_point: CodingAgentDataPoint,
        repo_dir: str,
        predicted_patch: str | None = None,
    ):
        """Execute the commands for the given data point.

        Args:
            data_point (CodingAgentDataPoint): The data point to execute the commands for.
            repo_dir (str): The directory to execute the commands for.
            predicted_patch (str | None): The predicted patch to apply.

        Returns:
            list[CommandExecutionResult]: The results of executing the commands.

        Raises:
            Exception: If the commands cannot be executed.
        """
        rich.print(
            f"[bold cyan]Executing tests for: {data_point.instance_id}[/bold cyan]"
        )
        rich.print(f"[cyan]Repository: {data_point.repo}[/cyan]")
        rich.print(f"[cyan]Base commit: {data_point.base_commit}[/cyan]")

        self.clone_repository(data_point, repo_dir)
        self.apply_patch(data_point, repo_dir, predicted_patch=predicted_patch)
        self.docker_compose_up(repo_dir)
        self.execute_cypress_tests(repo_dir)
        self.execute_cargo_test(repo_dir)
        self.docker_compose_down(repo_dir)

    def execute(
        self,
        data_point: CodingAgentDataPoint,
        predicted_patch: str | None = None,
        result_save_path: os.PathLike | None = None,
    ) -> EvaluationResult:
        """Execute the commands for the given data point.

        Args:
            data_point (CodingAgentDataPoint): The data point to execute the commands for.
            predicted_patch (str | None): The predicted patch to apply.
            result_save_path (os.PathLike | None): The path to save the results to.

        Returns:
            EvaluationResult: The evaluation result.

        Raises:
            Exception: If the commands cannot be executed.
        """
        working_dir = None
        try:
            with TemporaryDirectory(prefix="hyperswitch-testcase-") as working_dir:
                repo_dir = os.path.join(working_dir, "workspace")
                self.execute_commands(
                    data_point, repo_dir, predicted_patch=predicted_patch
                )
        except Exception as e:
            rich.print(f"[red]Error executing commands: {e}[/red]")
            raise
        finally:
            evaluation_result = self.get_evaluation_result()
            rich.print(f"[cyan]Saving results to {result_save_path}[/cyan]")
            if result_save_path is not None:
                with open(result_save_path, "w") as f:
                    json.dump(
                        {
                            "command_results": [
                                result.model_dump() for result in self.command_results
                            ],
                            "evaluation_results": evaluation_result.model_dump(),
                        },
                        f,
                        indent=4,
                    )
            if os.path.exists(working_dir):
                os.unlink(working_dir)
                rich.print("[cyan]Local executor cleanup complete[/cyan]")
            return evaluation_result

    def get_evaluation_result(self) -> EvaluationResult:
        """Get the evaluation result for the given data point.

        Returns:
            EvaluationResult: The evaluation result.
        """
        evaluation_result = EvaluationResult(
            total_score=0, command_results=self.command_results
        )
        if self.command_results[5].exit_code == 0:
            evaluation_result.total_score += 1
            evaluation_result.docker_compose_up_success = True
        if self.command_results[9].exit_code == 0:
            evaluation_result.total_score += 1
            evaluation_result.cypress_tests_success = True
        if self.command_results[10].exit_code == 0:
            evaluation_result.total_score += 1
            evaluation_result.cargo_test_success = True
        return evaluation_result
