import json
import os
import subprocess
import time
from tempfile import TemporaryDirectory

import rich
from pydantic import BaseModel

from hecm.dataset_generation.schemas import CodingAgentDataPoint


class CommandExecutionResult(BaseModel):
    command: str
    stdout: str
    stderror: str
    exit_code: int


class EvaluationResult(BaseModel):
    total_score: int
    command_results: list[CommandExecutionResult]
    docker_compose_up_success: bool = False
    cypress_tests_success: bool = False
    cargo_test_success: bool = False


def execute_multiple_commands(
    commands: list[str], environment: dict[str, str]
) -> list[CommandExecutionResult]:
    command_results: list[CommandExecutionResult] = []
    for command in commands:
        rich.print(f"[yellow]Executing command: {command}[/yellow]")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        rich.print(f"[green]Command executed successfully![/green]")
        command_results.append(
            CommandExecutionResult(
                command=command,
                stdout=result.stdout,
                stderror=result.stderr,
                exit_code=result.returncode,
            )
        )
    return command_results


class JuspayHyperswitchLocalTestExecutor:
    def __init__(
        self,
        environment: dict[str, str] = None,
        cypress_test_suffix: str = ":payments",
        health_check_url: str = "http://localhost:8080/health",
        health_check_timeout: int = 720,
        health_check_interval: int = 5,
    ):
        self.environment = environment
        self.cypress_test_suffix = cypress_test_suffix
        self.health_check_url = health_check_url
        self.health_check_timeout = health_check_timeout
        self.health_check_interval = health_check_interval
        self.command_results: list[CommandExecutionResult] = []

    def clone_repository(self, data_point: CodingAgentDataPoint, repo_dir: str):
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
        command_results = execute_multiple_commands(
            [
                f"cd {repo_dir} && docker compose --file docker-compose-development.yml up -d"
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
        result = execute_multiple_commands(
            [
                f"cd {repo_dir} && docker compose --file docker-compose-development.yml down"
            ],
            self.environment,
        )
        self.command_results.extend(result)

    def execute_cypress_tests(self, repo_dir: str):
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
