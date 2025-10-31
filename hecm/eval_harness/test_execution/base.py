import os
import subprocess
from abc import ABC, abstractmethod

import rich
from pydantic import BaseModel

from hecm.dataset_generation.schemas import CodingAgentDataPoint


class CommandExecutionResult(BaseModel):
    """
    Result of executing a command.

    Attributes:
        command (str): The command that was executed.
        stdout (str): The stdout of the command.
        stderror (str): The stderr of the command.
        exit_code (int): The exit code of the command.
    """

    command: str
    stdout: str
    stderror: str
    exit_code: int


class EvaluationResult(BaseModel):
    """
    Result of evaluating a data point.

    Attributes:
        total_score (int): The total score of the data point.
        command_results (list[CommandExecutionResult]): The results of executing the commands.
        docker_compose_up_success (bool): Whether the docker compose up was successful.
        cypress_tests_success (bool): Whether the cypress tests were successful.
        cargo_test_success (bool): Whether the cargo test was successful.
    """

    total_score: int
    command_results: list[CommandExecutionResult]
    docker_compose_up_success: bool = False
    cypress_tests_success: bool = False
    cargo_test_success: bool = False


def execute_multiple_commands(
    commands: list[str], environment: dict[str, str]
) -> list[CommandExecutionResult]:
    """Execute multiple commands sequentially.

    Args:
        commands (list[str]): The commands to execute.
        environment (dict[str, str]): The environment variables to use for the commands.

    Returns:
        list[CommandExecutionResult]: The results of executing the commands.

    Raises:
        Exception: If the commands cannot be executed.
    """
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
        rich.print(f"[cyan]Command: {command}[/cyan]")
        if result.stdout:
            rich.print(f"[blue]Stdout:\n{result.stdout}[/blue]")
        if result.stderr:
            rich.print(f"[red]Stderr:\n{result.stderr}[/red]")
        rich.print(f"[magenta]Exit code: {result.returncode}[/magenta]")
        command_results.append(
            CommandExecutionResult(
                command=command,
                stdout=result.stdout,
                stderror=result.stderr,
                exit_code=result.returncode,
            )
        )
    return command_results


class BaseLocalExecutor(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command_results: list[CommandExecutionResult] = []

    @abstractmethod
    def execute(
        self,
        data_point: CodingAgentDataPoint,
        predicted_patch: str | None = None,
        result_save_path: os.PathLike | None = None,
    ) -> EvaluationResult:
        raise NotImplementedError(
            "Subclasses to `BaseLocalExecutor` must implement the `execute` method."
        )

    @abstractmethod
    def get_evaluation_result(self) -> EvaluationResult:
        raise NotImplementedError(
            "Subclasses to `BaseLocalExecutor` must implement the `get_evaluation_result` method."
        )
