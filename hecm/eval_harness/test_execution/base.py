import os
import subprocess
from abc import ABC, abstractmethod

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
