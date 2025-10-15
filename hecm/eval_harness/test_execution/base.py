from abc import abstractmethod
from typing import Dict, List

import docker
import rich
from pydantic import BaseModel

from hecm.dataset_generation.schemas import CodingAgentDataPoint


class CommandExecutionResult(BaseModel):
    command: str
    exit_code: int
    output: str


class DataPointExecutionSummary(BaseModel):
    instance_id: str
    repo: str
    base_commit: str
    all_commands_executed_successfully: bool
    command_results: List[CommandExecutionResult]


class BaseSandboxedExecutor:
    """Executes commands/code in a sandboxed Docker container.

    Args:
        image (str): Docker image to use (default: rust:latest)
        working_dir (str): Working directory inside the container (default: /workspace)
        environment (Dict[str, str]): Environment variables to set in the container (default: None)
    """

    def __init__(
        self,
        image: str,
        working_dir: str = "/workspace",
        environment: Dict[str, str] = None,
    ):
        self.image = image
        self.working_dir = working_dir
        self.environment = environment

        self.client = docker.from_env()
        self.container = None
        self.create_container()

    def create_container(self):
        # Pull the image if not available
        rich.print(f"[cyan]Pulling Docker image: {self.image}[/cyan]")
        self.client.images.pull(self.image)

        # Create and start the container
        rich.print(f"[cyan]Creating container with image: {self.image}[/cyan]")
        self.container = self.client.containers.run(
            self.image,
            command="tail -f /dev/null",  # Keep container running
            detach=True,
            working_dir=self.working_dir,
            environment=self.environment or {},
            remove=False,  # Don't auto-remove so we can inspect
        )

    def execute_commands_in_container(
        self, commands: List[str]
    ) -> List[CommandExecutionResult]:
        """
        Execute a list of commands sequentially in a Docker container.

        Args:
            commands: List of shell commands to execute

        Returns (List[CommandExecutionResult]): List of `CommandExecutionResult`s
        """
        results: List[CommandExecutionResult] = []

        try:
            # Execute each command sequentially
            for i, command in enumerate(commands, 1):
                rich.print(
                    f"[yellow]Executing command {i}/{len(commands)}: {command}[/yellow]"
                )

                # Start execution with streaming enabled
                exec_id = self.client.api.exec_create(
                    self.container.id,
                    cmd=["sh", "-c", command],
                    workdir=self.working_dir,
                    environment=self.environment or {},
                )

                # Stream the output in real-time
                output_lines = []
                for chunk in self.client.api.exec_start(exec_id["Id"], stream=True):
                    decoded_chunk = chunk.decode("utf-8")
                    output_lines.append(decoded_chunk)
                    # Print output in real-time
                    print(decoded_chunk, end="")

                # Get the exit code after execution completes
                exec_inspect = self.client.api.exec_inspect(exec_id["Id"])
                exit_code = exec_inspect["ExitCode"]

                # Combine all output
                output = "".join(output_lines)

                results.append(
                    CommandExecutionResult(
                        command=command, exit_code=exit_code, output=output
                    )
                )

                if exit_code != 0:
                    rich.print(f"[red]Command failed with exit code {exit_code}[/red]")
                else:
                    rich.print(f"[green]Command succeeded[/green]")

        except Exception as e:
            rich.print(f"[red]Error during execution: {e}[/red]")
            raise

        return results

    @abstractmethod
    def get_commands(self, data_point: CodingAgentDataPoint) -> List[str]:
        pass

    def execute(self, data_point: CodingAgentDataPoint) -> DataPointExecutionSummary:
        """
        Execute tests for a given data point.

        Args:
            data_point: The coding agent data point containing test information

        Returns (DataPointExecutionSummary): Execution summary of the given data point.
        """
        rich.print(
            f"[bold cyan]Executing tests for: {data_point.instance_id}[/bold cyan]"
        )
        rich.print(f"[cyan]Repository: {data_point.repo}[/cyan]")
        rich.print(f"[cyan]Base commit: {data_point.base_commit}[/cyan]")

        # Get commands to execute
        commands = self.get_commands(data_point)

        # Execute commands
        results = self.execute_commands_in_container(commands=commands)

        # Analyze results
        all_succeeded = all(r.exit_code == 0 for r in results)

        return DataPointExecutionSummary(
            instance_id=data_point.instance_id,
            repo=data_point.repo,
            base_commit=data_point.base_commit,
            all_commands_executed_successfully=all_succeeded,
            command_results=results,
        )

    def cleanup(self):
        self.container.stop()
        self.container.remove()
        rich.print("[cyan]Container cleaned up[/cyan]")
