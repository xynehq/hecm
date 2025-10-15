from typing import Any, Dict, List

import docker
import rich

from hecm.dataset_generation.schemas import CodingAgentDataPoint


class JuspayHyperswitchTestExecutor:
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
        image: Docker image to use (default: rust:latest)
    """

    def __init__(self, image: str = "rust:latest"):
        self.image = image
        self.client = docker.from_env()

    def execute_commands_in_container(
        self,
        commands: List[str],
        working_dir: str = "/workspace",
        environment: Dict[str, str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a list of commands sequentially in a Docker container.

        Args:
            commands: List of shell commands to execute
            working_dir: Working directory inside the container
            environment: Environment variables to set in the container

        Returns:
            List of dictionaries containing command results with keys:
            - command: The command that was executed
            - exit_code: Exit code of the command
            - output: Combined stdout/stderr output
        """
        results = []
        container = None

        try:
            # Pull the image if not available
            rich.print(f"[cyan]Pulling Docker image: {self.image}[/cyan]")
            self.client.images.pull(self.image)

            # Create and start the container
            rich.print(f"[cyan]Creating container with image: {self.image}[/cyan]")
            container = self.client.containers.run(
                self.image,
                command="tail -f /dev/null",  # Keep container running
                detach=True,
                working_dir=working_dir,
                environment=environment or {},
                remove=False,  # Don't auto-remove so we can inspect
            )

            # Execute each command sequentially
            for i, command in enumerate(commands, 1):
                rich.print(
                    f"[yellow]Executing command {i}/{len(commands)}: {command}[/yellow]"
                )

                exec_result = container.exec_run(
                    cmd=["sh", "-c", command],
                    workdir=working_dir,
                    environment=environment or {},
                    demux=False,  # Combine stdout and stderr
                )

                output = (
                    exec_result.output.decode("utf-8") if exec_result.output else ""
                )
                exit_code = exec_result.exit_code

                result = {
                    "command": command,
                    "exit_code": exit_code,
                    "output": output,
                }
                results.append(result)

                # Print output
                if output:
                    rich.print(f"[dim]{output}[/dim]")

                if exit_code != 0:
                    rich.print(f"[red]Command failed with exit code {exit_code}[/red]")
                else:
                    rich.print(f"[green]Command succeeded[/green]")

        except Exception as e:
            rich.print(f"[red]Error during execution: {e}[/red]")
            raise
        finally:
            # Clean up container
            if container:
                try:
                    container.stop()
                    container.remove()
                    rich.print("[cyan]Container cleaned up[/cyan]")
                except Exception as e:
                    rich.print(
                        f"[yellow]Warning: Failed to clean up container: {e}[/yellow]"
                    )

        return results

    def execute(self, data_point: CodingAgentDataPoint) -> Dict[str, Any]:
        """
        Execute tests for a given data point.

        Args:
            data_point: The coding agent data point containing test information

        Returns:
            Dictionary containing test execution results
        """
        rich.print(
            f"[bold cyan]Executing tests for: {data_point.instance_id}[/bold cyan]"
        )
        rich.print(f"[cyan]Repository: {data_point.repo}[/cyan]")
        rich.print(f"[cyan]Base commit: {data_point.base_commit}[/cyan]")

        # Prepare commands for test execution
        commands = [
            # Clone the repository
            f"git clone https://github.com/{data_point.repo}.git /workspace/repo",
            # Change to repo directory
            "cd /workspace/repo",
            # Checkout the base commit
            f"cd /workspace/repo && git checkout {data_point.base_commit}",
        ]

        # Execute commands
        results = self.execute_commands_in_container(
            commands=commands,
            working_dir="/workspace",
        )

        # Analyze results
        all_succeeded = all(r["exit_code"] == 0 for r in results)

        return {
            "instance_id": data_point.instance_id,
            "repo": data_point.repo,
            "base_commit": data_point.base_commit,
            "all_tests_passed": all_succeeded,
            "command_results": results,
        }
