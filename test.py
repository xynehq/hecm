import os
import subprocess
from tempfile import TemporaryDirectory

import rich
from datasets import load_dataset

from hecm.dataset_generation.schemas import CodingAgentDataPoint


class JuspayHyperswitchLocalTestExecutor:
    def __init__(
        self,
        environment: dict[str, str] = None,
        cypress_test_suffix: str = ":payments",
        health_check_url: str = "http://localhost:8080/health",
        health_check_timeout: int = 3000,
        health_check_interval: int = 5,
    ):
        self.environment = environment
        self.cypress_test_suffix = cypress_test_suffix
        self.health_check_url = health_check_url
        self.health_check_timeout = health_check_timeout
        self.health_check_interval = health_check_interval
        self.command_results: list[dict[str, str]] = []

    def clone_repository(self, data_point: CodingAgentDataPoint, repo_dir: str):
        for command in [
            # Clone the repository
            f"git clone https://github.com/{data_point.repo}.git {repo_dir}",
            # Checkout the base commit
            f"cd {repo_dir} && git switch --detach {data_point.base_commit}",
        ]:
            result = subprocess.run(
                command, shell=True, check=True, env=self.environment
            )
            self.command_results.append(
                {
                    "command": command,
                    "stdout": result.stdout,
                    "stderror": result.stderr,
                    "exit_code": result.returncode,
                }
            )

    def apply_patch(
        self,
        data_point: CodingAgentDataPoint,
        repo_dir: str,
        predicted_patch: str | None = None,
    ):
        patch = data_point.patch if predicted_patch is None else predicted_patch
        for command in [
            f"cat > /tmp/change.patch <<'PATCH'\n{patch}\nPATCH",
            f"cd {repo_dir} && git stash && git apply /tmp/change.patch",
            f"rm /tmp/change.patch",
        ]:
            result = subprocess.run(
                command, shell=True, check=True, env=self.environment
            )
            self.command_results.append(
                {
                    "command": command,
                    "stdout": result.stdout,
                    "stderror": result.stderr,
                    "exit_code": result.returncode,
                }
            )

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
        self.apply_patch(data_point, repo_dir)

    def execute(
        self, data_point: CodingAgentDataPoint, predicted_patch: str | None = None
    ):
        working_dir = None
        try:
            with TemporaryDirectory(prefix="hyperswitch-testcase-") as working_dir:
                repo_dir = os.path.join(working_dir, "workspace")
                self.execute_commands(data_point, repo_dir, predicted_patch=None)
        except Exception as e:
            rich.print(f"[red]Error executing commands: {e}[/red]")
            raise
        finally:
            if os.path.exists(working_dir):
                os.unlink(working_dir)
                rich.print("[cyan]Local executor cleanup complete[/cyan]")


dataset = load_dataset("juspay/hyperswitch", split="train")
data_point = CodingAgentDataPoint.model_validate(dataset[0])
test_executor = JuspayHyperswitchLocalTestExecutor(
    environment={
        "CYPRESS_CONNECTOR": "connector_id",
        "CYPRESS_BASEURL": "base_url",
        "DEBUG": "cypress:cli",
        "CYPRESS_ADMINAPIKEY": "admin_api_key",
        "CYPRESS_CONNECTOR_AUTH_FILE_PATH": "/Users/geekyrakshit/Workspace/athena/hecm/creds.json",
    },
)
test_executor.execute(data_point)
