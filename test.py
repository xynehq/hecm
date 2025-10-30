import os
import subprocess
import time
from tempfile import TemporaryDirectory

import rich
from datasets import load_dataset

from hecm.dataset_generation.schemas import CodingAgentDataPoint
from hecm.eval_harness.agent.claude_code_agent import ClaudeCodeProxyAgent


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
        self.command_results: list[dict[str, str]] = []

    def clone_repository(self, data_point: CodingAgentDataPoint, repo_dir: str):
        for command in [
            # Clone the repository
            f"git clone https://github.com/{data_point.repo}.git {repo_dir}",
            # Checkout the base commit
            f"cd {repo_dir} && git switch --detach {data_point.base_commit}",
        ]:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                env=self.environment,
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
                command,
                shell=True,
                capture_output=True,
                text=True,
                env=self.environment,
            )
            self.command_results.append(
                {
                    "command": command,
                    "stdout": result.stdout,
                    "stderror": result.stderr,
                    "exit_code": result.returncode,
                }
            )

    def docker_compose_up(self, repo_dir: str):
        result = subprocess.run(
            f"cd {repo_dir} && docker compose --file docker-compose-development.yml up -d",
            shell=True,
            capture_output=True,
            text=True,
            env=self.environment,
        )
        self.command_results.append(
            {
                "command": f"cd {repo_dir} && docker compose --file docker-compose-development.yml up -d",
                "stdout": result.stdout,
                "stderror": result.stderr,
                "exit_code": result.returncode,
            }
        )

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
        result = subprocess.run(
            f"cd {repo_dir} && docker compose --file docker-compose-development.yml down",
            shell=True,
            capture_output=True,
            text=True,
            env=self.environment,
        )
        self.command_results.append(
            {
                "command": f"cd {repo_dir} && docker compose --file docker-compose-development.yml down",
                "stdout": result.stdout,
                "stderror": result.stderr,
                "exit_code": result.returncode,
            }
        )

    def execute_cypress_tests(self, repo_dir: str):
        test_dir = os.path.join(repo_dir, "cypress-tests-v2")
        for command in [
            # install nodejs and npm
            "sudo apt-get update",
            "sudo apt-get install -y nodejs npm",
            # install cypress dependencies
            f"cd {test_dir} && npm install",
            # run all tests in a headless manner
            f"cd {test_dir} && npx cypress run",
        ]:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                env=self.environment,
            )
            self.command_results.append(
                {
                    "command": command,
                    "stdout": result.stdout,
                    "stderror": result.stderr,
                    "exit_code": result.returncode,
                }
            )

    def execute_cargo_test(self, repo_dir: str):
        for command in [
            f"cd {repo_dir} && cargo test",
        ]:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                env=self.environment,
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
        self.apply_patch(data_point, repo_dir, predicted_patch=predicted_patch)
        self.docker_compose_up(repo_dir)
        self.execute_cypress_tests(repo_dir)
        self.execute_cargo_test(repo_dir)
        self.docker_compose_down(repo_dir)

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
agent = ClaudeCodeProxyAgent()
response = agent.get_agent_response(data_point)
rich.print(response)
# test_executor = JuspayHyperswitchLocalTestExecutor(
#     environment={
#         "CYPRESS_CONNECTOR": "connector_id",
#         "CYPRESS_BASEURL": "http://localhost:8080",
#         "DEBUG": "cypress:cli",
#         "CYPRESS_ADMINAPIKEY": "admin_api_key",
#         "CYPRESS_CONNECTOR_AUTH_FILE_PATH": "/Users/geekyrakshit/Workspace/athena/hecm/creds.json",
#     },
# )
# test_executor.execute(data_point, predicted_patch=response["claude_patch"])
