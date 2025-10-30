import logging
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from hecm.dataset_generation.schemas import CodingAgentDataPoint


def setup_logging(log_dir: Path, debug: bool = False):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if debug else logging.INFO

    # Main logger
    logger = logging.getLogger("ClaudeProxyEvaluator")
    logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging called multiple times
    if logger.handlers:
        return logger

    # File handler
    log_file = log_dir / f"evaluator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class ClaudeCodeProxyAgent:
    def __init__(
        self,
        proxy_repo_path: str | os.PathLike | None = None,
        proxy_repo_url: str = "https://github.com/fuergaosi233/claude-code-proxy",
        anthropic_base_url: str = "http://localhost:8082",
        anthropic_api_key: str = "dummy",
        openai_base_url: str = "http://127.0.0.1:8005/v1",
        openai_api_key: str = "dummy",
        openai_model: str = "archit11/Kwaipilot-KAT-Dev-Merged",
        big_model: str | None = None,
        small_model: str | None = None,
        middle_model: str | None = None,
        auto_clone: bool = True,
        proxy_startup_wait: int = 5,
        log_dir: str | os.PathLike | None = None,
        debug: bool = False,
    ):
        if proxy_repo_path is None:
            self.proxy_repo_path = Path(tempfile.gettempdir()) / "claude-code-proxy"
        else:
            self.proxy_repo_path = Path(proxy_repo_path)

        self.proxy_repo_url = proxy_repo_url
        self.anthropic_base_url = anthropic_base_url
        self.anthropic_api_key = anthropic_api_key
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.big_model = big_model or openai_model
        self.small_model = small_model or openai_model
        self.middle_model = middle_model or openai_model
        self.proxy_startup_wait = proxy_startup_wait
        self.debug = debug

        # Caches and logs
        self.repos_cache_dir = Path(tempfile.gettempdir()) / "claude_repos_cache"
        self.repos_cache_dir.mkdir(exist_ok=True)

        if log_dir is None:
            self.log_dir = Path(tempfile.gettempdir()) / "claude_evaluator_logs"
        else:
            self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(self.log_dir, debug)

        # Proxy process artifacts
        self.proxy_stdout_log = self.log_dir / "proxy_stdout.log"
        self.proxy_stderr_log = self.log_dir / "proxy_stderr.log"
        self.proxy_process = None
        self.proxy_stdout_file = None
        self.proxy_stderr_file = None

        if auto_clone and not self.proxy_repo_path.exists():
            self._clone_proxy_repo()

        self.logger.info("Initialized ClaudeCodeProxyAgent")

    def _clone_proxy_repo(self):
        self.logger.info(f"Cloning {self.proxy_repo_url} to {self.proxy_repo_path}")
        try:
            subprocess.run(
                ["git", "clone", self.proxy_repo_url, str(self.proxy_repo_path)],
                check=True,
                capture_output=True,
            )
            self.logger.info("Successfully cloned proxy repository")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to clone proxy repo: {e}")
            raise

    def _get_proxy_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "ANTHROPIC_BASE_URL": self.anthropic_base_url,
                "ANTHROPIC_API_KEY": self.anthropic_api_key,
                "OPENAI_BASE_URL": self.openai_base_url,
                "OPENAI_API_KEY": self.openai_api_key,
                "OPENAI_MODEL": self.openai_model,
                "BIG_MODEL": self.big_model,
                "SMALL_MODEL": self.small_model,
                "MIDDLE_MODEL": self.middle_model,
            }
        )
        return env

    def start_proxy(self):
        if self.proxy_process is not None:
            self.logger.warning("Proxy already running")
            return
        self.logger.info("Starting claude-code-proxy...")
        self.logger.info(f"Proxy stdout log: {self.proxy_stdout_log}")
        self.logger.info(f"Proxy stderr log: {self.proxy_stderr_log}")

        env = self._get_proxy_env()

        # Open log files
        self.proxy_stdout_file = open(self.proxy_stdout_log, "w")
        self.proxy_stderr_file = open(self.proxy_stderr_log, "w")

        try:
            self.proxy_process = subprocess.Popen(
                ["python3", "start_proxy.py"],
                cwd=str(self.proxy_repo_path),
                env=env,
                stdout=self.proxy_stdout_file,
                stderr=self.proxy_stderr_file,
                text=True,
            )

            time.sleep(self.proxy_startup_wait)

            if self.proxy_process.poll() is not None:
                self.logger.error("Proxy process died immediately after start")
                self.logger.error(f"Check logs: {self.proxy_stderr_log}")
                raise RuntimeError("Proxy failed to start")

            self.logger.info(
                f"Proxy started successfully (PID: {self.proxy_process.pid})"
            )
        except Exception as e:
            self.logger.error(f"Failed to start proxy: {e}")
            if self.proxy_stdout_file:
                self.proxy_stdout_file.close()
            if self.proxy_stderr_file:
                self.proxy_stderr_file.close()
            raise

    def stop_proxy(self):
        if self.proxy_process is None:
            return
        self.logger.info("Stopping claude-code-proxy...")
        self.proxy_process.terminate()
        try:
            self.proxy_process.wait(timeout=10)
            self.logger.info("Proxy stopped gracefully")
        except subprocess.TimeoutExpired:
            self.logger.warning("Proxy didn't stop gracefully, killing...")
            self.proxy_process.kill()
            self.proxy_process.wait()
            self.logger.info("Proxy killed")

        if self.proxy_stdout_file:
            self.proxy_stdout_file.close()
        if self.proxy_stderr_file:
            self.proxy_stderr_file.close()

        self.proxy_process = None

    def _get_cached_repo_path(self, repo: str) -> Path:
        safe_name = repo.replace("/", "_").replace("\\", "_")
        return self.repos_cache_dir / safe_name

    def _clone_or_update_repo(self, repo: str) -> Path:
        cached_path = self._get_cached_repo_path(repo)
        repo_url = f"https://github.com/{repo}.git"

        if cached_path.exists():
            self.logger.debug(f"Updating cached repo: {repo}")
            try:
                subprocess.run(
                    ["git", "fetch", "--all"],
                    cwd=str(cached_path),
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
                self.logger.debug(f"Successfully updated {repo}")
            except Exception as e:
                self.logger.warning(f"Failed to update repo {repo}: {e}")
        else:
            self.logger.info(f"Cloning repo: {repo} to temp cache")
            try:
                subprocess.run(
                    ["git", "clone", repo_url, str(cached_path)],
                    check=True,
                    capture_output=True,
                    timeout=300,
                )
                self.logger.info(f"Successfully cloned {repo}")
            except Exception as e:
                self.logger.error(f"Failed to clone {repo}: {e}")
                raise

        return cached_path

    def _setup_test_repo(self, data_point: CodingAgentDataPoint) -> Path:
        self.logger.debug(f"Setting up test repo for {data_point.instance_id}")
        cached_repo = self._clone_or_update_repo(data_point.repo)
        temp_dir = Path(
            tempfile.mkdtemp(prefix=f"claude_test_{data_point.instance_id}_")
        )
        self.logger.debug(f"Created temp dir: {temp_dir}")

        shutil.copytree(cached_repo, temp_dir, dirs_exist_ok=True)

        try:
            self.logger.debug(f"Checking out to {data_point.base_commit}")
            subprocess.run(
                ["git", "checkout", data_point.base_commit],
                cwd=str(temp_dir),
                check=True,
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["git", "reset", "--hard", "HEAD"],
                cwd=str(temp_dir),
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "clean", "-fdx"],
                cwd=str(temp_dir),
                check=True,
                capture_output=True,
            )
            self.logger.debug("Successfully set up test repo")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to checkout {data_point.base_commit}: {e}")
            raise

        return temp_dir

    def _run_claude_command(
        self, prompt: str, working_dir: Path, instance_id: str, timeout: int = 300
    ):
        env = self._get_proxy_env()
        claude_log_file = (
            self.log_dir
            / f"claude_{instance_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        self.logger.debug(f"Running claude for {instance_id}")
        self.logger.debug(f"Claude log file: {claude_log_file}")

        try:
            with open(claude_log_file, "w") as log_f:
                log_f.write("=" * 80 + "\n")
                log_f.write("PROMPT:\n")
                log_f.write("=" * 80 + "\n")
                log_f.write(prompt + "\n")
                log_f.write("=" * 80 + "\n\n")
                log_f.flush()
                prompt = f"### Plase change the files as instructed in the prompt. \n{prompt}"

                result = subprocess.run(
                    ["claude", "--allowedTools", "Bash", "Edit", "Write", "-p", prompt],
                    cwd=str(working_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                log_f.write("STDOUT:\n")
                log_f.write("=" * 80 + "\n")
                log_f.write(result.stdout + "\n")
                log_f.write("=" * 80 + "\n\n")

                log_f.write("STDERR:\n")
                log_f.write("=" * 80 + "\n")
                log_f.write(result.stderr + "\n")
                log_f.write("=" * 80 + "\n\n")

                log_f.write(f"EXIT CODE: {result.returncode}\n")

            self.logger.debug(
                f"Claude finished for {instance_id} with exit code {result.returncode}"
            )
            return result.stdout, result.stderr, result.returncode, str(claude_log_file)

        except subprocess.TimeoutExpired:
            self.logger.error(f"Claude command timed out for {instance_id}")
            return "", "Command timed out", -1, str(claude_log_file)
        except Exception as e:
            self.logger.error(f"Error running claude for {instance_id}: {e}")
            return "", str(e), -1, str(claude_log_file)

    def _get_git_diff(self, repo_path: Path) -> str:
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=True,
            )
            diff = result.stdout

            result_untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=True,
            )
            untracked_files = (
                result_untracked.stdout.strip().split("\n")
                if result_untracked.stdout.strip()
                else []
            )
            if untracked_files:
                for file in untracked_files:
                    if not file:
                        continue
                    diff += f"\n\n--- /dev/null\n+++ b/{file}\n"
                    try:
                        with open(repo_path / file, "r") as f:
                            content = f.read()
                            for line in content.split("\n"):
                                diff += f"+{line}\n"
                    except Exception:
                        pass
            logging.info("diff generated successfully {diff}")
            return diff
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get git diff: {e}")
            return ""

    def _get_changed_files(self, repo_path: Path) -> List[str]:
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=True,
            )
            changed = result.stdout.strip().split("\n") if result.stdout.strip() else []

            result_untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=True,
            )
            untracked = (
                result_untracked.stdout.strip().split("\n")
                if result_untracked.stdout.strip()
                else []
            )

            all_files = [f for f in changed + untracked if f]
            return all_files
        except subprocess.CalledProcessError:
            return []

    def get_agent_response(
        self,
        data_point: CodingAgentDataPoint,
        start_proxy: bool = False,
        stop_proxy: bool = False,
    ) -> str:
        start_time = datetime.now()
        test_repo = None
        try:
            if start_proxy:
                self.start_proxy()
        except Exception as e:
            self.logger.error(f"Error starting proxy: {e}")
            raise

            # Prepare repo copy at base commit
            test_repo = self._setup_test_repo(data_point)

            # Build directive prompt similar to original script
            directive_prompt = f"""{data_point.problem_statement}\nIMPORTANT: You must actually edit the files to fix this issue. Do not just explain what needs to be done.\nMake the necessary code changes directly."""

            stdout, stderr, exit_code, log_file = self._run_claude_command(
                prompt=directive_prompt,
                working_dir=test_repo,
                instance_id=data_point.instance_id,
                timeout=300,
            )

            claude_patch = self._get_git_diff(test_repo)
            files_changed = self._get_changed_files(test_repo)

            success = exit_code == 0 and bool(claude_patch)
            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "claude_patch": claude_patch,
                "files_changed": files_changed,
                "success": success,
                "execution_time": execution_time,
                "claude_log_file": log_file,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
            }
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(
                f"Error processing {data_point.instance_id}: {e}", exc_info=True
            )
            return {
                "claude_patch": "",
                "files_changed": [],
                "success": False,
                "execution_time": execution_time,
                "claude_log_file": None,
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
            }
        finally:
            if stop_proxy:
                self.stop_proxy()
            if test_repo and test_repo.exists():
                shutil.rmtree(test_repo, ignore_errors=True)
