import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm.auto import tqdm

from hecm.dataset_generation.schemas import CodingAgentDataPoint

# -------------------------------
# Setup Logging
# -------------------------------


def setup_logging(log_dir: Path, debug: bool = False):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if debug else logging.INFO

    # Main logger
    logger = logging.getLogger("ClaudeProxyGenerator")
    logger.setLevel(log_level)

    # File handler
    log_file = log_dir / f"generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


# -------------------------------
# Schema for Claude's Attempt
# -------------------------------


@dataclass
class ClaudeAttemptDataPoint:
    """Stores both the gold patch and Claude's attempt."""

    instance_id: str
    repo: str
    problem_statement: str
    base_commit: str

    # Gold data (from dataset)
    gold_patch: str
    gold_test_patch: str
    hints_text: str
    test_instructions: str

    # Claude's attempt
    claude_patch: str
    claude_success: bool
    claude_stdout: str
    claude_stderr: str
    claude_execution_time: float
    claude_files_changed: List[str]

    # Metadata
    created_at: str
    model_config: Dict[str, str]
    error: Optional[str] = None

    # Log files
    claude_log_file: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class ClaudeAttemptDataset:
    data_points: List[ClaudeAttemptDataPoint]
    dataset_metadata: Dict[str, Any]

    def to_dict(self):
        return {
            "data_points": [dp.to_dict() for dp in self.data_points],
            "metadata": self.dataset_metadata,
        }

    def save(self, filepath: os.PathLike):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: os.PathLike):
        with open(filepath, "r") as f:
            data = json.load(f)

        data_points = [ClaudeAttemptDataPoint(**dp) for dp in data["data_points"]]
        return cls(data_points=data_points, dataset_metadata=data["metadata"])


# -------------------------------
# Claude Proxy Patch Generator from Dataset
# -------------------------------


class ClaudeProxyDatasetGenerator:
    def __init__(
        self,
        proxy_repo_path: Optional[str] = None,
        proxy_repo_url: str = "https://github.com/fuergaosi233/claude-code-proxy",
        anthropic_base_url: str = "http://localhost:8082",
        anthropic_api_key: str = "dummy",
        openai_base_url: str = "http://127.0.0.1:8005/v1",
        openai_api_key: str = "dummy",
        openai_model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        big_model: Optional[str] = None,
        small_model: Optional[str] = None,
        middle_model: Optional[str] = None,
        auto_clone: bool = True,
        proxy_startup_wait: int = 5,
        log_dir: Optional[str] = None,
        debug: bool = False,
    ):
        # Use system temp directory for proxy
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

        # Use system temp directory for repos cache
        self.repos_cache_dir = Path(tempfile.gettempdir()) / "claude_repos_cache"
        self.repos_cache_dir.mkdir(exist_ok=True)

        # Setup log directory
        if log_dir is None:
            self.log_dir = Path(tempfile.gettempdir()) / "claude_generator_logs"
        else:
            self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(self.log_dir, debug)

        # Proxy log files
        self.proxy_stdout_log = self.log_dir / "proxy_stdout.log"
        self.proxy_stderr_log = self.log_dir / "proxy_stderr.log"

        self.proxy_process = None
        self.proxy_stdout_file = None
        self.proxy_stderr_file = None

        if auto_clone and not self.proxy_repo_path.exists():
            self._clone_proxy_repo()

        self.logger.info("Initialized ClaudeProxyDatasetGenerator")
        self.logger.info(f"Proxy repo: {self.proxy_repo_path}")
        self.logger.info(f"Repos cache: {self.repos_cache_dir}")
        self.logger.info(f"Log directory: {self.log_dir}")

    def _clone_proxy_repo(self):
        """Clone the claude-code-proxy repository to temp."""
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
        """Get environment variables for the proxy."""
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
        """Start the claude-code-proxy server with isolated logging."""
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

            # Wait for proxy to start
            time.sleep(self.proxy_startup_wait)

            # Check if proxy is still running
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
        """Stop the claude-code-proxy server."""
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

        # Close log files
        if self.proxy_stdout_file:
            self.proxy_stdout_file.close()
        if self.proxy_stderr_file:
            self.proxy_stderr_file.close()

        self.proxy_process = None

    def _get_cached_repo_path(self, repo: str) -> Path:
        """Get the cached repository path in temp directory."""
        safe_name = repo.replace("/", "_").replace("\\", "_")
        return self.repos_cache_dir / safe_name

    def _clone_or_update_repo(self, repo: str) -> Path:
        """Clone the repository or update if it exists in temp cache."""
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
        """Setup a test repository from the data point in a temp directory."""
        self.logger.debug(f"Setting up test repo for {data_point.instance_id}")

        # Clone or update the cached repo in temp
        cached_repo = self._clone_or_update_repo(data_point.repo)

        # Create a temporary copy for this test (also in temp)
        temp_dir = Path(
            tempfile.mkdtemp(prefix=f"claude_test_{data_point.instance_id}_")
        )
        self.logger.debug(f"Created temp dir: {temp_dir}")

        # Copy the repo
        shutil.copytree(cached_repo, temp_dir, dirs_exist_ok=True)

        # Checkout to base commit
        try:
            self.logger.debug(f"Checking out to {data_point.base_commit}")
            subprocess.run(
                ["git", "checkout", data_point.base_commit],
                cwd=str(temp_dir),
                check=True,
                capture_output=True,
                timeout=30,
            )

            # Create a clean working state
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
        self,
        prompt: str,
        working_dir: Path,
        instance_id: str,
        timeout: int = 300,
    ) -> tuple[str, str, int, str]:
        """Run claude CLI command with a prompt and isolated logging."""
        env = self._get_proxy_env()

        # Create log file for this specific run
        claude_log_file = (
            self.log_dir
            / f"claude_{instance_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        self.logger.debug(f"Running claude for {instance_id}")
        self.logger.debug(f"Claude log file: {claude_log_file}")

        try:
            with open(claude_log_file, "w") as log_f:
                # Write the prompt to log
                log_f.write("=" * 80 + "\n")
                log_f.write("PROMPT:\n")
                log_f.write("=" * 80 + "\n")
                log_f.write(prompt + "\n")
                log_f.write("=" * 80 + "\n\n")
                log_f.flush()

                # Run claude
                result = subprocess.run(
                    ["claude", "--allowedTools", "Bash", "Edit", "Write", "-p", prompt],
                    cwd=str(working_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                # Write output to log
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
        """Get the git diff of all changes in the repository."""
        try:
            # Get diff of all changes (staged and unstaged)
            result = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=True,
            )
            diff = result.stdout

            # Also get untracked files
            result_untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=True,
            )

            untracked_files = result_untracked.stdout.strip().split("\n")
            if untracked_files and untracked_files[0]:
                for file in untracked_files:
                    if file:
                        diff += f"\n\n--- /dev/null\n+++ b/{file}\n"
                        try:
                            with open(repo_path / file, "r") as f:
                                content = f.read()
                                for line in content.split("\n"):
                                    diff += f"+{line}\n"
                        except:
                            pass

            return diff
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get git diff: {e}")
            return ""

    def _get_changed_files(self, repo_path: Path) -> List[str]:
        """Get list of changed files."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=True,
            )
            changed = result.stdout.strip().split("\n") if result.stdout.strip() else []

            # Add untracked files
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

    def _generate_claude_attempt(
        self,
        data_point: CodingAgentDataPoint,
        timeout: int = 300,
    ) -> Optional[ClaudeAttemptDataPoint]:
        """Generate Claude's attempt for a single data point."""
        start_time = datetime.now()
        test_repo = None

        self.logger.info(f"Processing {data_point.instance_id}")

        try:
            # Setup test repository at base commit (in temp)
            test_repo = self._setup_test_repo(data_point)

            # Run claude command with the problem statement
            directive_prompt = f"""{data_point.problem_statement}
            IMPORTANT: You must actually edit the files to fix this issue. Do not just explain what needs to be done.
            Make the necessary code changes directly."""
            stdout, stderr, exit_code, log_file = self._run_claude_command(
                prompt=directive_prompt,
                working_dir=test_repo,
                instance_id=data_point.instance_id,
                timeout=timeout,
            )

            # Extract Claude's patch
            claude_patch = self._get_git_diff(test_repo)
            print("Claude Patch:\n", claude_patch)
            self.logger.info(
                f"Claude Patch for {data_point.instance_id}:\n{claude_patch}"
            )
            files_changed = self._get_changed_files(test_repo)
            self.logger.info(
                f"Files changed for {data_point.instance_id}: {files_changed}"
            )
            print("Files Changed:\n", files_changed)

            success = exit_code == 0 and bool(claude_patch)
            execution_time = (datetime.now() - start_time).total_seconds()

            self.logger.info(
                f"Completed {data_point.instance_id}: "
                f"success={success}, time={execution_time:.2f}s, "
                f"files_changed={len(files_changed)}"
            )

            return ClaudeAttemptDataPoint(
                instance_id=data_point.instance_id,
                repo=data_point.repo,
                problem_statement=data_point.problem_statement,
                base_commit=data_point.base_commit,
                gold_patch=data_point.patch,
                gold_test_patch=data_point.test_patch,
                hints_text=data_point.hints_text,
                test_instructions=data_point.test_instructions,
                claude_patch=claude_patch,
                claude_success=success,
                claude_stdout=stdout,
                claude_stderr=stderr,
                claude_execution_time=execution_time,
                claude_files_changed=files_changed,
                created_at=start_time.isoformat(),
                model_config={
                    "openai_model": self.openai_model,
                    "big_model": self.big_model,
                    "small_model": self.small_model,
                    "middle_model": self.middle_model,
                },
                error=None if success else stderr,
                claude_log_file=log_file,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(
                f"Error processing {data_point.instance_id}: {e}", exc_info=True
            )
            return ClaudeAttemptDataPoint(
                instance_id=data_point.instance_id,
                repo=data_point.repo,
                problem_statement=data_point.problem_statement,
                base_commit=data_point.base_commit,
                gold_patch=data_point.patch,
                gold_test_patch=data_point.test_patch,
                hints_text=data_point.hints_text,
                test_instructions=data_point.test_instructions,
                claude_patch="",
                claude_success=False,
                claude_stdout="",
                claude_stderr=str(e),
                claude_execution_time=execution_time,
                claude_files_changed=[],
                created_at=start_time.isoformat(),
                model_config={
                    "openai_model": self.openai_model,
                    "big_model": self.big_model,
                    "small_model": self.small_model,
                    "middle_model": self.middle_model,
                },
                error=str(e),
                claude_log_file=None,
            )

        finally:
            # Cleanup test repo (it's in temp, but still good to clean up)
            if test_repo and test_repo.exists():
                shutil.rmtree(test_repo, ignore_errors=True)

    def generate_from_hf_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        output_name: str = "claude_attempts",
        save_to: Optional[os.PathLike] = None,
        timeout_per_task: int = 300,
        max_workers: int = 1,
        start_proxy: bool = True,
    ) -> ClaudeAttemptDataset:
        """
        Generate Claude attempts from a HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "juspay/hyperswitch")
            split: Dataset split to use
            max_samples: Maximum number of samples to process
            output_name: Name for the output dataset
            save_to: Optional filepath to save the dataset
            timeout_per_task: Timeout for each Claude task in seconds
            max_workers: Number of parallel workers (default 1 for safety)
            start_proxy: Whether to start/stop the proxy automatically

        Returns:
            ClaudeAttemptDataset with gold patches and Claude's attempts
        """
        self.logger.info(f"Loading dataset: {dataset_name} (split: {split})")
        dataset = load_dataset(dataset_name, split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.logger.info(f"Processing {len(dataset)} samples")

        # Convert to CodingAgentDataPoint objects
        data_points = []
        for idx, sample in enumerate(dataset):
            try:
                data_point = CodingAgentDataPoint.model_validate(sample)
                data_points.append(data_point)
            except Exception as e:
                self.logger.warning(f"Failed to validate sample {idx}: {e}")

        self.logger.info(f"Successfully loaded {len(data_points)} data points")

        if start_proxy:
            self.start_proxy()

        try:
            claude_attempts: List[ClaudeAttemptDataPoint] = []

            if max_workers == 1:
                # Sequential processing
                for data_point in tqdm(data_points, desc="Generating Claude attempts"):
                    attempt = self._generate_claude_attempt(
                        data_point, timeout_per_task
                    )
                    if attempt:
                        claude_attempts.append(attempt)
            else:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._generate_claude_attempt, data_point, timeout_per_task
                        ): data_point.instance_id
                        for data_point in data_points
                    }

                    for future in tqdm(
                        as_completed(futures),
                        desc="Generating Claude attempts",
                        total=len(data_points),
                    ):
                        attempt = future.result()
                        if attempt:
                            claude_attempts.append(attempt)

            result_dataset = ClaudeAttemptDataset(
                data_points=claude_attempts,
                dataset_metadata={
                    "name": output_name,
                    "source_dataset": dataset_name,
                    "source_split": split,
                    "model_config": {
                        "openai_model": self.openai_model,
                        "big_model": self.big_model,
                        "small_model": self.small_model,
                        "middle_model": self.middle_model,
                    },
                    "num_samples": len(claude_attempts),
                    "num_successful": sum(
                        1 for a in claude_attempts if a.claude_success
                    ),
                    "num_failed": sum(
                        1 for a in claude_attempts if not a.claude_success
                    ),
                    "created_at": datetime.now().isoformat(),
                    "proxy_config": {
                        "anthropic_base_url": self.anthropic_base_url,
                        "openai_base_url": self.openai_base_url,
                    },
                    "temp_directories": {
                        "proxy_repo": str(self.proxy_repo_path),
                        "repos_cache": str(self.repos_cache_dir),
                        "log_dir": str(self.log_dir),
                    },
                },
            )

            if save_to:
                result_dataset.save(save_to)
                self.logger.info(f"Dataset saved to {save_to}")

            self.logger.info(f"Generation complete: {len(claude_attempts)} attempts")
            self.logger.info(
                f"Successful: {result_dataset.dataset_metadata['num_successful']}"
            )
            self.logger.info(f"Failed: {result_dataset.dataset_metadata['num_failed']}")

            return result_dataset

        finally:
            if start_proxy:
                self.stop_proxy()

    def cleanup_temp_directories(self):
        """Manually cleanup temp directories if needed."""
        self.logger.info("Cleaning up temp directories...")

        if self.repos_cache_dir.exists():
            shutil.rmtree(self.repos_cache_dir, ignore_errors=True)
            self.logger.info(f"Removed: {self.repos_cache_dir}")

        if self.proxy_repo_path.exists() and self.proxy_repo_path.parent == Path(
            tempfile.gettempdir()
        ):
            shutil.rmtree(self.proxy_repo_path, ignore_errors=True)
            self.logger.info(f"Removed: {self.proxy_repo_path}")


# -------------------------------
# Example Usage
# -------------------------------


def main():
    """Example usage."""

    # Initialize the generator with custom log directory and debug mode
    generator = ClaudeProxyDatasetGenerator(
        anthropic_base_url="http://localhost:8082",
        anthropic_api_key="dummy",
        openai_base_url="http://127.0.0.1:8005/v1",
        openai_api_key="dummy",
        openai_model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        auto_clone=True,
        log_dir="./logs",  # Custom log directory (not in temp)
        debug=True,  # Enable debug logging
    )

    try:
        # Generate from HuggingFace dataset
        dataset = generator.generate_from_hf_dataset(
            dataset_name="juspay/hyperswitch",
            split="train",
            max_samples=2,  # Start with just 2 samples
            output_name="hyperswitch_claude_attempts",
            save_to="claude_attempts.json",
            timeout_per_task=300,
            max_workers=1,
            start_proxy=True,
        )

        # Print results
        print(f"\n{'=' * 60}")
        print(f"Results Summary")
        print(f"{'=' * 60}")
        print(f"Total samples: {len(dataset.data_points)}")
        print(f"Successful: {dataset.dataset_metadata['num_successful']}")
        print(f"Failed: {dataset.dataset_metadata['num_failed']}")
        print(f"\nLogs saved to: {generator.log_dir}")
        print(f"  - Proxy stdout: {generator.proxy_stdout_log}")
        print(f"  - Proxy stderr: {generator.proxy_stderr_log}")

        # Show first result
        if dataset.data_points:
            first = dataset.data_points[0]
            print(f"\n{'=' * 60}")
            print(f"First Data Point: {first.instance_id}")
            print(f"{'=' * 60}")
            print(f"Repo: {first.repo}")
            print(f"Base Commit: {first.base_commit}")
            print(f"Success: {first.claude_success}")
            print(f"Execution Time: {first.claude_execution_time:.2f}s")
            print(f"Files Changed: {first.claude_files_changed}")
            print(f"Claude Log: {first.claude_log_file}")
            print(f"\nGold Patch Length: {len(first.gold_patch)} chars")
            print(f"Claude Patch Length: {len(first.claude_patch)} chars")

    finally:
        # Optional: Clean up temp directories when done
        # generator.cleanup_temp_directories()
        pass


if __name__ == "__main__":
    main()
