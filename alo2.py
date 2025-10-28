from pathlib import Path
from tempfile import TemporaryDirectory

import litellm
import weave
from datasets import load_dataset
from dotenv import load_dotenv
from git import Repo
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

from hecm.dataset_generation.schemas import CodingAgentDataPoint

litellm._turn_on_debug()

INSTANCE_TEMPLATE = """\
You are solving the following task:

{{task}}

You are currently in directory: {{cwd}}

Rules:
- Reply with exactly ONE bash code block containing exactly ONE shell command (you may chain multiple commands with &&).
- The shell command will be executed and you'll see the output.
- Use bash commands to explore the repository, understand the codebase, make changes, and validate them.
- Do NOT require interactive input in your commands.
- You can read files with cat/head/tail, search with grep/find, edit with sed/awk/perl, etc.
- Run linters/tests to validate your changes as needed.

When you are COMPLETELY DONE with the task and have validated your changes:
1. Stage all your edits: `git add -A >/dev/null 2>&1`
2. Print this exact marker: `printf "MINI_SWE_AGENT_FINAL_OUTPUT\\n"`
3. Show the diff: `git -c core.safecrlf=false diff --no-color --staged`

Chain them together like this:
git add -A >/dev/null 2>&1 && printf "MINI_SWE_AGENT_FINAL_OUTPUT\\n" && git -c core.safecrlf=false diff --no-color --staged
"""

SYSTEM_TEMPLATE = "You are a senior software engineer. You solve tasks by executing bash commands step-by-step. You explore the codebase, understand it, make minimal necessary changes, and validate your work."


@weave.op
def run_mini_and_get_patch(repo_root: str | Path, problem: str) -> str:
    repo_root = str(Path(repo_root).resolve())
    model = LitellmModel(model_name="openai/gpt-4o", model_kwargs={"temperature": 0})
    env = LocalEnvironment(cwd=repo_root, timeout=300)
    agent = DefaultAgent(
        model,
        env,
        system_template=SYSTEM_TEMPLATE,
        instance_template=INSTANCE_TEMPLATE,
        cost_limit=5.0,
        step_limit=50,  # Allow up to 50 steps for the agent to explore and make changes
    )
    status, final = agent.run(
        task=problem  # Just pass the problem statement, the template handles the rest
    )
    if status != "Submitted":
        raise RuntimeError(
            f"Agent did not submit a solution (status={status}). Output:\n{final}"
        )
    patch = final.strip()
    return patch


@weave.op
def test_agent():
    dataset = load_dataset("juspay/hyperswitch", split="train")
    data_point = CodingAgentDataPoint.model_validate(dataset[2])
    with TemporaryDirectory() as temp_dir:
        repo = Repo.clone_from(
            url=f"https://github.com/juspay/hyperswitch", to_path=temp_dir
        )
        repo.git.checkout(data_point.base_commit)
        patch = run_mini_and_get_patch(temp_dir, data_point.problem_statement)
    return patch


if __name__ == "__main__":
    load_dotenv()
    weave.init(project_name="hyperswitch")
    test_agent()
