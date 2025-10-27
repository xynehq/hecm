import asyncio
import subprocess
from tempfile import TemporaryDirectory

import weave
from claude_agent_sdk import ClaudeAgentOptions, query

from hecm.dataset_generation.schemas import CodingAgentDataPoint
from hecm.eval_harness.evaluation.base import BaseEvaluator


class ClauseAgentEvaluator(BaseEvaluator):
    async def execute_claude_agent(
        self, data_point: CodingAgentDataPoint, repo_dir: str
    ) -> list:
        subprocess.run(
            f"git clone https://github.com/{data_point.repo}.git {repo_dir}",
            shell=True,
        )
        options = ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "findreplaceWrite", "Bash"],
            system_prompt="You are an experienced software engineer. You solve tasks by executing bash commands step-by-step. You explore the codebase, understand it, make minimal necessary changes, and validate your work.",
            permission_mode="acceptEdits",
            cwd=repo_dir,
            max_turns=10,
        )
        messages = []
        async for message in query(
            prompt=f"You must solve the following task:\n\n{data_point.problem_statement}",
            options=options,
        ):
            messages.append(message)
        return messages

    @weave.op
    def get_agent_response(self, data_point: CodingAgentDataPoint) -> str:
        with TemporaryDirectory() as temp_dir:
            messages = asyncio.run(self.execute_claude_agent(data_point, temp_dir))
            git_diff = subprocess.run(
                f"git diff", shell=True, capture_output=True
            ).stdout.decode("utf-8")
        return {
            "messages": messages,
            "patch": git_diff,
        }
