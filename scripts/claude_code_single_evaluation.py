from typing import Any, Dict, List

import rich
import weave
from datasets import load_dataset

from hecm.dataset_generation.schemas import CodingAgentDataPoint
from hecm.eval_harness.evaluation.claude_code_evaluator import ClaudeProxyEvaluator
from hecm.eval_harness.test_execution.base import BaseLocalExecutor


class MinimalLocalExecutor(BaseLocalExecutor):
    """A minimal concrete implementation of BaseLocalExecutor for testing.

    It implements the abstract get_commands method and provides a simple
    execute/cleanup behavior so you can run the evaluator locally without
    depending on your full execution harness.
    """

    def get_commands(self, data_point: CodingAgentDataPoint) -> List[str]:
        # Return an empty list of commands — the evaluator itself runs the
        # claude flow and git interactions. Implement this to match your
        # real executor's contract when integrating.
        return []

    def execute(self, data_point: CodingAgentDataPoint) -> Dict[str, Any]:
        # Simple placeholder execution result
        return {"status": "skipped", "instance_id": data_point.instance_id}

    def cleanup(self) -> None:
        # No-op for the minimal executor
        return None


def main():
    weave.init(project_name="claude_code_evaluator")
    executor = MinimalLocalExecutor()

    evaluator = ClaudeProxyEvaluator(
        executor=executor,
        anthropic_base_url="http://localhost:8082",
        anthropic_api_key="dummy",
        openai_base_url="http://127.0.0.1:8005/v1",
        openai_api_key="dummy",
        openai_model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        log_dir="./logs",
        debug=True,
    )

    evaluator.start_proxy()

    dataset = load_dataset("geekyrakshit/rust-dev", split="train")
    data_point = CodingAgentDataPoint.model_validate(dataset[0])
    rich.print(data_point)

    # Run one test sample
    result = evaluator.get_agent_response(data_point)

    print("\n===== ClaudeProxyEvaluator Test Result =====")
    for key, value in result.items():
        if isinstance(value, str) and len(value) > 300:
            print(f"{key}: {value[:300]}... [truncated]")
        else:
            print(f"{key}: {value}")

    # Stop proxy after test
    evaluator.stop_proxy()


if __name__ == "__main__":
    main()
