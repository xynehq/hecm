import json
import os

import rich
from datasets import load_dataset

from hecm.dataset_generation.schemas import CodingAgentDataPoint
from hecm.eval_harness.agent import ClaudeCodeProxyAgent
from hecm.eval_harness.test_execution import JuspayHyperswitchLocalTestExecutor
from hecm.eval_harness.test_execution.juspay_hyperswitch import EvaluationResult


class Evaluator:
    def __init__(
        self, agent: ClaudeCodeProxyAgent, executor: JuspayHyperswitchLocalTestExecutor
    ):
        self.agent: ClaudeCodeProxyAgent = agent
        self.executor: JuspayHyperswitchLocalTestExecutor = executor

    def evaluate(
        self,
        dataset: str,
        split: str = "train",
        max_data_points: int = None,
        result_save_path: os.PathLike | None = None,
    ):
        dataset = (
            load_dataset(dataset, split=split) if isinstance(dataset, str) else dataset
        )
        dataset = (
            dataset.select(range(max_data_points))
            if max_data_points is not None
            else dataset
        )
        evaluation_results: list[EvaluationResult] = []
        for idx, data_point in enumerate(dataset):
            data_point = CodingAgentDataPoint.model_validate(data_point)
            rich.print(
                f"[bold cyan]Evaluating data point {idx + 1}/{len(dataset)}[/bold cyan]"
            )
            response = self.agent.get_agent_response(data_point)
            evaluation_result = self.executor.execute(
                data_point, response["claude_patch"]
            )
            evaluation_results.append(evaluation_result)
            rich.print(
                f"[bold green]Data point {idx + 1}/{len(dataset)} evaluated successfully[/bold green]"
            )

        if result_save_path is not None:
            with open(result_save_path, "w") as f:
                json.dump(
                    [result.model_dump() for result in evaluation_results], f, indent=4
                )
        return evaluation_results
