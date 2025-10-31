import json
import os

import rich
from datasets import load_dataset

from hecm.dataset_generation.schemas import CodingAgentDataPoint
from hecm.eval_harness.agent.base import BaseAgent
from hecm.eval_harness.test_execution.base import BaseLocalExecutor, EvaluationResult


class Evaluator:
    """
    Evaluator class that evaluates the performance of the agent on the dataset.

    !!! example
        ```python
        from hecm.eval_harness.agent import ClaudeCodeProxyAgent
        from hecm.eval_harness.evaluation import Evaluator
        from hecm.eval_harness.test_execution import JuspayHyperswitchLocalTestExecutor

        evaluator = Evaluator(
            agent=ClaudeCodeProxyAgent(),
            executor=JuspayHyperswitchLocalTestExecutor(),
        )
        evaluator.evaluate(
            dataset="juspay/hyperswitch",
            split="train",
            max_data_points=8,
            result_save_path="results2.json",
        )
        ```

    Args:
        agent: The agent to evaluate. Must implement the `get_agent_response` method.
        executor: The executor to use for the evaluation. Must implement the `execute` method.
    """

    def __init__(self, agent: BaseAgent, executor: BaseLocalExecutor):
        self.agent: BaseAgent = agent
        self.executor: BaseLocalExecutor = executor

    def evaluate(
        self,
        dataset: str,
        split: str = "train",
        max_data_points: int = None,
        result_save_path: os.PathLike | None = None,
    ):
        """Evaluate the agent on the dataset.

        Args:
            dataset (str): The dataset to evaluate on.
            split (str): The split of the dataset to evaluate on.
            max_data_points (int): The maximum number of data points to evaluate on.
            result_save_path (os.PathLike | None): The path to save the evaluation results.

        Returns:
            list[EvaluationResult]: The evaluation results.
        """
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
            evaluation_result = self.executor.execute(data_point, response.patch)
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
