import json
import os
from abc import ABC, abstractmethod
from typing import Union

import rich
import weave
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm

from hecm.dataset_generation.schemas import CodingAgentDataPoint
from hecm.eval_harness.test_execution.base import (
    BaseLocalExecutor,
    BaseSandboxedExecutor,
)


class BaseEvaluator(ABC):
    def __init__(self, executor: Union[BaseLocalExecutor, BaseSandboxedExecutor]):
        super().__init__()
        self.executor = executor
        self.executor.show_output_logs = False
        self.results = []

    @abstractmethod
    def get_agent_response(self, data_point: CodingAgentDataPoint) -> str:
        pass

    @weave.op
    def evaluate(
        self,
        dataset: Union[Dataset, str],
        max_data_points: int = None,
        result_save_path: os.PathLike | None = None,
    ):
        dataset = (
            load_dataset(dataset, split="train")
            if isinstance(dataset, str)
            else dataset
        )
        dataset = (
            dataset.select(range(max_data_points))
            if max_data_points is not None
            else dataset
        )
        dataset_iterator = (
            tqdm(enumerate(dataset), desc="Evaluating dataset", total=len(dataset))
            if self.executor.show_output_logs
            else enumerate(dataset)
        )
        for idx, data_point in dataset_iterator:
            if self.executor.show_output_logs:
                # rich.print(
                #     f"[bold cyan]Evaluating data point {idx + 1}/{len(dataset)}[/bold cyan]"
                # )
            results = self.executor.execute(data_point)
            self.executor.cleanup()
            self.results.append(results)
            if self.executor.show_output_logs:
                rich.print(
                    f"[bold green]Data point {idx + 1}/{len(dataset)} evaluated successfully[/bold green]"
                )

        if result_save_path is not None:
            with open(result_save_path, "w") as f:
                json.dump(self.results, f)

        return self.results
