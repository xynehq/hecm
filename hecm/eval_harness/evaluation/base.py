from abc import ABC, abstractmethod
from typing import Union

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
    def evaluate(self, dataset: Union[Dataset, str], max_data_points: int = None):
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
        for data_point in tqdm(dataset, desc="Evaluating dataset", total=len(dataset)):
            results = self.executor.execute(data_point)
            self.executor.cleanup()
            self.results.append(results)
        return self.results
