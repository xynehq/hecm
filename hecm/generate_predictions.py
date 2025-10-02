import os
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory

import dspy
import rich
from git import Repo
from pydantic import BaseModel


class SWEBenchDataPoint(BaseModel):
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str
    version: str
    FAIL_TO_PASS: str
    PASS_TO_PASS: str
    environment_setup_commit: str


class SWEBenchPredictionGenerator(dspy.Module):
    def __init__(self, callbacks=None):
        super().__init__(callbacks)

    def index_repository(self, repository_dir: os.PathLike):
        all_files = [str(p) for p in Path(repository_dir).rglob("*") if p.is_file()]
        return all_files

    def forward(self, data_point: SWEBenchDataPoint) -> str:
        with TemporaryDirectory(prefix="swebench_") as temp_dir:
            repository_url = f"https://github.com/{data_point.repo}"
            repository = Repo.clone_from(url=repository_url, to_path=temp_dir)
            repository.git.checkout(data_point.base_commit)
            rich.print(glob(os.path.join(temp_dir, "*")))
