from typing import List, Literal, Optional

from datasets import Dataset, concatenate_datasets, load_dataset
from pydantic import BaseModel
from tqdm.auto import tqdm


class PRComment(BaseModel):
    comment_body: str
    diff_hunk: Optional[str] = None


class LinkedPR(BaseModel):
    number: int
    title: str
    body: Optional[str] = None
    base_commit: str
    created_at: str
    comments: List[PRComment] = []

    def get_hints_text(self) -> str:
        hints_text = self.body if self.body else ""
        for comment in self.comments:
            hints_text = (
                hints_text + f"\n\n{comment.diff_hunk}" if comment.diff_hunk else ""
            )
            hints_text += f"\n\n{comment.comment_body}\n\n" + "-" * 100
        return hints_text


class GithubIssue(BaseModel):
    number: int
    title: str
    body: Optional[str] = None
    state: Literal["open", "closed"]
    url: str
    linked_pr: Optional[LinkedPR] = None


class CodingAgentDataPoint(BaseModel):
    repo: str
    instance_id: str
    problem_statement: str
    patch: str
    test_patch: str
    created_at: str
    hints_text: str
    # version: str
    base_commit: str
    # environment_setup_commit: str


class CodingAgentDataset(BaseModel):
    data_points: List[CodingAgentDataPoint]

    def __len__(self):
        return len(self.data_points)

    def export_to_csv(self, filename: str):
        content = "repo, instance_id, problem_statement, patch, test_patch, created_at, hints_text, version, base_commit, environment_setup_commit\n"
        for data_point in tqdm(self.data_points, desc="Exporting to CSV"):
            content += f"{data_point.repo}, "
            content += f"{data_point.instance_id}, "
            content += f"{data_point.problem_statement}, "
            content += f"{data_point.patch}, "
            content += f"{data_point.test_patch}, "
            content += f"{data_point.created_at}, "
            content += f"{data_point.hints_text}, "
            # content += f"{data_point.version}, "
            content += f"{data_point.base_commit}, "
            # content += f"{data_point.environment_setup_commit}\n"
        with open(filename, "w") as f:
            f.write(content)

    def export_to_huggingface(
        self, dataset_name: str, append_to_dataset: bool = False
    ) -> Dataset:
        keys = self.data_points[0].model_fields.keys()
        dataset_dict = {key: [] for key in keys}
        for data_point in tqdm(self.data_points, desc="Exporting to Hugging Face"):
            for key in keys:
                dataset_dict[key].append(getattr(data_point, key))
        dataset = Dataset.from_dict(dataset_dict)
        if append_to_dataset:
            existing_dataset = load_dataset(dataset_name)["train"]
            dataset = concatenate_datasets([existing_dataset, dataset])
        dataset.push_to_hub(dataset_name)
        return dataset
