from typing import List, Literal, Optional

from datasets import Dataset, concatenate_datasets, load_dataset
from pydantic import BaseModel
from tqdm.auto import tqdm


class PRComment(BaseModel):
    """A comment on a PR.
    Args:
        comment_body (str): The body of the comment.
        diff_hunk (Optional[str]): The diff hunk of the comment.

    Returns:
        str: The hints text.
    """

    comment_body: str
    diff_hunk: Optional[str] = None


class LinkedPR(BaseModel):
    """A linked PR.

    Args:
        number (int): The number of the PR.
        title (str): The title of the PR.
        body (Optional[str]): The body of the PR.
        base_commit (str): The base commit of the PR.
        created_at (str): The creation date of the PR.
        comments (List[PRComment]): The comments on the PR.
    """

    number: int
    title: str
    body: Optional[str] = None
    base_commit: str
    created_at: str
    comments: List[PRComment] = []

    def get_hints_text(self, get_comments: bool = True) -> str:
        hints_text = self.body if self.body else ""
        if get_comments:
            for comment in self.comments:
                hints_text = (
                    hints_text + f"\n\n{comment.diff_hunk}" if comment.diff_hunk else ""
                )
                hints_text += f"\n\n{comment.comment_body}\n\n" + "-" * 100
        return hints_text


class GithubIssue(BaseModel):
    """A Github issue.

    Args:
        number (int): The number of the issue.
        title (str): The title of the issue.
        body (Optional[str]): The body of the issue.
        state (Literal["open", "closed"]): The state of the issue.
        url (str): The URL of the issue.
        linked_pr (Optional[LinkedPR]): The linked PR for the issue.
    """

    number: int
    title: str
    body: Optional[str] = None
    state: Literal["open", "closed"]
    url: str
    linked_pr: Optional[LinkedPR] = None


class CodingAgentDataPoint(BaseModel):
    """A data point for a coding agent evaluation.
    Args:
        repo (str): The repository of the data point.
        instance_id (str): The instance ID of the data point.
        problem_statement (str): The problem statement of the data point.
        patch (str): The patch of the data point.
        test_patch (str): The test patch of the data point.
        created_at (str): The creation date of the data point.
        hints_text (str): The hints text of the data point.
        test_instructions (Optional[str]): The test instructions of the data point.
        base_commit (str): The base commit of the data point.
        script_to_run_tests (str): The script to run tests of the data point.
    """

    repo: str
    instance_id: str
    problem_statement: str
    patch: str
    test_patch: str
    created_at: str
    hints_text: str
    test_instructions: Optional[str] = None
    # version: str
    base_commit: str
    script_to_run_tests: str = "unknown"


class CodingAgentDataset(BaseModel):
    """A dataset of coding agent evaluation data points.

    Args:
        data_points (List[CodingAgentDataPoint]): The data points in the dataset.
    """

    data_points: List[CodingAgentDataPoint]

    def __len__(self):
        return len(self.data_points)

    def export_to_csv(self, filename: str):
        """Export the dataset to a CSV file.
        Args:
            filename (str): The name of the file to export to.
        """
        content = "repo, instance_id, problem_statement, patch, test_patch, created_at, hints_text, version, base_commit, environment_setup_commit, script_to_run_tests\n"
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
            content += f"{data_point.test_instructions}, "
            content += f"{data_point.script_to_run_tests}\n"
        with open(filename, "w") as f:
            f.write(content)

    def export_to_huggingface(
        self, dataset_name: str, append_to_dataset: bool = False
    ) -> Dataset:
        """Export the dataset to a Hugging Face dataset.
        Args:
            dataset_name (str): The name of the dataset to export to.
            append_to_dataset (bool): Whether to append the dataset to an existing dataset.

        Returns:
            Dataset: The dataset.
        """
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
