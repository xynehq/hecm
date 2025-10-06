from typing import List, Literal, Optional, Union

import requests
import weave
from bs4 import BeautifulSoup
from pydantic import BaseModel
from tqdm.auto import tqdm

from hecm.utils import (
    get_last_release_before_pr_merge,
    keep_only_dir_from_diff,
    remove_dir_from_diff,
)


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


class SWEBenchDataPoint(BaseModel):
    repo: str
    instance_id: str
    problem_statement: str
    patch: str
    test_patch: str
    created_at: str
    hints_text: str
    version: str


class SWEBenchDataGenerator:
    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
        github_token: Optional[str] = None,
        gold_patch_ignore_dirs: List[str] = [".github"],
        test_dirs: List[str] = [],
    ):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.github_token = github_token
        self.gold_patch_ignore_dirs = gold_patch_ignore_dirs
        self.test_dirs = test_dirs
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        if github_token:
            self.headers["Authorization"] = f"token {github_token}"

    @weave.op
    def get_linked_prs(self, url: str) -> Union[List[int], None]:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        elements = soup.find_all(class_="HeaderMetadata-module__metadataContent--HC0b2")
        data = [element.get_text(strip=True) for element in elements]
        try:
            return [int(item.split("#")[-1]) for item in data]
        except:
            return None

    @weave.op
    def fetch_pr_data(self, pr_number: int) -> LinkedPR:
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/pulls/{pr_number}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        pr_data = response.json()

        comments_url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/pulls/{pr_number}/comments"
        comments_response = requests.get(comments_url, headers=self.headers)
        comments_response.raise_for_status()
        comments_data = comments_response.json()
        comments = [
            PRComment(
                comment_body=comment["body"],
                diff_hunk=comment["diff_hunk"] if "diff_hunk" in comment else None,
            )
            for comment in comments_data
        ]
        return LinkedPR(
            number=pr_data["number"],
            title=pr_data["title"],
            body=pr_data["body"],
            base_commit=pr_data["base"]["sha"],
            created_at=pr_data["created_at"],
            comments=comments,
        )

    @weave.op
    def get_patch(self, pr_number: int) -> str:
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/pulls/{pr_number}"
        headers = self.headers.copy()
        headers["Accept"] = "application/vnd.github.v3.diff"
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        gold_patch = response.text
        for dir in self.gold_patch_ignore_dirs:
            gold_patch = remove_dir_from_diff(gold_patch, dir)

        test_patch = response.text
        for dir in self.test_dirs:
            test_patch = keep_only_dir_from_diff(test_patch, dir)

        return gold_patch, test_patch

    @weave.op
    def fetch_issues_by_state(
        self, state: str, max_issues: Optional[int] = None
    ) -> List[GithubIssue]:
        """Helper function to fetch issues by state with pagination."""
        issues = []
        page = 1
        per_page = 100  # Maximum allowed by GitHub API

        with tqdm(desc=f"Fetching {state} issues", unit="page") as pbar:
            while True:
                url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues"
                params = {
                    "state": state,
                    "page": page,
                    "per_page": per_page,
                    "sort": "created",
                    "direction": "desc",
                }

                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()

                page_issues = response.json()

                # Break if no more issues
                if not page_issues:
                    break

                # Filter out pull requests (they appear in issues endpoint too)
                for issue_data in page_issues:
                    if "pull_request" not in issue_data:
                        issue_number = issue_data["number"]
                        issues.append(
                            GithubIssue(
                                number=issue_number,
                                title=issue_data["title"],
                                body=issue_data["body"],
                                state=issue_data["state"],
                                url=issue_data["html_url"],
                            )
                        )

                pbar.update(1)
                pbar.set_postfix({"total_issues": len(issues)})

                if max_issues is not None and len(issues) >= max_issues:
                    return issues[:max_issues]

                if len(page_issues) < per_page:
                    break

                page += 1

        return issues

    @weave.op
    def fetch_issues(self, max_issues: Optional[int] = None) -> List[SWEBenchDataPoint]:
        """
        Fetch all open and closed issues from a GitHub repository using the REST API.

        Returns:
            List[SWEBenchDataPoint] object containing all open and closed issues
        """
        closed_issues = self.fetch_issues_by_state("closed", max_issues)

        for idx, issue in tqdm(
            enumerate(closed_issues),
            desc="Fetching linked PRs",
            total=len(closed_issues),
        ):
            linked_pr_numbers = self.get_linked_prs(issue.url)
            if linked_pr_numbers:
                closed_issues[idx].linked_pr = self.fetch_pr_data(linked_pr_numbers[0])

        data_points: List[SWEBenchDataPoint] = []
        for idx, issue in enumerate(closed_issues):
            if issue.linked_pr:
                issue_body = issue.body if issue.body else ""
                gold_patch, test_patch = self.get_patch(issue.linked_pr.number)
                data_points.append(
                    SWEBenchDataPoint(
                        repo=f"{self.repo_owner}/{self.repo_name}",
                        instance_id=f"{self.repo_owner}__{self.repo_name}-{issue.number}",
                        problem_statement=f"Bug: {issue.title}\n\n\n\n{issue_body}",
                        patch=gold_patch,
                        test_patch=test_patch,
                        created_at=issue.linked_pr.created_at,
                        hints_text=issue.linked_pr.get_hints_text(),
                        version=get_last_release_before_pr_merge(
                            self.repo_owner, self.repo_name, issue.linked_pr.number
                        )["tag_name"],
                    )
                )

        return data_points
