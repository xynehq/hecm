from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union

import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from hecm.schemas import (
    GithubIssue,
    LinkedPR,
    PRComment,
    SWEBenchDataPoint,
    SWEBenchDataset,
)
from hecm.utils import (
    get_last_release_before_pr_merge,
    keep_only_dir_from_diff,
    remove_dir_from_diff,
)


class SWEBenchDataGenerator:
    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
        github_token: Optional[str] = None,
        gold_patch_ignore_dirs: List[str] = [".github"],
        test_dirs: List[str] = [],
        issues_page_counter: int = 1,
        register_commit_messages: bool = False,
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
        self.issues_page_counter = issues_page_counter
        self.register_commit_messages = register_commit_messages

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
            comments=comments if self.register_commit_messages else [],
        )

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

    def fetch_issues(
        self,
        state: str = "closed",
        max_issues: Optional[int] = None,
    ) -> List[GithubIssue]:
        """Helper function to fetch issues by state with pagination."""
        issues = []
        per_page = 100  # Maximum allowed by GitHub API

        with tqdm(desc=f"Fetching {state} issues", unit="page") as pbar:
            while True:
                url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues"
                params = {
                    "state": state,
                    "page": self.issues_page_counter,
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

                self.issues_page_counter += 1

        return issues

    def _fetch_linked_pr_for_issue(
        self, issue: GithubIssue
    ) -> Union[GithubIssue, None]:
        """Helper method to fetch linked PR for a single issue."""
        try:
            linked_pr_numbers = self.get_linked_prs(issue.url)
            if linked_pr_numbers:
                issue.linked_pr = self.fetch_pr_data(linked_pr_numbers[0])
            return issue
        except:
            return None

    def _create_data_point_from_issue(
        self, issue: GithubIssue
    ) -> Optional[SWEBenchDataPoint]:
        """Helper method to create a data point from an issue."""
        if not issue.linked_pr:
            return None

        issue_body = issue.body if issue.body else ""
        gold_patch, test_patch = self.get_patch(issue.linked_pr.number)

        if issue.linked_pr is not None:
            try:
                return SWEBenchDataPoint(
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
                    base_commit=issue.linked_pr.base_commit,
                    environment_setup_commit=issue.linked_pr.base_commit,
                )
            except ValueError:
                return None
        else:
            return None

    def generate_issues(
        self,
        max_issues: Optional[int] = None,
        max_workers: int = 10,
    ) -> List[GithubIssue]:
        issues = self.fetch_issues("closed", max_issues)

        # Parallelize fetching linked PRs
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_linked_pr_for_issue, issue): idx
                for idx, issue in enumerate(issues)
            }

            for future in tqdm(
                as_completed(futures),
                desc="Fetching linked PRs",
                total=len(issues),
            ):
                idx = futures[future]
                result = future.result()
                if result is not None:
                    issues[idx] = result

        return issues

    def generate_data_points(
        self, issues: List[GithubIssue], max_workers: int = 10
    ) -> SWEBenchDataset:
        data_points: List[SWEBenchDataPoint] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._create_data_point_from_issue, issue): idx
                for idx, issue in enumerate(issues)
            }

            for future in tqdm(
                as_completed(futures),
                desc="Creating data points",
                total=len(issues),
            ):
                data_point = future.result()
                if data_point is not None:
                    data_points.append(data_point)
        return SWEBenchDataset(data_points=data_points)
