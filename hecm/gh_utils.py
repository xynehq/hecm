from typing import List, Literal, Optional

import requests
from pydantic import BaseModel
from tqdm.auto import tqdm


class GithubIssue(BaseModel):
    number: int
    title: str
    state: Literal["open", "closed"]
    url: str


class GithubIssuesData(BaseModel):
    repo_owner: str
    repo_name: str
    open_issues: List[GithubIssue]
    closed_issues: List[GithubIssue]


class GithubIssueAnalyzer:
    def __init__(
        self, repo_owner: str, repo_name: str, github_token: Optional[str] = None
    ):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.github_token = github_token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        if github_token:
            self.headers["Authorization"] = f"token {github_token}"

    def fetch_issues_by_state(self, state: str) -> List[GithubIssue]:
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
                                state=issue_data["state"],
                                url=issue_data["html_url"],
                            )
                        )

                pbar.update(1)
                pbar.set_postfix({"total_issues": len(issues)})

                if len(page_issues) < per_page:
                    break

                page += 1

        return issues

    def fetch_issues(self) -> GithubIssuesData:
        """
        Fetch all open and closed issues from a GitHub repository using the REST API.

        Returns:
            GithubIssuesData object containing all open and closed issues
        """
        open_issues = self.fetch_issues_by_state("open")
        closed_issues = self.fetch_issues_by_state("closed")

        return GithubIssuesData(
            repo_owner=self.repo_owner,
            repo_name=self.repo_name,
            open_issues=open_issues,
            closed_issues=closed_issues,
        )
