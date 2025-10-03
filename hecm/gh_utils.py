from typing import List, Literal, Optional, Union

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from tqdm.auto import tqdm


class GithubIssue(BaseModel):
    number: int
    title: str
    state: Literal["open", "closed"]
    url: str
    linked_pr_numbers: Optional[List[int]] = None


class SWEBenchDataPoint(BaseModel):
    repo: str
    instance_id: List[str]
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

    def fetch_issues(self, max_issues: Optional[int] = None) -> GithubIssuesData:
        """
        Fetch all open and closed issues from a GitHub repository using the REST API.

        Returns:
            GithubIssuesData object containing all open and closed issues
        """
        closed_issues = self.fetch_issues_by_state("closed", max_issues)

        for idx, issue in tqdm(
            enumerate(closed_issues),
            desc="Fetching linked PRs",
            total=len(closed_issues),
        ):
            closed_issues[idx].linked_pr_numbers = self.get_linked_prs(issue.url)

        return SWEBenchDataPoint(
            repo=f"{self.repo_owner}/{self.repo_name}",
            instance_id=[
                f"{self.repo_owner}__{self.repo_name}-{issue.number}"
                for issue in closed_issues
            ],
            closed_issues=closed_issues,
        )
