import os
from typing import List, Literal, Optional

import requests
from pydantic import BaseModel
from tqdm.auto import tqdm


class GithubIssue(BaseModel):
    number: int
    title: str
    state: Literal["open", "closed"]
    url: str
    comments: List[str] = []


class GithubIssuesData(BaseModel):
    repo_owner: str
    repo_name: str
    open_issues: List[GithubIssue]
    closed_issues: List[GithubIssue]


def fetch_issues(
    repo_owner: str,
    repo_name: str,
    github_token: Optional[str] = None,
) -> GithubIssuesData:
    """
    Fetch all open and closed issues from a GitHub repository using the REST API.

    Args:
        repo_owner: The owner/organization of the repository
        repo_name: The name of the repository
        github_token: Optional GitHub personal access token for authentication (to avoid rate limits)

    Returns:
        GithubIssuesData object containing all open and closed issues
    """
    base_url = "https://api.github.com"
    headers = {
        "Accept": "application/vnd.github.v3+json",
    }

    # Add authentication if token is provided
    if github_token is None:
        github_token = os.getenv("GITHUB_TOKEN")

    if github_token:
        headers["Authorization"] = f"token {github_token}"

    def fetch_issues_by_state(state: str) -> List[GithubIssue]:
        """Helper function to fetch issues by state with pagination."""
        issues = []
        page = 1
        per_page = 100  # Maximum allowed by GitHub API

        with tqdm(desc=f"Fetching {state} issues", unit="page") as pbar:
            while True:
                url = f"{base_url}/repos/{repo_owner}/{repo_name}/issues"
                params = {
                    "state": state,
                    "page": page,
                    "per_page": per_page,
                    "sort": "created",
                    "direction": "desc",
                }

                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()

                page_issues = response.json()

                # Break if no more issues
                if not page_issues:
                    break

                # Filter out pull requests (they appear in issues endpoint too)
                for issue_data in page_issues:
                    if "pull_request" not in issue_data:
                        issues.append(
                            GithubIssue(
                                number=issue_data["number"],
                                title=issue_data["title"],
                                state=issue_data["state"],
                                url=issue_data["html_url"],
                                comments=[],  # Can be extended to fetch comments
                            )
                        )

                pbar.update(1)
                pbar.set_postfix({"total_issues": len(issues)})

                # Check if there are more pages
                if len(page_issues) < per_page:
                    break

                page += 1

        return issues

    # Fetch open and closed issues
    open_issues = fetch_issues_by_state("open")
    closed_issues = fetch_issues_by_state("closed")

    return GithubIssuesData(
        repo_owner=repo_owner,
        repo_name=repo_name,
        open_issues=open_issues,
        closed_issues=closed_issues,
    )
