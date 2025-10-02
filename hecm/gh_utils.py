import os
import time
from typing import List, Literal

import github3
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


def get_issues(
    repo_owner: str, repo_name: str, max_retries: int = 3, retry_delay: int = 5
) -> GithubIssuesData:
    """
    Get the github issue data from a github repository.

    Args:
        repo_owner: The owner/organization of the repository
        repo_name: The repository name
        max_retries: Maximum number of retries on timeout (default: 3)
        retry_delay: Delay in seconds between retries (default: 5)

    Returns: GithubIssuesData
    """
    gh = github3.login(token=os.getenv("GITHUB_TOKEN"))
    repository = gh.repository(repo_owner, repo_name)
    if not repository:
        raise ValueError(f"Repository {repo_owner}/{repo_name} not found")

    # Count issues by iterating (filtering out PRs)
    open_issues: List[GithubIssue] = []
    closed_issues: List[GithubIssue] = []

    def fetch_issues_with_retry(state: str) -> List[GithubIssue]:
        """Helper function to fetch issues with retry logic."""
        issues = []
        retries = 0

        while retries < max_retries:
            try:
                for issue in tqdm(
                    repository.issues(state=state), desc=f"Looking for {state} issues"
                ):
                    if not issue.pull_request_urls:
                        issues.append(
                            GithubIssue(
                                number=issue.number,
                                title=issue.title,
                                state=issue.state,
                                url=issue.url,
                            )
                        )
                break  # Success, exit retry loop
            except github3.exceptions.ConnectionError as e:
                retries += 1
                if retries >= max_retries:
                    print(
                        f"Failed to fetch {state} issues after {max_retries} retries: {e}"
                    )
                    raise
                print(
                    f"Timeout occurred, retrying in {retry_delay}s... ({retries}/{max_retries})"
                )
                time.sleep(retry_delay)

        return issues

    # Fetch open and closed issues
    open_issues = fetch_issues_with_retry("open")
    closed_issues = fetch_issues_with_retry("closed")

    return GithubIssuesData(
        repo_owner=repo_owner,
        repo_name=repo_name,
        open_issues=open_issues,
        closed_issues=closed_issues,
    )
