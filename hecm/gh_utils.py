import os
import time
from typing import List, Literal, Optional

import github3
from pydantic import BaseModel
from tqdm.auto import tqdm


class GithubIssue(BaseModel):
    number: int
    title: str
    state: Literal["open", "closed"]
    url: str
    linked_pr_url: Optional[str] = None


class GithubIssuesData(BaseModel):
    repo_owner: str
    repo_name: str
    open_issues: List[GithubIssue]
    closed_issues: List[GithubIssue]


def get_issues(
    repo_owner: str,
    repo_name: str,
    max_open_issues: Optional[int] = None,
    max_closed_issues: Optional[int] = None,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> GithubIssuesData:
    """
    Get the github issue data from a github repository.

    Args:
        repo_owner: The owner/organization of the repository
        repo_name: The repository name
        max_open_issues: Maximum number of open issues to fetch (default: None)
        max_closed_issues: Maximum number of closed issues to fetch (default: None)
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

    def fetch_issues_with_retry(
        state: str, max_issues: Optional[int] = None
    ) -> List[GithubIssue]:
        """Helper function to fetch issues with retry logic."""
        issues = []
        total_issues = 1
        for issue in tqdm(
            repository.issues(state=state), desc=f"Looking for {state} issues"
        ):
            try:
                if not issue.pull_request_urls:
                    issues.append(
                        GithubIssue(
                            number=issue.number,
                            title=issue.title,
                            state=issue.state,
                            url=issue.url,
                        )
                    )
            except Exception:
                pass

            total_issues += 1
            if max_issues and total_issues >= max_issues:
                break

        return issues

    # Fetch open and closed issues
    open_issues = fetch_issues_with_retry("open", max_issues=max_open_issues)
    closed_issues = fetch_issues_with_retry("closed", max_issues=max_closed_issues)

    return GithubIssuesData(
        repo_owner=repo_owner,
        repo_name=repo_name,
        open_issues=open_issues,
        closed_issues=closed_issues,
    )
