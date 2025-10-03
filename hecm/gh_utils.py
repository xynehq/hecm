import os
from typing import List, Literal, Optional

import github3
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


class GithubRepositoryAnalyzer:
    def __init__(self, repo_owner: str, repo_name: str) -> None:
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        gh = github3.login(token=os.getenv("GITHUB_TOKEN"))
        self.repository = gh.repository(repo_owner, repo_name)
        if not self.repository:
            raise ValueError(f"Repository {repo_owner}/{repo_name} not found")

    def fetch_issues_with_retry(
        self, state: str, max_issues: Optional[int] = None
    ) -> List[GithubIssue]:
        """Helper function to fetch issues with retry logic."""
        issues = []
        total_issues = 1
        for issue in tqdm(
            self.repository.issues(state=state), desc=f"Looking for {state} issues"
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

    def get_issues(
        self,
        max_open_issues: Optional[int] = None,
        max_closed_issues: Optional[int] = None,
    ) -> GithubIssuesData:
        """
        Get the github issue data from a github repository.

        Args:
            repo_owner: The owner/organization of the repository
            repo_name: The repository name
            max_open_issues: Maximum number of open issues to fetch (default: None)
            max_closed_issues: Maximum number of closed issues to fetch (default: None)

        Returns: GithubIssuesData
        """

        # Count issues by iterating (filtering out PRs)
        open_issues: List[GithubIssue] = []
        closed_issues: List[GithubIssue] = []

        # Fetch open and closed issues
        open_issues = self.fetch_issues_with_retry("open", max_issues=max_open_issues)
        closed_issues = self.fetch_issues_with_retry(
            "closed", max_issues=max_closed_issues
        )

        return GithubIssuesData(
            repo_owner=self.repo_owner,
            repo_name=self.repo_name,
            open_issues=open_issues,
            closed_issues=closed_issues,
        )
