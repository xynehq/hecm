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

    def get_closing_pr_url(issue) -> Optional[str]:
        """Helper function to find the PR that closed an issue."""
        try:
            # First, try to find cross-referenced PRs in the timeline
            for event in issue.events():
                # Check if there's a closed event with a commit
                if event.event == "closed" and event.commit_id:
                    commit_sha = event.commit_id
                    # Search for a PR with this merge commit
                    # More efficient: use the search API or iterate through a limited set
                    try:
                        # Try to find PR by searching recent closed PRs
                        # Limit to recently updated PRs for efficiency
                        for pr in repository.pull_requests(
                            state="closed", sort="updated", direction="desc"
                        ):
                            if pr.merge_commit_sha == commit_sha:
                                return pr.html_url
                            # Only check recent PRs (optimization)
                            # If PR is too old, stop searching
                            if pr.updated_at and issue.closed_at:
                                time_diff = (
                                    issue.closed_at - pr.updated_at
                                ).total_seconds()
                                # If PR was updated more than 30 days before issue closed, skip rest
                                if time_diff > 30 * 24 * 3600:
                                    break
                    except Exception:
                        pass
                    break

            # Alternative: Check timeline for cross-referenced PRs
            # This is more reliable but requires timeline access
            try:
                # github3 might not have timeline, so we'll use events and cross-references
                for event in issue.events():
                    if event.event == "cross-referenced" and hasattr(event, "source"):
                        source = event.source
                        if hasattr(source, "issue") and hasattr(
                            source.issue, "pull_request"
                        ):
                            # This is a PR that references the issue
                            pr = source.issue.pull_request
                            if pr and hasattr(pr, "html_url"):
                                return pr.html_url
            except Exception:
                pass

        except Exception as e:
            # If we can't fetch PR info, just return None
            print(f"Warning: Could not fetch PR info for issue #{issue.number}: {e}")
        return None

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
                        # For closed issues, try to find the linked PR
                        linked_pr_url = None
                        if state == "closed":
                            linked_pr_url = get_closing_pr_url(issue)

                        issues.append(
                            GithubIssue(
                                number=issue.number,
                                title=issue.title,
                                state=issue.state,
                                url=issue.url,
                                linked_pr_url=linked_pr_url,
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
