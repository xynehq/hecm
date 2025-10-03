import os

import rich
from dotenv import load_dotenv

from hecm.gh_utils import GithubIssueAnalyzer

load_dotenv()

analyzer = GithubIssueAnalyzer(
    repo_owner="juspay",
    repo_name="hyperswitch",
    github_token=os.getenv("GITHUB_TOKEN"),
)

issues = analyzer.fetch_issues(max_issues=100)
