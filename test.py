import os

import weave
from dotenv import load_dotenv

from hecm.gh_utils import GithubIssueAnalyzer

load_dotenv()
weave.init(project_name="hyperswitch")

analyzer = GithubIssueAnalyzer(
    repo_owner="juspay",
    repo_name="hyperswitch",
    github_token=os.getenv("GITHUB_TOKEN"),
)

data_points = analyzer.fetch_issues(max_issues=50)
