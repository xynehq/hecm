import rich
from dotenv import load_dotenv

from hecm.gh_utils import GithubRepositoryAnalyzer

load_dotenv()
analyzer = GithubRepositoryAnalyzer(
    repo_owner="juspay",
    repo_name="hyperswitch",
)
issues = analyzer.get_issues(
    max_open_issues=100,
    max_closed_issues=100,
)


rich.print("Sample open issue =================================\n")
rich.print(issues.open_issues[0])
rich.print("===================================================\n")

rich.print("Sample closed issue ===============================\n")
rich.print(issues.closed_issues[0])
rich.print("===================================================\n")
