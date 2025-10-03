import os

import rich
from dotenv import load_dotenv

from hecm.gh_utils import fetch_issues

load_dotenv()
analyzer = fetch_issues(
    repo_owner="juspay",
    repo_name="hyperswitch",
    github_token=os.getenv("GITHUB_TOKEN"),
)


rich.print("Sample open issue =================================\n")
rich.print(analyzer.open_issues[0])
rich.print("===================================================\n")

rich.print("Sample closed issue ===============================\n")
rich.print(analyzer.closed_issues[0])
rich.print("===================================================\n")
