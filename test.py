import rich
from dotenv import load_dotenv

from hecm.gh_utils import get_issues

load_dotenv()
issues = get_issues(repo_owner="juspay", repo_name="hyperswitch")

rich.print("Sample open issue =================================\n")
rich.print(issues.open_issues[0])
rich.print("===================================================\n")

rich.print("Sample closed issue ===============================\n")
rich.print(issues.closed_issues[0])
rich.print("===================================================\n")
