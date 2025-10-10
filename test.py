import os

from dotenv import load_dotenv

from hecm import SWEBenchDataGenerator
from hecm.utils import load_issues

load_dotenv()

analyzer = SWEBenchDataGenerator(
    repo_owner="juspay",
    repo_name="hyperswitch",
    github_token=os.getenv("GITHUB_TOKEN"),
    gold_patch_ignore_dirs=[".github", "cypress-tests", "cypress-test-files"],
    test_dirs=["cypress-tests", "cypress-test-files"],
)
# issues = analyzer.generate_issues(save_to="data/issues/juspay___hyperswitch.json")
issues = load_issues("data/issues/juspay___hyperswitch.json")
