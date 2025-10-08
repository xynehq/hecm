import os

from dotenv import load_dotenv

from hecm import SWEBenchDataGenerator

load_dotenv()

analyzer = SWEBenchDataGenerator(
    repo_owner="juspay",
    repo_name="hyperswitch",
    github_token=os.getenv("GITHUB_TOKEN"),
    gold_patch_ignore_dirs=[".github", "cypress-tests", "cypress-test-files"],
    test_dirs=["cypress-tests", "cypress-test-files"],
)
issues = analyzer.generate_issues(max_issues=300)
print(f"{analyzer.issues_page_counter=}")
data_points = analyzer.generate_data_points(issues)
data_points.export_to_huggingface("geekyrakshit/hyperswitch", append_to_dataset=True)
