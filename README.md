# HECM

Hollistic evaluation of Coding Models (LLMs)


## Installation

```bash
git clone https://github.com/AthenaAgent/hecm
cd hecm
uv pip install -e .
```

## Usage

```python
import os

from dotenv import load_dotenv

from hecm.dataset_generation import CodingAgentDataGenerator
from hecm.dataset_generation.utils import load_issues

load_dotenv()

analyzer = CodingAgentDataGenerator(
    repo_owner="juspay",
    repo_name="hyperswitch",
    github_token=os.getenv("GITHUB_TOKEN"),
    gold_patch_ignore_dirs=[
        ".github",
        ".devcontainer",
        "api-reference",
        "cypress-tests",
        "cypress-test-files",
        "docs",
    ],
    test_dirs=["cypress-tests", "cypress-test-files"],
)
issues = analyzer.generate_issues(save_to="data/issues/juspay___hyperswitch.json")
issues_with_linked_prs = analyzer.generate_linked_prs(
    issues, save_to="data/issues/juspay___hyperswitch.json"
)
data_points = analyzer.generate_data_points(issues_with_linked_prs)
data_points.export_to_huggingface("geekyrakshit/rust-dev", append_to_dataset=False)
```
