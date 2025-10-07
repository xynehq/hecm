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

from hecm import SWEBenchDataGenerator

load_dotenv()

analyzer = SWEBenchDataGenerator(
    repo_owner="juspay",
    repo_name="hyperswitch",
    github_token=os.getenv("GITHUB_TOKEN"),
    gold_patch_ignore_dirs=[".github", "cypress-tests", "cypress-test-files"],
    test_dirs=["cypress-tests", "cypress-test-files"],
)

data_points = analyzer.fetch_issues(max_issues=50)

# export to csv
data_points.export_to_csv("swebench_dataset.csv")

# export to huggingface
data_points.export_to_huggingface("geekyrakshit/hyperswitch")
```
