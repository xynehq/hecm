# HECM:Hollistic evaluation of Coding Models

HECM is a library meant to hollistically evaluate the agentic capbilities of coding LLMs. It consists of 2 primary features:

1. Mining data from Github issues to create agentic benchmarks for evaluating the agenting capabilities of coding models to solve problems.
2. An evaluation harness with a flexible API, designed to evaluate agents and models by executing corresponding testcases both in sandboxed and un-sandboxed manner.


## Installation

```bash
git clone https://github.com/xynehq/hecm
cd hecm
uv pip install -r pyproject.toml --group dev
```

## Usage

### Generating Coding Agent evaluation data for a given repository

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
issues = analyzer.generate_issues(
    save_to="data/issues/juspay___hyperswitch.json"
)
issues_with_linked_prs = analyzer.generate_linked_prs(
    issues, save_to="data/issues/juspay___hyperswitch.json"
)
data_points = analyzer.generate_data_points(issues_with_linked_prs)
data_points.export_to_huggingface(
    "juspay/hyperswitch", append_to_dataset=False
)
```

### Running the evaluation harness

```python
from hecm.eval_harness.agent import ClaudeCodeProxyAgent
from hecm.eval_harness.evaluation import Evaluator
from hecm.eval_harness.test_execution import JuspayHyperswitchLocalTestExecutor

evaluator = Evaluator(
    agent=ClaudeCodeProxyAgent(),
    executor=JuspayHyperswitchLocalTestExecutor(),
)
evaluator.evaluate(
    dataset="juspay/hyperswitch", # 🤗 address of the dataset
    split="train",
    max_data_points=8,
    result_save_path="results.json",
)
```
