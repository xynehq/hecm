# HECM

Hollistic evaluation of Coding Models (LLMs)


## Installation

```bash
git clone https://github.com/AthenaAgent/hecm
cd hecm
uv pip install -e .
```

If you want to use a sandboxed environment, you need to install [Docker](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install). Also, make sure that the docker daemon is running and you have the necessary permissions to run docker commands. Note that for certain local executors like `JuspayHyperswitchLocalTestExecutor` as well, you need to have docker and docker compose installed.

## Usage

### Generating Coding Agent Data for a given repository

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

### Evaluating Coding Agent Data for a given repository

#### Running in a sandboxed environment

```python
import weave
from datasets import load_dataset

from hecm.dataset_generation.schemas import CodingAgentDataPoint
from hecm.eval_harness.test_execution import JuspayHyperswitchSandboxedTestExecutor


@weave.op
def test_cypress_execution():
    dataset = load_dataset("geekyrakshit/rust-dev", split="train")
    data_point = CodingAgentDataPoint.model_validate(dataset[0])
    executor = JuspayHyperswitchSandboxedTestExecutor(show_output_logs=True)
    results = executor.execute(data_point)
    executor.cleanup()
    return results


weave.init(project_name="hyperswitch")
test_cypress_execution()
```

#### Running in a local environment

```python
import weave
from datasets import load_dataset

from hecm.dataset_generation.schemas import CodingAgentDataPoint
from hecm.eval_harness.test_execution import JuspayHyperswitchLocalTestExecutor


@weave.op
def test_cypress_execution():
    dataset = load_dataset("geekyrakshit/rust-dev", split="train")
    data_point = CodingAgentDataPoint.model_validate(dataset[0])
    executor = JuspayHyperswitchLocalTestExecutor(show_output_logs=True)
    results = executor.execute(data_point)
    executor.cleanup()
    return results


weave.init(project_name="hyperswitch")
test_cypress_execution()
```
