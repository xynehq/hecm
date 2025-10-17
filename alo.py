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
