from hecm.eval_harness.agent import ClaudeCodeProxyAgent
from hecm.eval_harness.evaluation import Evaluator
from hecm.eval_harness.test_execution import JuspayHyperswitchLocalTestExecutor

if __name__ == "__main__":
    evaluator = Evaluator(
        agent=ClaudeCodeProxyAgent(), executor=JuspayHyperswitchLocalTestExecutor()
    )
    evaluator.evaluate(
        dataset="juspay/hyperswitch",
        split="train",
        max_data_points=8,
        result_save_path="results2.json",
    )
