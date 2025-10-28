import weave
from datasets import load_dataset

from hecm.eval_harness import ClaudeProxyEvaluator, JuspayHyperswitchLocalTestExecutor


@weave.op
def test_claude_code_eval():
    executor = JuspayHyperswitchLocalTestExecutor()
    evaluator = ClaudeProxyEvaluator(
        executor=executor,
        anthropic_base_url="http://localhost:8082",
        anthropic_api_key="dummy",
        openai_base_url="http://127.0.0.1:8005/v1",
        openai_api_key="dummy",
        openai_model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        log_dir="./logs",
        debug=True,
    )
    evaluator.start_proxy()
    dataset = load_dataset("geekyrakshit/rust-dev", split="train")
    results = evaluator.evaluate_dataset(dataset, max_data_points=2)
    return results


if __name__ == "__main__":
    weave.init(project_name="claude_code_evaluator")
    test_claude_code_eval()
