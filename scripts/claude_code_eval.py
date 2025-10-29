from datasets import load_dataset

from hecm.eval_harness import ClaudeProxyEvaluator, JuspayHyperswitchLocalTestExecutor


def test_claude_code_eval():
    executor = JuspayHyperswitchLocalTestExecutor(
        show_output_logs=True,
        environment={
            "CYPRESS_CONNECTOR": "cybersource",
            "CYPRESS_BASEURL": "http://localhost:8080",
            "DEBUG": "cypress:cli",
            "CYPRESS_ADMINAPIKEY": "test_admin",
            "CYPRESS_CONNECTOR_AUTH_FILE_PATH": "/home/azureuser/soumik/hecm/creds.json",
        },
    )
    evaluator = ClaudeProxyEvaluator(
        executor=executor,
        anthropic_base_url="http://localhost:8082",
        anthropic_api_key="dummy",
        openai_base_url="http://127.0.0.1:8005/v1",
        openai_api_key="dummy",
        openai_model="archit11/Kwaipilot-KAT-Dev-Merged",
        log_dir="./logs",
        debug=True,
    )
    evaluator.start_proxy()
    dataset = load_dataset("geekyrakshit/rust-dev", split="train")
    results = evaluator.evaluate_dataset(
        dataset, max_data_points=2, result_save_path="./results.json"
    )
    return results


if __name__ == "__main__":
    test_claude_code_eval()
