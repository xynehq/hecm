def main():
    import logging
    import os

    import datasets
    from datasets import load_dataset

    from hecm.dataset_generation.schemas import CodingAgentDataPoint
    from hecm.eval_harness.evaluation.claude_code_evaluator import ClaudeProxyEvaluator
    from hecm.eval_harness.test_execution.base import BaseLocalExecutor

    """Simple test entrypoint for ClaudeProxyEvaluator."""
    # Initialize a local executor (or use your sandboxed one)

    # Instantiate evaluator
    evaluator = ClaudeProxyEvaluator(
        executor=None,
        anthropic_base_url="http://localhost:8082",
        anthropic_api_key="dummy",
        openai_base_url="http://127.0.0.1:8005/v1",
        openai_api_key="dummy",
        openai_model="archit11/Kwaipilot-KAT-Dev-Merged",
        log_dir="./logs",
        debug=True,
    )

    # Start proxy manually (optional)
    evaluator.start_proxy()

    # Create a fake minimal CodingAgentDataPoint for testing
    dataset = load_dataset("juspay/hyperswitch", split="train")
    print(dataset[0])
    data_point = CodingAgentDataPoint.model_validate(dataset[0])
    # data_point = CodingAgentDataPoint(
    #     instance_id="test_001",
    #     repo="fuergaosi233/claude-code-proxy",
    #     problem_statement="Fix a typo in the README file.",
    #     base_commit="main",
    #     patch="",
    #     test_patch="",
    #     hints_text="",
    #     test_instructions="",
    # )

    # Run one test sample
    result = evaluator.get_agent_response(data_point)

    print("\n===== ClaudeProxyEvaluator Test Result =====")
    for key, value in result.items():
        if isinstance(value, str) and len(value) > 300:
            print(f"{key}: {value[:300]}... [truncated]")
        else:
            print(f"{key}: {value}")

    # Stop proxy after test
    evaluator.stop_proxy()

    from hecm.eval_harness.test_execution.juspay_hyperswitch import (
        JuspayHyperswitchLocalTestExecutor,
    )

    juspay_executor = JuspayHyperswitchLocalTestExecutor(
        show_output_logs=True,
        environment={
            "CYPRESS_CONNECTOR": "cybersource",
            "CYPRESS_BASEURL": "http://localhost:8080",
            "DEBUG": "cypress:cli",
            "CYPRESS_ADMINAPIKEY": "test_admin",
            "CYPRESS_CONNECTOR_AUTH_FILE_PATH": "/home/azureuser/soumik/hecm/creds.json",
        },
    )

    print(juspay_executor)


if __name__ == "__main__":
    main()
