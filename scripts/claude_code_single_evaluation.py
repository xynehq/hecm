import rich
from datasets import load_dataset

from hecm.dataset_generation.schemas import CodingAgentDataPoint
from hecm.eval_harness.agent import ClaudeCodeProxyAgent
from hecm.eval_harness.test_execution import JuspayHyperswitchLocalTestExecutor

if __name__ == "__main__":
    dataset = load_dataset("juspay/hyperswitch", split="train")
    data_point = CodingAgentDataPoint.model_validate(dataset[0])
    agent = ClaudeCodeProxyAgent()
    response = agent.get_agent_response(data_point, start_proxy=True, stop_proxy=True)
    rich.print(response)
    rich.print(response)
    test_executor = JuspayHyperswitchLocalTestExecutor(
        environment={
            "CYPRESS_CONNECTOR": "connector_id",
            "CYPRESS_BASEURL": "http://localhost:8080",
            "DEBUG": "cypress:cli",
            "CYPRESS_ADMINAPIKEY": "admin_api_key",
            "CYPRESS_CONNECTOR_AUTH_FILE_PATH": "/Users/geekyrakshit/Workspace/athena/hecm/creds.json",
        },
    )
    test_executor.execute(
        data_point,
        predicted_patch=response["claude_patch"],
        result_save_path="./results.json",
    )
