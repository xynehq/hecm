import argparse

from hecm.eval_harness.agent import ClaudeCodeProxyAgent
from hecm.eval_harness.evaluation import Evaluator
from hecm.eval_harness.test_execution import JuspayHyperswitchLocalTestExecutor


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an agent on a dataset using a test executor"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to evaluate on (e.g., 'juspay/hyperswitch')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: 'train')",
    )
    parser.add_argument(
        "--max-data-points",
        type=int,
        default=None,
        help="Maximum number of data points to evaluate (default: None, evaluates all)",
    )
    parser.add_argument(
        "--result-save-path",
        type=str,
        default="results.json",
        help="Path to save evaluation results JSON file (default: None, don't save)",
    )

    args = parser.parse_args()

    evaluator = Evaluator(
        agent=ClaudeCodeProxyAgent(), executor=JuspayHyperswitchLocalTestExecutor()
    )
    evaluator.evaluate(
        dataset=args.dataset,
        split=args.split,
        max_data_points=args.max_data_points,
        result_save_path=args.result_save_path,
    )


if __name__ == "__main__":
    main()
