import json
import os

import weave


@weave.op
def upload(result_file: os.PathLike):
    with open(result_file, "r") as f:
        results = json.load(f)
    return results


if __name__ == "__main__":
    weave.init(project_name="hecm")
    upload("results.json")
