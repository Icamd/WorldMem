import json
import os
import subprocess
from pathlib import Path


def test_launcher_passes_model_and_additional_server_arguments(tmp_path):
    executable = tmp_path / "vllm"
    output = tmp_path / "arguments.json"
    executable.write_text(
        '#!/usr/bin/env python3\nimport json, os, sys\nopen(os.environ["ARGUMENT_RECORD"], "w").write(json.dumps(sys.argv[1:]))\n'
    )
    executable.chmod(0o755)
    script = Path(__file__).parents[1] / "scripts/serve_vllm.sh"
    environment = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        ARGUMENT_RECORD=str(output),
        MODEL_ID="fixture/model",
        SERVED_MODEL_NAME="fixture-served",
    )
    subprocess.run(["bash", str(script), "--enforce-eager"], env=environment, check=True)
    arguments = json.loads(output.read_text())
    assert arguments[:2] == ["serve", "fixture/model"]
    assert arguments[arguments.index("--served-model-name") + 1] == "fixture-served"
    assert arguments[-1] == "--enforce-eager"
