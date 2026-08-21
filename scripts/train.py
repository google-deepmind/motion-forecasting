#!/usr/bin/env python3
"""Launch training for the motion forecasting model.

Usage:
    python scripts/train.py                  # conf/train.yaml
    python scripts/train.py epochs=10 lr=1e-4   # extra Hydra overrides

Equivalent to `python -m engine.train --config-name=train` with a
timestamped experiment name.
"""

import os
import sys
import datetime
import subprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
command = [
    sys.executable, "-m", "engine.train", "--config-name=train",
    f"experiment=train_{timestamp}",
    *sys.argv[1:],  # pass through extra Hydra overrides
]
sys.exit(subprocess.run(command).returncode)
