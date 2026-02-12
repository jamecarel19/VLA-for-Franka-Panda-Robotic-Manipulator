#!/Users/srisanvi/miniforge/bin/python
"""
Model Inference Controller - Runs trained model autonomously
"""

import sys
from pathlib import Path

# Add dataset_recorder to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset_recorder"))

# Enable model mode
sys.argv.append('--model')

from dataset_recorder import PandaTeleopRecorder

if __name__ == "__main__":
    controller = PandaTeleopRecorder()
    controller.run()
