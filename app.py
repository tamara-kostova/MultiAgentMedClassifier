"""Thin entrypoint for the Gradio UI.

Accepts the same config flags as run_pipeline.py, e.g.
    python app.py --load_4bit --calibration_file temps.json --pipeline_mode forest
"""

import sys

from ui.demo import launch


if __name__ == "__main__":
    launch(sys.argv[1:])
