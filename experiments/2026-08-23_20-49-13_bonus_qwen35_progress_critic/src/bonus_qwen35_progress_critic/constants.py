from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = EXPERIMENT_ROOT / "configs/protocol.yaml"
ARTIFACTS_DIR = EXPERIMENT_ROOT / "artifacts"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
TARGET_GOAL_INSTRUCTIONS = {
    "open the middle drawer of the cabinet",
    "put the bowl on the stove",
    "put the wine bottle on top of the cabinet",
}
