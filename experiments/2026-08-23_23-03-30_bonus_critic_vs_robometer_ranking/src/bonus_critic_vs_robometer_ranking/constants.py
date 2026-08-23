from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "protocol.yaml"
ARTIFACTS_DIR = EXPERIMENT_ROOT / "artifacts"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
SUMMARY_DIR = RESULTS_DIR / "summary"
LOG_DIR = RESULTS_DIR / "logs"
MANIFEST_PATH = ARTIFACTS_DIR / "video_manifest_blind.jsonl"
OWN_SCORES_PATH = RAW_DIR / "own_critic_scores.jsonl"
ROBOMETER_SCORES_PATH = RAW_DIR / "robometer_scores.jsonl"
SEAL_PATH = ARTIFACTS_DIR / "scoring_complete.json"
STATUS_PATH = RESULTS_DIR / "status.json"
