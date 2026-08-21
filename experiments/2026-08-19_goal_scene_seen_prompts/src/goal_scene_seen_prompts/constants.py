from __future__ import annotations

from pathlib import Path

EXPERIMENT_NAME = "goal_scene_seen_prompts"
TRACKIO_PROJECT = "goal-scene-seen-prompts"
DATE = "2026-08-19"

# Frozen official-data pretrain (seen positive control 20/20) — same
# checkpoint as the seen-scene probes and the cost-curve baselines.
CHECKPOINT_PATH = Path(
    "/var/tmp/vla_outputs/seen_libero90_official_20260817/checkpoints/030000/pretrained_model"
)

SUITE = "libero_goal"  # the NOVEL scene — evaluation only
SEEN_SUITE = "libero_90"  # used only to VALIDATE that twin strings are trained
MASTER_SEED = 1000
N_EVAL_EPISODES = 20
EVAL_BATCH_SIZE = 4

# Probe 3 of the diagnostic plan, mirror of the seen-scene probes: the SCENE
# is novel (libero_goal), the prompt strings are the ones the selector knows.
# Question: does the exact trained string retrieve/execute its skill in an
# unseen scene?  Success = the PROMPTED task's predicate (v2 machinery);
# behavioral metrics (eef->target distance, target displacement) discriminate
# conditions if success stays at the floor.

# goal env language -> the verbatim libero_90-trained twin string of the SAME
# skill.  Twins are validated against the libero_90 benchmark by language.
TWIN_PAIRS = (
    {
        "goal": "open the top drawer and put the bowl inside",
        "seen_twin": "open the top drawer of the cabinet and put the bowl in it",
    },
    {
        "goal": "put the bowl on top of the cabinet",
        "seen_twin": "put the black bowl on top of the cabinet",
    },
    {
        "goal": "put the bowl on the plate",
        "seen_twin": "put the black bowl on the plate",
    },
    {
        "goal": "put the wine bottle on the rack",
        "seen_twin": "put the wine bottle on the wine rack",
    },
)

# Anchor env whose TRUE instruction is itself a verbatim trained string
# (string side already optimal -> pure scene effect).
ANCHOR_TRUE_TRAINED = "turn on the stove"

# seen_cross block: swap the two bowl twins between their goal envs — does a
# trained string execute ITS OWN skill in the novel scene regardless of the
# env's task?  (Both predicates are evaluable: one scene.)
SEEN_CROSS_PAIR = ("put the bowl on the plate", "put the bowl on top of the cabinet")

NONSENSE_PROMPT = "perform the dax florp twice"
# nonsense runs on the anchor env and the first twin env.


def experiment_root() -> Path:
    return Path(__file__).resolve().parents[2]
