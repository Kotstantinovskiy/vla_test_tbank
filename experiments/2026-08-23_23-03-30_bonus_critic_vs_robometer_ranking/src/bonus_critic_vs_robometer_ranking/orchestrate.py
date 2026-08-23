from __future__ import annotations

import traceback

from .aggregate import aggregate, write_seal
from .manifest import prepare
from .score import score
from .trackio_report import log_results
from .utils import load_config, set_status, utc_now


def main() -> None:
    config = load_config()
    expected = int(config["scope"]["expected_videos"])
    try:
        set_status("preparing", started_at=utc_now(), expected_predictions=expected * 2, completed_predictions=0)
        prepare()
        score("own")
        score("robometer")
        set_status("sealing", completed_predictions=expected * 2, expected_predictions=expected * 2)
        write_seal(expected)
        set_status("aggregating", completed_predictions=expected * 2, expected_predictions=expected * 2)
        aggregate()
        set_status("logging_trackio", completed_predictions=expected * 2, expected_predictions=expected * 2)
        log_results()
        set_status("complete", completed_predictions=expected * 2, expected_predictions=expected * 2)
    except Exception as error:
        set_status("failed", error=repr(error), traceback=traceback.format_exc())
        raise
