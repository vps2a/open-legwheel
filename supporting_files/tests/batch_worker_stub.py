"""A real child process for testing progress, failures and immediate cancellation."""

import json
from pathlib import Path
import signal
import sys
import time

from legwheel_batch_builder.model import EVENT_PREFIX


def emit(kind, **values):
    print(EVENT_PREFIX + json.dumps({"type": kind, **values}), flush=True)


run = Path(sys.argv[1])
outcome = sys.argv[2]
output = run / "derived" / "stub_dataset"
if outcome == "skipped":
    emit("result", status="skipped", message="Matching dataset already exists.", output=str(output))
    raise SystemExit(0)

output.mkdir(parents=True)
(output / "packets").mkdir()
emit("output", path=str(output))
emit("progress", fraction=0.25, label="Reading test recording")
if outcome == "hang":
    # SIGTERM is deliberately ignored: the GUI must kill this process directly.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(30)
    (output / "should_not_exist").touch()
if outcome == "failed":
    print("Synthetic worker failure", file=sys.stderr, flush=True)
    raise SystemExit(2)

(output / "provenance.json").write_text(json.dumps({
    "stage": "stage2_complete", "terrain_and_light_ID": "TL001",
    "experiment_config": {"environment": {"terrain_and_light_ID": "TL001"}},
}))
(output / "manifest.jsonl").write_text("")
(output / "features.jsonl").write_text("")
if outcome == "late_cancel":
    emit("progress", fraction=0.99, label="Finalizing test output")
    time.sleep(30)
emit("progress", fraction=1.0, label="Complete")
emit("result", status="completed", message="Dataset built.", output=str(output))
