"""Watch a Kaggle kernel until it finishes, then download its output.

Usage: python3 kaggle-kernel/watch.py <owner>/<slug> <out-dir> [max_polls]
Polls `kaggle kernels status` every 120s. On "complete" it runs
`kaggle kernels output -p <out-dir> -o` and prints the metrics.json
contents if present. On error/failed/cancelled it exits 1 with the
status output. This script is meant to run in the background while
the kernel trains; its completion is what wakes the main agent.
"""

import json
import subprocess
import sys
import time
from pathlib import Path


def run(*args):
    return subprocess.run(args, capture_output=True, text=True, timeout=120)


def main():
    kernel, out_dir = sys.argv[1], sys.argv[2]
    max_polls = int(sys.argv[3]) if len(sys.argv) > 3 else 90  # 90 x 120s = 3h
    for i in range(max_polls):
        status = run("kaggle", "kernels", "status", kernel)
        text = (status.stdout + status.stderr).lower()
        print(f"[{i+1}/{max_polls}] status: {(status.stdout or status.stderr).strip()[:300]}",
              flush=True)
        if "complete" in text:
            print("Kernel complete — downloading output…", flush=True)
            dl = run("kaggle", "kernels", "output", kernel, "-p", out_dir, "-o")
            print(dl.stdout, dl.stderr, flush=True)
            metrics = Path(out_dir) / "metrics.json"
            if metrics.exists():
                print("METRICS:", metrics.read_text(), flush=True)
            return 0
        if any(word in text for word in ("error", "failed", "cancelled", "canceled")):
            print("Kernel did not complete.", flush=True)
            return 1
        time.sleep(120)
    print("Watcher timed out waiting for the kernel.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
