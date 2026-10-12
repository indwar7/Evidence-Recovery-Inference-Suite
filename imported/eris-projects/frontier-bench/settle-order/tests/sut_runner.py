"""Runs the submitted store, unprivileged, in its own process.

This is the only place the agent's code is imported. Its import path holds the
submitted store, the neutral driver, and this file — nothing else. The reference
and the expected answers live in a root-only directory that the account running
this cannot read.

Reads a JSON batch of programs, writes a JSON list of results. It never decides
anything: pytest does that, in a process that never imports the store.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import driver  # noqa: E402
import store  # noqa: E402


def run_one(program, mode):
    started = time.perf_counter()
    try:
        trace = driver.run(store.Store, program, record=(True if mode == "full" else "ops"))
        ordered, counts = driver.canon(trace)
        return {
            "ok": True,
            "ordered": [list(entry) for entry in ordered],
            "counts": counts,
            "elapsed": time.perf_counter() - started,
        }
    except BaseException as exc:  # noqa: BLE001 - a crash is a result, not an error
        return {
            "ok": False,
            "error": type(exc).__name__,
            "elapsed": time.perf_counter() - started,
        }


def main():
    with open(sys.argv[1]) as handle:
        payload = json.load(handle)
    mode = payload.get("mode", "full")
    results = [run_one(program, mode) for program in payload["programs"]]
    with open(sys.argv[2], "w") as handle:
        json.dump({"results": results}, handle)


if __name__ == "__main__":
    main()
