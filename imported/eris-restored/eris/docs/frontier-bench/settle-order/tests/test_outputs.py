"""Grader.

Runs as root. Never imports the submitted store: that happens only inside the
unprivileged subprocess spawned by `_run_sut`. The reward is decided from this
process's exit status, so a submission cannot write it.

Three layers:
  * sixteen hand-derived corner cases with hardcoded answers, which the sealed
    reference is checked against as well, so a drifting reference fails here
    rather than quietly redefining the task;
  * fifty randomised differential programs compared against the reference;
  * one timed run over a store that is mostly idle.
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import driver  # noqa: E402
import gen  # noqa: E402
import reference  # noqa: E402

SUT_DIR = os.environ.get("GRADER_SUT_DIR", "/run/sut")
WORK = os.environ.get("GRADER_WORK", "/tmp/grader")
RUNNER = os.environ.get("GRADER_RUNNER_USER", "runner")

CORNERS = gen.corners()
RANDOM = gen.random_programs(50, seed=7000)


def _norm(entries):
    return [list(entry) for entry in entries]


def _run_sut(programs, mode, tag, timeout):
    """Run the submitted store on a batch of programs, unprivileged."""
    inp = os.path.join(WORK, "in_%s.json" % tag)
    outp = os.path.join(WORK, "out_%s.json" % tag)
    for stale in (inp, outp):
        if os.path.exists(stale):
            os.unlink(stale)
    with open(inp, "w") as handle:
        json.dump({"mode": mode, "programs": programs}, handle)
    os.chmod(inp, 0o644)

    cmd = ["runuser", "-u", RUNNER, "--",
           sys.executable, os.path.join(SUT_DIR, "sut_runner.py"), inp, outp]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail("the store did not finish %s within %.0fs" % (tag, timeout))
    if not os.path.exists(outp):
        pytest.fail("the store produced no result for %s (exit %s)\n%s\n%s"
                    % (tag, proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:]))
    with open(outp) as handle:
        return json.load(handle)["results"]


@pytest.fixture(scope="session")
def corner_results():
    return _run_sut([program for _, program, _, _ in CORNERS], "full", "corners", 300)


@pytest.fixture(scope="session")
def random_results():
    return _run_sut([program for _, program in RANDOM], "full", "random", 600)


@pytest.fixture(scope="session")
def timed_result():
    return _run_sut([gen.timed_program()], "ops", "timed",
                    gen.TIMED_CAP_SEC + 15)


# ------------------------------------------------------------------- reference

@pytest.mark.parametrize("index", range(len(CORNERS)),
                         ids=[name for name, _, _, _ in CORNERS])
def test_reference_matches_hand_derived_answer(index):
    """The sealed reference must reproduce the hand-derived answers exactly."""
    name, program, expected_ordered, expected_counts = CORNERS[index]
    ordered, counts = driver.canon(driver.run(reference.Store, program))
    assert _norm(ordered) == _norm(expected_ordered), name
    assert counts == expected_counts, name


# ------------------------------------------------------------------- submission

@pytest.mark.parametrize("index", range(len(CORNERS)),
                         ids=[name for name, _, _, _ in CORNERS])
def test_store_matches_hand_derived_answer(index, corner_results):
    name, _program, expected_ordered, expected_counts = CORNERS[index]
    result = corner_results[index]
    assert result["ok"], "%s: store raised %s" % (name, result.get("error"))
    assert result["ordered"] == _norm(expected_ordered), (
        "%s: observable events differ" % name)
    assert result["counts"] == expected_counts, (
        "%s: evaluation counts differ" % name)


@pytest.mark.parametrize("index", range(len(RANDOM)),
                         ids=["seed%d" % seed for seed, _ in RANDOM])
def test_store_matches_reference_on_random_program(index, random_results):
    seed, program = RANDOM[index]
    expected_ordered, expected_counts = driver.canon(driver.run(reference.Store, program))
    result = random_results[index]
    assert result["ok"], "seed %d: store raised %s" % (seed, result.get("error"))
    assert result["ordered"] == _norm(expected_ordered), (
        "seed %d: observable events differ" % seed)
    assert result["counts"] == expected_counts, (
        "seed %d: evaluation counts differ" % seed)


def test_idle_cells_cost_nothing(timed_result):
    """Ten thousand cells nobody is looking at, five thousand writes.

    A store that evaluates only what is needed finishes this in a fraction of a
    second. One that keeps every cell current does tens of millions of
    evaluations. The three reads at the end must still be right, so this cannot
    be passed by computing nothing.
    """
    result = timed_result[0]
    assert result["ok"], "store raised %s on the timed run" % result.get("error")
    assert result["ordered"] == gen.timed_expected_tail(), (
        "timed run produced the wrong values")
    assert result["elapsed"] <= gen.TIMED_CAP_SEC, (
        "timed run took %.1fs, the ceiling is %.1fs"
        % (result["elapsed"], gen.TIMED_CAP_SEC))
