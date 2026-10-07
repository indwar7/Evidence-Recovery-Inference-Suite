"""Program generator: hand-derived corner cases, randomised differential
programs, and the timed adversarial schedule.

The corner cases carry hardcoded answers derived from the rules by hand. The
test suite checks the reference against them too, so a reference that drifts
fails the suite rather than silently redefining the task.
"""

import random


def _inp(label, value):
    return {"label": label, "kind": "input", "value": value}


def _f(label, expr):
    return {"label": label, "kind": "formula", "expr": expr}


def _h(label, reads, writes=None, cleanup=False):
    spec = {"label": label, "kind": "hook", "reads": reads}
    if writes:
        spec["writes"] = writes
    if cleanup:
        spec["cleanup"] = True
    return spec


R = lambda lab: ["r", lab]          # noqa: E731 - tracked read
U = lambda lab: ["u", lab]          # noqa: E731 - untracked read
C = lambda v: ["c", v]              # noqa: E731 - constant
ADD = lambda *xs: ["add", list(xs)]  # noqa: E731
MOD = lambda e, k: ["mod", e, k]     # noqa: E731
GT = lambda e, k: ["gt", e, k]       # noqa: E731
IF = lambda c, a, b: ["if", c, a, b]  # noqa: E731


# --------------------------------------------------------------------- corners

def corners():
    """[(name, program, expected_ordered, expected_eval_counts)]"""
    out = []

    # 1. A formula nothing needs is never evaluated, however many writes land.
    out.append((
        "lazy-never",
        {"nodes": [_inp("a", 0), _f("f", ADD(R("a"), C(1)))],
         "ops": [["set", "a", 1], ["set", "a", 2]]},
        [], {},
    ))

    # 2. Reading it is what forces the single evaluation.
    out.append((
        "lazy-on-read",
        {"nodes": [_inp("a", 0), _f("f", ADD(R("a"), C(1)))],
         "ops": [["set", "a", 1], ["get", "f"]]},
        [("get", "f", 2)], {"f": 1},
    ))

    # 3. Two paths to the same input: the shared cell is evaluated once, and the
    #    hook runs once.
    out.append((
        "diamond-once",
        {"nodes": [_inp("a", 0), _f("b", ADD(R("a"), C(0))),
                   _f("c", ADD(R("a"), C(1))),
                   _f("d", ADD(R("b"), R("c"))), _h("h", [R("d")])],
         "ops": [["set", "a", 1]]},
        [("hook", "h"), ("hook", "h")], {"b": 2, "c": 2, "d": 2},
    ))

    # 4. An equal result absorbs the write: nothing downstream is touched.
    out.append((
        "cutoff-absorbs",
        {"nodes": [_inp("a", 0), _f("b", MOD(R("a"), 2)),
                   _f("c", ADD(R("b"), C(100))), _h("h", [R("c")])],
         "ops": [["set", "a", 2]]},
        [("hook", "h")], {"b": 2, "c": 1},
    ))

    # 5. The cutoff stops the walk: the rest of the chain is never looked at.
    out.append((
        "cutoff-stops-chain",
        {"nodes": [_inp("a", 0), _f("g", MOD(R("a"), 2)),
                   _f("c1", ADD(R("g"), C(1))), _f("c2", ADD(R("c1"), C(1))),
                   _h("h", [R("c2")])],
         "ops": [["set", "a", 2]]},
        [("hook", "h")], {"g": 2, "c1": 1, "c2": 1},
    ))

    # 6. Dependencies follow the branch actually taken.
    out.append((
        "dynamic-deps",
        {"nodes": [_inp("a", 1), _inp("b", 10), _inp("c", 20),
                   _f("f", IF(GT(R("a"), 0), R("b"), R("c"))), _h("h", [R("f")])],
         "ops": [["set", "c", 99], ["set", "b", 11],
                 ["set", "a", 0], ["set", "b", 12]]},
        [("hook", "h"), ("hook", "h"), ("hook", "h")], {"f": 3},
    ))

    # 7. peek never creates a dependency, so the write reaches nothing.
    out.append((
        "peek-no-dep",
        {"nodes": [_inp("a", 0), _f("f", ADD(U("a"), C(0))), _h("h", [R("f")])],
         "ops": [["set", "a", 5], ["get", "f"]]},
        [("hook", "h"), ("get", "f", 0)], {"f": 1},
    ))

    # 8. Hooks run in creation order, not in the order staleness is discovered.
    out.append((
        "hook-creation-order",
        {"nodes": [_inp("a", 0), _f("f1", ADD(R("a"), C(1))),
                   _f("f2", ADD(R("a"), C(2))), _f("f3", ADD(R("a"), C(3))),
                   _h("h1", [R("f3")]), _h("h2", [R("f2")]), _h("h3", [R("f1")])],
         "ops": [["set", "a", 1]]},
        [("hook", "h1"), ("hook", "h2"), ("hook", "h3"),
         ("hook", "h1"), ("hook", "h2"), ("hook", "h3")],
        {"f1": 2, "f2": 2, "f3": 2},
    ))

    # 9. Cleanup runs immediately before the re-run.
    out.append((
        "cleanup-before-rerun",
        {"nodes": [_inp("a", 0), _h("h", [R("a")], cleanup=True)],
         "ops": [["set", "a", 1]]},
        [("hook", "h"), ("clean", "h"), ("hook", "h")], {},
    ))

    # 10. Disposal runs the pending cleanup one last time.
    out.append((
        "dispose-runs-cleanup",
        {"nodes": [_inp("a", 0), _h("h", [R("a")], cleanup=True)],
         "ops": [["dispose", "h"], ["set", "a", 1]]},
        [("hook", "h"), ("clean", "h"), ("dispose", "h")], {},
    ))

    # 11. A disposed cell is gone: reading it raises, and it stops being reached.
    out.append((
        "disposed-cell-raises",
        {"nodes": [_inp("a", 0), _f("f", ADD(R("a"), C(1)))],
         "ops": [["get", "f"], ["dispose", "f"], ["get", "f"], ["set", "a", 1]]},
        [("get", "f", 1), ("dispose", "f"), ("err", "DisposedCell")], {"f": 1},
    ))

    # 12. Writing the value that is already there is not a write.
    out.append((
        "set-equal-is-noop",
        {"nodes": [_inp("a", 0), _h("h", [R("a")])],
         "ops": [["set", "a", 0]]},
        [("hook", "h")], {},
    ))

    # 13. A batch collapses two writes into one hook run.
    out.append((
        "batch-collapses",
        {"nodes": [_inp("a", 0), _inp("b", 0), _f("f", ADD(R("a"), R("b"))),
                   _h("h", [R("f")])],
         "ops": [["batch", [["set", "a", 1], ["set", "b", 2]]]]},
        [("hook", "h"), ("hook", "h")], {"f": 2},
    ))

    # 14. Reads inside a batch are up to date even though hooks are held.
    out.append((
        "batch-read-is-current",
        {"nodes": [_inp("a", 0), _f("f", ADD(R("a"), C(1))), _h("h", [R("f")])],
         "ops": [["batch", [["set", "a", 5], ["get", "f"]]]]},
        [("hook", "h"), ("get", "f", 6), ("hook", "h")], {"f": 2},
    ))

    # 15. A hook write opens another pass, in which the dependent hook runs.
    out.append((
        "hook-write-second-pass",
        {"nodes": [_inp("a", 0), _inp("b", 0),
                   _h("h1", [R("a")], writes=[["b", ADD(R("a"), C(1))]]),
                   _h("h2", [R("b")])],
         "ops": [["set", "a", 1]]},
        [("hook", "h1"), ("hook", "h2"), ("hook", "h1"), ("hook", "h2")], {},
    ))

    # 16. A hook that keeps re-triggering itself gives up after 100 passes.
    out.append((
        "unstable-graph",
        {"nodes": [_inp("a", 0),
                   _h("h", [R("a")], writes=[["a", ADD(R("a"), C(1))]])],
         "ops": []},
        [("hook", "h")] * 100 + [("err", "UnstableGraph")], {},
    ))

    return out


# ------------------------------------------------------------------- randomised

def _rand_expr(rng, labels, depth=0):
    if depth >= 2 or rng.random() < 0.35:
        lab = rng.choice(labels)
        return U(lab) if rng.random() < 0.12 else R(lab)
    pick = rng.random()
    if pick < 0.3:
        return ADD(_rand_expr(rng, labels, depth + 1), _rand_expr(rng, labels, depth + 1))
    if pick < 0.6:
        return MOD(_rand_expr(rng, labels, depth + 1), rng.choice([2, 3, 4]))
    if pick < 0.75:
        return ADD(_rand_expr(rng, labels, depth + 1), C(rng.randint(-3, 3)))
    return IF(GT(_rand_expr(rng, labels, depth + 1), rng.randint(-1, 3)),
              _rand_expr(rng, labels, depth + 1),
              _rand_expr(rng, labels, depth + 1))


def random_program(seed):
    rng = random.Random(seed)
    nodes = []
    inputs = []
    for i in range(rng.randint(3, 5)):
        lab = "i%d" % i
        inputs.append(lab)
        nodes.append(_inp(lab, rng.randint(0, 4)))

    formulas = []
    visible = list(inputs)
    for i in range(rng.randint(6, 12)):
        lab = "f%d" % i
        nodes.append(_f(lab, _rand_expr(rng, visible)))
        formulas.append(lab)
        visible.append(lab)

    hooks = []
    for i in range(rng.randint(2, 4)):
        lab = "h%d" % i
        reads = [R(rng.choice(formulas)) for _ in range(rng.randint(1, 2))]
        writes = None
        if rng.random() < 0.2:
            writes = [[rng.choice(inputs), C(rng.randint(0, 4))]]
        nodes.append(_h(lab, reads, writes=writes, cleanup=rng.random() < 0.3))
        hooks.append(lab)

    ops = []
    disposed = False
    for _ in range(rng.randint(10, 20)):
        pick = rng.random()
        if pick < 0.45:
            ops.append(["set", rng.choice(inputs), rng.randint(0, 5)])
        elif pick < 0.65:
            ops.append(["get", rng.choice(formulas)])
        elif pick < 0.75:
            ops.append(["peek", rng.choice(formulas)])
        elif pick < 0.92:
            inner = [["set", rng.choice(inputs), rng.randint(0, 5)]
                     for _ in range(rng.randint(2, 3))]
            ops.append(["batch", inner])
        elif not disposed and hooks:
            disposed = True
            ops.append(["dispose", rng.choice(hooks)])
        else:
            ops.append(["set", rng.choice(inputs), rng.randint(0, 5)])
    return {"nodes": nodes, "ops": ops}


def random_programs(count, seed=0):
    return [(seed + i, random_program(seed + i)) for i in range(count)]


# ------------------------------------------------------------------------ timed

TIMED_IDLE = 10000
TIMED_WRITES = 5000

# Wall-clock ceiling for the timed run. The reference finishes in well under a
# second; a store that keeps idle cells up to date takes over a minute.
TIMED_CAP_SEC = 25.0


def timed_program():
    """A store that is mostly not being looked at.

    `TIMED_IDLE` formula cells read the input and are read by nothing: no hook
    depends on them and nothing gets them until the very last operation. Rule 1
    says they are never evaluated, so the whole region costs nothing however
    many writes land on the input. The only live path is a single cell reduced
    modulo 2 with a hook behind it, and every write is even, so rule 3 absorbs
    it and the hook never runs either.

    A store that keeps its cells up to date instead of waiting to be asked does
    `TIMED_IDLE` evaluations per write and runs out of time. The two reads at
    the end are there so speed cannot come from computing nothing at all.
    """
    nodes = [_inp("a", 0), _f("gate", MOD(R("a"), 2)),
             _f("live", ADD(R("gate"), C(1))), _h("h", [R("live")])]
    for i in range(TIMED_IDLE):
        nodes.append(_f("idle%d" % i, ADD(R("a"), C(i))))

    ops = [["set", "a", 2 * (w + 1)] for w in range(TIMED_WRITES)]
    last = 2 * TIMED_WRITES
    ops.append(["get", "live"])
    ops.append(["get", "idle0"])
    ops.append(["get", "idle%d" % (TIMED_IDLE - 1)])
    return {"nodes": nodes, "ops": ops, "_last": last}


def timed_expected_tail():
    """What the timed run must still report, so speed cannot come from skipping
    the work entirely."""
    last = 2 * TIMED_WRITES
    return [["get", "live", 1],
            ["get", "idle0", last],
            ["get", "idle%d" % (TIMED_IDLE - 1), last + TIMED_IDLE - 1]]
