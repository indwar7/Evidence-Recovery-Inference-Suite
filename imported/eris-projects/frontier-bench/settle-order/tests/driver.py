"""Neutral program driver.

Builds a cell graph described by a plain-data program and runs a list of
operations against it, recording everything observable as a flat trace. The same
driver runs the reference and the submitted store, and the two traces are
compared entry for entry.

Holds no expected answers and no reference logic.
"""


def _ev(expr, store, ids, tracked=True):
    kind = expr[0]
    if kind == "c":
        return expr[1]
    if kind == "r":
        return store.get(ids[expr[1]])
    if kind == "u":
        return store.peek(ids[expr[1]])
    if kind == "add":
        total = 0
        for sub in expr[1]:
            total += _ev(sub, store, ids)
        return total
    if kind == "mod":
        return _ev(expr[1], store, ids) % expr[2]
    if kind == "gt":
        return 1 if _ev(expr[1], store, ids) > expr[2] else 0
    if kind == "if":
        if _ev(expr[1], store, ids):
            return _ev(expr[2], store, ids)
        return _ev(expr[3], store, ids)
    raise ValueError("bad expr %r" % (kind,))


OP_KINDS = ("get", "peek", "dispose", "err")


class _Filter(list):
    """Trace sink that keeps only externally visible operations.

    The timed run drives millions of evaluations; keeping every one would cost
    more memory than the work being measured. Reads, disposals and errors are
    still kept, so the timed run still has to produce the right answers.
    """

    def append(self, item):
        if item[0] in OP_KINDS:
            list.append(self, item)


def _sink(record):
    if record is True:
        return []
    if record == "ops":
        return _Filter()
    return _Filter()


def build(Store, program, record=True, trace=None, ids=None):
    """Create the graph. Returns (store, ids, trace)."""
    store = Store()
    if ids is None:
        ids = {}
    if trace is None:
        trace = _sink(record)

    def make_formula(label, expr):
        def fn(s):
            trace.append(("eval", label))
            return _ev(expr, s, ids)
        return fn

    def make_hook(label, spec):
        reads = spec.get("reads", [])
        writes = spec.get("writes", [])
        has_cleanup = spec.get("cleanup", False)

        def fn(s):
            trace.append(("hook", label))
            for expr in reads:
                _ev(expr, s, ids)
            for target, expr in writes:
                s.set(ids[target], _ev(expr, s, ids))
            if has_cleanup:
                def cleanup():
                    trace.append(("clean", label))
                return cleanup
            return None
        return fn

    for spec in program["nodes"]:
        label = spec["label"]
        kind = spec["kind"]
        if kind == "input":
            ids[label] = store.input(spec["value"])
        elif kind == "formula":
            ids[label] = store.formula(make_formula(label, spec["expr"]))
        elif kind == "hook":
            ids[label] = store.hook(make_hook(label, spec))
        else:
            raise ValueError("bad node kind %r" % (kind,))
    return store, ids, trace


def _run_ops(store, ids, trace, ops):
    for op in ops:
        name = op[0]
        try:
            if name == "set":
                store.set(ids[op[1]], op[2])
            elif name == "get":
                trace.append(("get", op[1], store.get(ids[op[1]])))
            elif name == "peek":
                trace.append(("peek", op[1], store.peek(ids[op[1]])))
            elif name == "dispose":
                store.dispose(ids[op[1]])
                trace.append(("dispose", op[1]))
            elif name == "batch":
                inner = op[1]

                def body(s, inner=inner):
                    _run_ops(s, ids, trace, inner)

                store.batch(body)
            else:
                raise ValueError("bad op %r" % (name,))
        except Exception as exc:  # noqa: BLE001 - the class name is the observable
            trace.append(("err", type(exc).__name__))


def run(Store, program, record=True):
    """Run a whole program and return its trace."""
    ids = {}
    trace = _sink(record)
    try:
        store, ids, trace = build(Store, program, record=record, trace=trace, ids=ids)
    except Exception as exc:  # noqa: BLE001 - construction can legitimately fail
        trace.append(("err", type(exc).__name__))
        return trace
    _run_ops(store, ids, trace, program["ops"])
    return trace


def canon(trace):
    """The comparable form of a trace.

    Formula evaluations are compared by count, not position: the rules fix how
    many times each cell may be evaluated but do not fix the order in which an
    implementation resolves a cell's sources. Everything externally observable —
    hook runs, cleanups, reads and their values, disposals, errors — is compared
    in order.
    """
    ordered = [tuple(e) for e in trace if e[0] != "eval"]
    counts = {}
    for entry in trace:
        if entry[0] == "eval":
            counts[entry[1]] = counts.get(entry[1], 0) + 1
    return ordered, counts
