"""Scratch harness.

Runs a few small graphs by hand and prints what the store did, so you can watch
the behaviour rather than reason about it in the abstract. Nothing here is a
test and nothing here is graded — add cases of your own.

    python /app/harness.py
"""

import sys

sys.path.insert(0, "/app/src")

from store import Store  # noqa: E402


def watched(label, log):
    """A formula body that announces every evaluation."""
    def make(fn):
        def body(store):
            log.append("eval %s" % label)
            return fn(store)
        return body
    return make


def example_idle_cell():
    """A formula nobody reads. How many times does it get evaluated?"""
    log = []
    store = Store()
    a = store.input(0)
    store.formula(watched("f", log)(lambda s: s.get(a) + 1))
    store.set(a, 1)
    store.set(a, 2)
    print("idle cell         ->", log)


def example_equal_result():
    """The formula's value comes out the same after the write."""
    log = []
    store = Store()
    a = store.input(0)
    b = store.formula(watched("b", log)(lambda s: s.get(a) % 2))
    c = store.formula(watched("c", log)(lambda s: s.get(b) + 100))
    store.hook(lambda s: log.append("hook saw %s" % s.get(c)))
    store.set(a, 2)
    print("equal result      ->", log)


def example_diamond():
    """Two paths from one input meet again."""
    log = []
    store = Store()
    a = store.input(0)
    b = store.formula(watched("b", log)(lambda s: s.get(a) + 0))
    c = store.formula(watched("c", log)(lambda s: s.get(a) + 1))
    d = store.formula(watched("d", log)(lambda s: s.get(b) + s.get(c)))
    store.hook(lambda s: log.append("hook saw %s" % s.get(d)))
    store.set(a, 1)
    print("diamond           ->", log)


def example_branch():
    """The formula reads one of two cells depending on a switch."""
    log = []
    store = Store()
    switch = store.input(1)
    left = store.input(10)
    right = store.input(20)
    f = store.formula(watched("f", log)(
        lambda s: s.get(left) if s.get(switch) > 0 else s.get(right)))
    store.hook(lambda s: log.append("hook saw %s" % s.get(f)))
    store.set(right, 99)
    store.set(left, 11)
    print("branch            ->", log)


def example_hook_order():
    """Three hooks, created in one order, depending on cells made in another."""
    log = []
    store = Store()
    a = store.input(0)
    f1 = store.formula(lambda s: s.get(a) + 1)
    f2 = store.formula(lambda s: s.get(a) + 2)
    f3 = store.formula(lambda s: s.get(a) + 3)
    store.hook(lambda s: (log.append("h1"), s.get(f3)) and None)
    store.hook(lambda s: (log.append("h2"), s.get(f2)) and None)
    store.hook(lambda s: (log.append("h3"), s.get(f1)) and None)
    log.append("--- write ---")
    store.set(a, 1)
    print("hook order        ->", log)


if __name__ == "__main__":
    example_idle_cell()
    example_equal_result()
    example_diamond()
    example_branch()
    example_hook_order()
