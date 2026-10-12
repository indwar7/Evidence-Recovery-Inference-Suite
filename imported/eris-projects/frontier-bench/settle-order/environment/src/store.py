"""Recalculation engine for the view store.

This is the version that shipped. It keeps the screen correct, which is why it
survived review, but it does far more work than it needs to and it gets several
of the rules in README.md wrong.

How it currently works: every write marks the whole graph stale, then every
formula cell is recomputed in dependency order, then every hook that has ever
depended on anything that was recomputed is run. It is simple and it is always
up to date. It is also why the profiler shows the settle cost tracking the size
of the graph rather than the size of the change.
"""

INPUT = 0
FORMULA = 1
HOOK = 2

MAX_ROUNDS = 100


class DisposedCell(Exception):
    """Raised when a disposed cell is read, written, or disposed again."""


class UnstableGraph(Exception):
    """Raised when hook writes keep the graph settling past the round limit."""


class _Node:
    __slots__ = ("cid", "kind", "value", "fn", "sources", "disposed", "cleanup")

    def __init__(self, cid, kind, value=None, fn=None):
        self.cid = cid
        self.kind = kind
        self.value = value
        self.fn = fn
        self.sources = {}
        self.disposed = False
        self.cleanup = None


class Store:
    def __init__(self):
        self._nodes = {}
        self._next = 1
        self._stack = []
        self._depth = 0
        self._dirty = False

    # ------------------------------------------------------------------ build

    def input(self, value):
        cid = self._next
        self._next += 1
        self._nodes[cid] = _Node(cid, INPUT, value=value)
        return cid

    def formula(self, fn):
        cid = self._next
        self._next += 1
        node = _Node(cid, FORMULA, fn=fn)
        self._nodes[cid] = node
        self._evaluate(node)
        return cid

    def hook(self, fn):
        cid = self._next
        self._next += 1
        node = _Node(cid, HOOK, fn=fn)
        self._nodes[cid] = node
        self._evaluate(node)
        return cid

    # ------------------------------------------------------------------- read

    def get(self, cid):
        node = self._node(cid)
        if self._stack:
            self._stack[-1].sources[node.cid] = None
        return node.value

    def peek(self, cid):
        node = self._node(cid)
        return node.value

    # ------------------------------------------------------------------ write

    def set(self, cid, value):
        node = self._node(cid)
        if node.kind != INPUT:
            raise TypeError("only input cells can be set")
        node.value = value
        self._dirty = True
        if self._depth == 0:
            self._settle()

    def batch(self, fn):
        self._depth += 1
        try:
            fn(self)
        finally:
            self._depth -= 1
        if self._depth == 0:
            self._settle()

    # ---------------------------------------------------------------- dispose

    def dispose(self, cid):
        node = self._node(cid)
        node.disposed = True

    # --------------------------------------------------------------- internal

    def _node(self, cid):
        node = self._nodes.get(cid)
        if node is None:
            raise KeyError(cid)
        if node.disposed:
            raise DisposedCell(cid)
        return node

    def _evaluate(self, node):
        self._stack.append(node)
        try:
            result = node.fn(self)
        finally:
            self._stack.pop()
        if node.kind == FORMULA:
            node.value = result

    def _order(self):
        """Every formula, in an order where sources come before the cells that
        read them. Recomputed from scratch on every settle."""
        formulas = [n for n in self._nodes.values()
                    if n.kind == FORMULA and not n.disposed]
        done = set()
        out = []

        def visit(node):
            if node.cid in done:
                return
            done.add(node.cid)
            for sid in list(node.sources):
                src = self._nodes.get(sid)
                if src is not None and src.kind == FORMULA and not src.disposed:
                    visit(src)
            out.append(node)

        for node in formulas:
            visit(node)
        return out

    def _settle(self):
        rounds = 0
        while self._dirty:
            rounds += 1
            if rounds > MAX_ROUNDS:
                raise UnstableGraph("graph did not settle")
            self._dirty = False
            for node in self._order():
                self._evaluate(node)
            for node in list(self._nodes.values()):
                if node.kind == HOOK and not node.disposed:
                    self._evaluate(node)
