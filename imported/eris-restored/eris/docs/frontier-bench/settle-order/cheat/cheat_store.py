"""A deliberate cheating attempt. Never executed by the harness.

The shortcut worth trying is to keep the shipped eager engine — which is already
correct on values — and buy back the timed run by refusing to evaluate a formula
until somebody asks for it, while leaving everything else alone: no change
cutoff, dependencies never pruned, hooks run in whatever order they are found.

It passes the timed run, because idle cells are genuinely never evaluated. It
scores 0 anyway:

  * `cutoff-absorbs` and `cutoff-stops-chain` expect a hook not to run and the
    rest of a chain not to be evaluated when a formula comes out equal. This
    runs the hook and evaluates the chain, so the evaluation counts differ.
  * `dynamic-deps` writes to a cell only the untaken branch reads. This still
    has it registered as a dependency, so the formula is evaluated an extra
    time and the hook runs an extra time.
  * `hook-creation-order` needs h1, h2, h3. This produces them in dependency
    discovery order.
  * `peek-no-dep` treats peek as a tracked read, so the write reaches the
    formula that should have been insulated from it.

Every one of those is a separate assertion and the suite is all-or-nothing, so
covering the timed run alone gets nowhere. Both hills have to be climbed, which
is the point of the task.

Copy over /app/src/store.py to try it.
"""

INPUT = 0
FORMULA = 1
HOOK = 2


class DisposedCell(Exception):
    pass


class UnstableGraph(Exception):
    pass


class _Node:
    __slots__ = ("cid", "kind", "value", "fn", "sources", "disposed", "cleanup", "stale")

    def __init__(self, cid, kind, value=None, fn=None):
        self.cid = cid
        self.kind = kind
        self.value = value
        self.fn = fn
        self.sources = {}
        self.disposed = False
        self.cleanup = None
        self.stale = True


class Store:
    """Lazy enough to be fast, wrong everywhere else."""

    def __init__(self):
        self._nodes = {}
        self._next = 1
        self._stack = []
        self._depth = 0
        self._dirty = set()

    def input(self, value):
        cid = self._next
        self._next += 1
        self._nodes[cid] = _Node(cid, INPUT, value=value)
        return cid

    def formula(self, fn):
        cid = self._next
        self._next += 1
        self._nodes[cid] = _Node(cid, FORMULA, fn=fn)
        return cid

    def hook(self, fn):
        cid = self._next
        self._next += 1
        node = _Node(cid, HOOK, fn=fn)
        self._nodes[cid] = node
        self._evaluate(node)
        return cid

    def get(self, cid):
        node = self._node(cid)
        if node.kind == FORMULA and node.stale:
            self._evaluate(node)
        if self._stack:
            self._stack[-1].sources[node.cid] = None
        return node.value

    # peek is treated as an ordinary read - one of the ways this is wrong.
    peek = get

    def set(self, cid, value):
        node = self._node(cid)
        node.value = value
        self._mark(node)
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

    def dispose(self, cid):
        self._node(cid).disposed = True

    def _node(self, cid):
        node = self._nodes.get(cid)
        if node is None:
            raise KeyError(cid)
        if node.disposed:
            raise DisposedCell(cid)
        return node

    def _mark(self, node):
        for other in self._nodes.values():
            if other.disposed or node.cid not in other.sources:
                continue
            if other.kind == FORMULA:
                other.stale = True
                self._mark(other)
            else:
                self._dirty.add(other.cid)

    def _evaluate(self, node):
        self._stack.append(node)
        try:
            result = node.fn(self)
        finally:
            self._stack.pop()
        if node.kind == FORMULA:
            node.value = result
            node.stale = False

    def _settle(self):
        rounds = 0
        while self._dirty:
            rounds += 1
            if rounds > 100:
                raise UnstableGraph("graph did not settle")
            due = list(self._dirty)
            self._dirty.clear()
            for cid in due:
                node = self._nodes.get(cid)
                if node is not None and not node.disposed:
                    self._evaluate(node)
