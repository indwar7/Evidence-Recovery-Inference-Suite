"""Sealed reference implementation.

Root-readable only. The account that runs submitted code cannot read this file.

The core is that "which cells need recomputing" is not known up front. A formula
whose value comes out equal to what it already was does not count as a change,
so a cell three levels downstream may or may not need work depending on results
that are not known until the walk has already started. This is handled with two
grades of staleness:

  DIRTY  - a source definitely changed; this cell must be recomputed.
  CHECK  - something upstream *may* have changed; walk the sources first and
           find out. If none of them actually reported a change, this cell is
           clean and is never recomputed.

A write marks its direct observers DIRTY and everything further downstream
CHECK, and stops descending as soon as it reaches a cell that is already marked
(that cell's observers were marked when it was first marked). Recomputation
happens only on demand: reading a cell, or a hook that needs to run. A formula
that recomputes to an equal value notifies nobody, so the CHECK walk downstream
of it resolves to clean and stops. That is what keeps a settle proportional to
the cells actually affected instead of the size of the graph.
"""

CLEAN = 0
CHECK = 1
DIRTY = 2

INPUT = 0
FORMULA = 1
HOOK = 2

MAX_ROUNDS = 100


class DisposedCell(Exception):
    """Raised when a disposed cell is read, written, or disposed again."""


class UnstableGraph(Exception):
    """Raised when hook writes keep the graph settling past the round limit."""


class _Node:
    __slots__ = ("cid", "kind", "value", "fn", "state", "sources", "observers",
                 "disposed", "cleanup", "retrigger")

    def __init__(self, cid, kind, value=None, fn=None):
        self.cid = cid
        self.kind = kind
        self.value = value
        self.fn = fn
        self.state = CLEAN
        self.sources = {}
        self.observers = {}
        self.disposed = False
        self.cleanup = None
        self.retrigger = False


def _same(a, b):
    """Equality as the spec defines it: `==`, and anything that raises is a change."""
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:
        return False


class Store:
    def __init__(self):
        self._nodes = {}
        self._next = 1
        self._stack = []
        self._pending = {}
        self._depth = 0
        self._settling = False

    # ------------------------------------------------------------------ build

    def input(self, value):
        cid = self._next
        self._next += 1
        self._nodes[cid] = _Node(cid, INPUT, value=value)
        return cid

    def formula(self, fn):
        cid = self._next
        self._next += 1
        self._nodes[cid] = _Node(cid, FORMULA, fn=fn)
        # Never evaluated at creation: a formula nobody needs is never run.
        self._nodes[cid].state = DIRTY
        return cid

    def hook(self, fn):
        cid = self._next
        self._next += 1
        node = _Node(cid, HOOK, fn=fn)
        node.state = DIRTY
        self._nodes[cid] = node
        # A hook runs once when created, to establish what it depends on. That
        # first run goes through the same settle as everything else, so a hook
        # created inside a batch or inside another evaluation waits its turn.
        self._pending[cid] = None
        self._maybe_settle()
        return cid

    # ------------------------------------------------------------------- read

    def get(self, cid):
        node = self._node(cid)
        if node.kind != INPUT:
            self._update(node)
        if self._stack:
            cur = self._stack[-1]
            cur.sources[node.cid] = None
            node.observers[cur.cid] = None
        return node.value

    def peek(self, cid):
        node = self._node(cid)
        if node.kind != INPUT:
            self._update(node)
        return node.value

    # ------------------------------------------------------------------ write

    def set(self, cid, value):
        node = self._node(cid)
        if node.kind != INPUT:
            raise TypeError("only input cells can be set")
        if _same(node.value, value):
            return
        node.value = value
        # A write that reaches a cell which is evaluating right now invalidates
        # the value that evaluation just observed, so that cell has to run
        # again. Change propagation during a pull is different: there the cell
        # is reading the new value as it goes, and must not be re-run.
        live = {n.cid for n in self._stack}
        for oid in list(node.observers):
            self._notify(self._nodes[oid], DIRTY, live)
        self._maybe_settle()

    def batch(self, fn):
        self._depth += 1
        try:
            fn(self)
        finally:
            self._depth -= 1
        self._maybe_settle()

    def _maybe_settle(self):
        # A write from inside a hook body or a formula must not start its own
        # settle: it joins the one already running, or the batch already open.
        if self._depth == 0 and not self._settling and not self._stack:
            self._settle()

    # ---------------------------------------------------------------- dispose

    def dispose(self, cid):
        node = self._node(cid)
        node.disposed = True
        self._pending.pop(cid, None)
        for sid in list(node.sources):
            src = self._nodes.get(sid)
            if src is not None:
                src.observers.pop(cid, None)
        node.sources = {}
        for oid in list(node.observers):
            obs = self._nodes.get(oid)
            if obs is not None:
                obs.sources.pop(cid, None)
        node.observers = {}
        if node.kind == HOOK and node.cleanup is not None:
            cb = node.cleanup
            node.cleanup = None
            cb()

    # --------------------------------------------------------------- internal

    def _node(self, cid):
        node = self._nodes.get(cid)
        if node is None:
            raise KeyError(cid)
        if node.disposed:
            raise DisposedCell(cid)
        return node

    def _notify(self, node, state, live=()):
        # Iterative: chains can be deeper than the interpreter's recursion limit.
        # `live` holds the cells that are mid-evaluation, which only happens for
        # a write made from inside a hook body. That case walks the whole cone
        # so it cannot miss one; the ordinary case short-circuits at the first
        # already-stale cell, which is what keeps a write proportional to what
        # it actually reaches.
        seen = set() if live else None
        stack = [(node, state)]
        while stack:
            cur, st = stack.pop()
            if cur.disposed:
                continue
            if live:
                if cur.cid in seen:
                    continue
                seen.add(cur.cid)
                if cur.cid in live:
                    cur.retrigger = True
                    if cur.kind == HOOK:
                        self._pending[cur.cid] = None
                descend = True
            else:
                if cur.state >= st:
                    continue
                # Observers were already marked when this cell first became
                # stale, so only a clean -> stale transition has to descend.
                descend = cur.state == CLEAN
            if cur.state < st:
                cur.state = st
                if cur.kind == HOOK:
                    self._pending[cur.cid] = None
            if descend:
                for oid in cur.observers:
                    obs = self._nodes.get(oid)
                    if obs is not None and not obs.disposed:
                        stack.append((obs, CHECK))

    def _update(self, node):
        if node.disposed or node.state == CLEAN:
            return
        if node.state == CHECK:
            for sid in list(node.sources):
                src = self._nodes.get(sid)
                if src is None or src.disposed:
                    continue
                if src.kind != INPUT:
                    self._update(src)
                if node.state == DIRTY:
                    break
            if node.state == CHECK:
                node.state = CLEAN
                return
        if node.state == DIRTY:
            self._recompute(node)

    def _recompute(self, node):
        if node.kind == HOOK and node.cleanup is not None:
            cb = node.cleanup
            node.cleanup = None
            cb()

        old_sources = node.sources
        node.sources = {}
        node.retrigger = False
        self._stack.append(node)
        try:
            result = node.fn(self)
        finally:
            self._stack.pop()

        # Dependencies are exactly what this evaluation read. Anything read last
        # time but not this time stops affecting the cell.
        for sid in old_sources:
            if sid not in node.sources:
                src = self._nodes.get(sid)
                if src is not None:
                    src.observers.pop(node.cid, None)

        if node.retrigger:
            node.retrigger = False
            node.state = DIRTY
            if node.kind == HOOK:
                self._pending[node.cid] = None
        else:
            node.state = CLEAN

        if node.kind == FORMULA:
            old = node.value
            node.value = result
            if not _same(old, result):
                for oid in list(node.observers):
                    obs = self._nodes.get(oid)
                    if obs is not None:
                        self._notify(obs, DIRTY)
        else:
            node.cleanup = result if callable(result) else None

    def _settle(self):
        self._settling = True
        try:
            rounds = 0
            while self._pending:
                rounds += 1
                if rounds > MAX_ROUNDS:
                    raise UnstableGraph("graph did not settle")
                due = sorted(self._pending)
                self._pending = {}
                for cid in due:
                    node = self._nodes.get(cid)
                    if node is None or node.disposed:
                        continue
                    self._update(node)
        finally:
            self._settling = False
