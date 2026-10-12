# The view store — how recalculation is supposed to behave

This is the full specification. Everything the store is graded on is written
here. There is nothing else to find.

## The three kinds of cell

**Input cells** hold a value that the application writes directly.

**Formula cells** hold a value produced by a function of other cells. The
function is given the store and reads whatever it needs through `get`.

**Hooks** exist for their side effects — patching the screen, writing to a log.
A hook's function is given the store, reads whatever it needs, and returns
either nothing or a single callable to be run before the next time it runs.

## The API, which must not change

```python
store.input(value)   -> cid      create an input cell
store.formula(fn)    -> cid      create a formula cell; fn(store) -> value
store.hook(fn)       -> cid      create a hook; fn(store) -> None | callable
store.get(cid)       -> value    read, and depend on it if read during evaluation
store.peek(cid)      -> value    read without depending on it, ever
store.set(cid, v)    -> None     write an input cell
store.batch(fn)      -> None     run fn(store) with hooks held until it finishes
store.dispose(cid)   -> None     retire a cell or hook permanently
```

Two exception classes, both raised by name: `DisposedCell` and `UnstableGraph`.

Keep the class name `Store`, every method name, and every signature exactly as
they are. Standard library only.

## The rules

**1 — Nothing is computed until it is needed.** A formula cell is evaluated only
when its value is actually required: when something reads it, or when a hook
that depends on it has to run. A formula cell that nothing reads and no hook
depends on is never evaluated, not even once, no matter how many writes land
upstream of it. Creating a formula cell does not evaluate it.

A hook is different: it runs once at the moment it is created, so that it knows
what it depends on.

**2 — A cell never sees a half-updated world.** When a formula is evaluated,
every cell it reads gives the value it is supposed to have at that moment. If
two of its sources both trace back to the same input, it must not see the new
value through one and the old value through the other.

**3 — An equal result is not a change.** After a formula is evaluated, compare
the result with the value it held before, using `==`. If they are equal, then as
far as everything downstream is concerned nothing happened: cells that read this
one are not evaluated because of it, and hooks that depend on it do not run
because of it. If the comparison itself raises, treat the result as changed.

Writing an input cell a value equal to the one it already holds is likewise not
a change, and must cause no evaluation and no hook run at all.

**4 — Each thing happens at most once per pass.** Within a single settle, and as
long as no hook writes, a formula cell is evaluated at most once and a hook runs
at most once — however many paths reach them.

**5 — Dependencies are what was read last time.** A cell depends on exactly the
cells its most recent evaluation read through `get`. A cell that an earlier
evaluation read but the latest one did not no longer affects it. A formula that
takes one branch this time must not be woken by writes to cells only the other
branch reads.

**6 — `peek` never creates a dependency.** It returns the same value `get` would
return at that moment, including evaluating the cell if it is stale, but reading
through it never makes the reader depend on it.

**7 — Hooks run in the order they were created.** When several hooks have to run
in the same pass, they run in creation order, regardless of the shape of the
graph or the order in which they were found to be stale.

**8 — Cleanup runs first.** If a hook returned a callable, that callable is run
immediately before the hook's function runs again, and again when the hook is
disposed. It runs once per hook run, never twice for the same run.

**9 — Hooks may write.** A hook's function may call `set`. Writes made during a
pass do not interrupt it; they open another pass once the current one finishes,
and the store keeps going until no hook is left to run. If that has not happened
after 100 passes, raise `UnstableGraph`.

**10 — Disposal is permanent.** A disposed cell is never evaluated again and
never causes anything else to be evaluated. `get`, `peek`, `set`, and `dispose`
on a disposed cell all raise `DisposedCell`. Disposing a hook runs its pending
cleanup one last time.

**11 — Batching holds hooks, not values.** Inside `batch`, writes are applied but
no hook runs until the outermost batch finishes. Reads inside a batch still
return up-to-date values — rule 2 holds inside a batch exactly as outside it.

**12 — Settling costs what the change costs.** The work of a write must follow
the cells the change actually reaches, not the number of cells in the store. A
write that is absorbed near the top by rule 3 must not walk the cells beyond
that point. The store holds tens of thousands of cells and is written to
constantly; a settle that scans the whole graph is a bug even when its answers
are right, and there is a timed run built to catch it.

## Where this bites

Rules 1, 2 and 3 pull against each other, and that is the whole problem.

Rule 3 means you cannot know whether a cell needs evaluating until the cells it
reads have been resolved — and resolving one of those may end in "no change",
which means this cell does not need evaluating after all. So staleness is not a
property you can stamp on the graph at write time and then act on. Marking
everything downstream and recomputing it all satisfies rules 2 and 3's letter
while breaking rule 1 and rule 12: it evaluates cells nothing needed, and it
does work proportional to the graph.

Rule 2 is what stops you from fixing that with a simple recursive pull that
recomputes a cell whenever any source is stale — down a diamond that reaches the
same input twice, that evaluates the shared cell more than once and can hand the
two branches values from different moments, which breaks rule 4 as well.

A chain where each cell reads the one above it, and a write at the top comes out
equal at the second cell, must stop there. Everything below it is untouched and
must not be looked at.
