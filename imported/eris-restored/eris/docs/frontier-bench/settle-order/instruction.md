# The store does way too much work on every write

Quick background on how the app works. Screen state sits in a small store in
memory. Some cells just hold a value. Other cells are formulas built on top of
those cells. Hooks sit at the edge and run side effects when a cell they read
changes.

The engine is correct. It always has been. It is just slow.

Here is the reason. Every write recomputes the whole store. Take a screen with a
few thousand cells on it. Most of them sit in panels that are closed and nobody
is looking at. We still pay for all of them on every keystroke.

There is a second problem. Hooks fire when nothing they can see has really
changed. A formula recomputes, gets back the same value it already had, and that
still counts as a change on the way down.

Rewrite the engine so it only does the work the change deserves.

Where everything is:

* The engine is all in `/app/src/store.py`. That is the only file you change.
* The rules are written out in full in `/app/README.md`. Read it first. Some of
  them are the opposite of what an engine like this normally does, and any one of
  them is enough to fail you on its own.
* `/app/harness.py` is a scratch file. Use it to drive the store and watch what
  it does.
* Keep the class named `Store`. Keep every method name and every signature the
  same.
* Standard library only. No outside packages.

## The hard part

First rule. A formula is evaluated only when something needs its value. If
nothing reads a cell and no hook depends on it, it never gets evaluated at all.
It does not matter how many writes land above it. So you cannot just keep the
whole store fresh and hope the reads come out cheap.

Second rule. Say a formula runs and comes back with the same value it already
had. Then nothing under it has changed, and nothing under it is allowed to be
evaluated because of it.

Put those two together and that is the job. You cannot work out ahead of time
which cells need evaluating. It depends on results you do not have yet, and some
of those results will come back unchanged and stop the walk right there. So
marking everything below a write and recomputing it is wrong two ways over. It
evaluates cells nobody asked for, and the cost grows with the size of the store.

Third rule. A formula must never see a half updated world. Two of its sources
might trace back to the same input cell. It must not get the new value through
one and the old value through the other. And it must not be run a second time to
clean that up.

The rest of them:

* Dependencies are whatever the last run actually read. If a formula took one
  branch, writes to cells only the other branch reads have to stop waking it.
* `peek` never creates a dependency.
* Hooks run in the order they were created, not the order you found out they were
  stale.
* A hook's cleanup runs right before that hook runs again. It runs again when the
  hook is disposed.
* Disposal is permanent.
* Inside `batch`, reads are up to date, but hooks wait until the end.
* A hook is allowed to write. That starts another pass. If things have not
  settled after 100 passes, raise `UnstableGraph`.

## How we mark it

We drive your store from outside, on programs you have not seen. Then we compare
everything it does against a reference. Every hook run. Every cleanup. Every read
and the value it handed back. Every disposal. Every exception class. All of it in
order.

We also count how many times each formula cell was evaluated. People skim past
that one. It is usually what decides pass or fail, so read it twice.

Both have to match exactly. There is no partial credit. A rule you get right nine
times out of ten scores the same as a rule you never wrote.

There is a timed run as well. It builds ten thousand formula cells that nothing
reads and no hook depends on. It writes to the input five thousand times. Then it
reads three cells at the end, and those three have to come back right. The limit
is twenty five seconds. An engine that only evaluates what it needs is done in
well under a second.

Only `/app/src/store.py` gets read back. Anything else you make is scratch.

You have 10800 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
