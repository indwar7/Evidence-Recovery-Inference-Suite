"""Palimpsest reference solution.

Three ideas, each measurable on its own:

1. TRAIN THE OBJECTIVE YOU ARE SCORED ON.  Every snapshot has exactly one
   parent, drawn from {the other snapshots of its item} u {ROOT}. That is an
   (n+1)-way classification per snapshot, so the loss is a cross-entropy over
   candidates *within the item* -- not an independent binary label per pair.

2. SCORE ORDERED PAIRS ASYMMETRICALLY.  The features for (u -> v) are the
   token-level diff: what v ADDS relative to u and what it REMOVES. Swapping u
   and v swaps the two bags, so the model can learn that revisions tend to add
   listings, prices and links and to remove dead entries and typos.

3. DECODE WITH A MATCHING, NOT AN ARGMAX.  Each lineage is a chain, so every
   snapshot has AT MOST ONE child. Choosing parents is therefore an assignment
   problem: children on one side, candidate parents (plus a few ROOT slots) on
   the other, solved optimally with the Hungarian algorithm. Any residual cycle
   is broken by detaching its weakest edge to ROOT.
"""
import re, math, json, random, time
from collections import Counter
import numpy as np

TOK = re.compile(r"[A-Za-z0-9_']+|[^\sA-Za-z0-9]")
DIM = 1 << 18
MAX_TOK = 128
N_NUM = 12


def tokenize(t):
    return TOK.findall(t or "")


def hidx(tok, slot):
    return (hash((slot, tok)) & 0x7FFFFFFF) % DIM


def pair_features(cu, cv, lu, lv):
    """Diff features for the ordered pair u -> v."""
    added = cv - cu
    removed = cu - cv
    na, nr = sum(added.values()), sum(removed.values())
    inter = sum((cu & cv).values())
    union = sum((cu | cv).values()) or 1
    lu, lv = max(1, lu), max(1, lv)
    num = np.array([
        math.log1p(na), math.log1p(nr),
        (na - nr) / (lu + lv),
        inter / union,
        math.log1p(lu), math.log1p(lv),
        (lv - lu) / lu,
        na / lu, nr / lu,
        1.0 if na > nr else 0.0,
        1.0 if lv > lu else 0.0,
        abs(lv - lu) / (lu + lv),
    ], dtype=np.float32)
    a_ids = [hidx(t, 0) for t, _ in added.most_common(MAX_TOK)] or [0]
    r_ids = [hidx(t, 1) for t, _ in removed.most_common(MAX_TOK)] or [0]
    return a_ids, r_ids, num


def featurise_item(sids, texts, truth=None):
    """All ordered pairs of one item, plus the per-snapshot root features."""
    toks = [tokenize(t) for t in texts]
    counts = [Counter(t) for t in toks]
    lens = [len(t) for t in toks]
    n = len(sids)
    A, R, NUM, PAIRS = [], [], [], []
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            a, r, num = pair_features(counts[u], counts[v], lens[u], lens[v])
            A.append(a); R.append(r); NUM.append(num); PAIRS.append((u, v))
    d = dict(n=n, sids=sids, pairs=PAIRS, A=A, R=R,
             NUM=np.array(NUM, dtype=np.float32),
             sizes=np.array([len(t) for t in texts], dtype=np.float32))
    if truth is not None:
        idx = {s: i for i, s in enumerate(sids)}
        d["target"] = np.array([n if truth[s] == "ROOT" else idx[truth[s]]
                                for s in sids], dtype=np.int64)   # n == ROOT class
    return d


# ------------------------------------------------------------------ model
def make_model(torch, nn, dim=128):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.ea = nn.EmbeddingBag(DIM, dim, mode="mean")
            self.er = nn.EmbeddingBag(DIM, dim, mode="mean")
            self.num = nn.Sequential(nn.Linear(N_NUM, 96), nn.GELU(), nn.Linear(96, 96))
            self.head = nn.Sequential(
                nn.LayerNorm(dim * 2 + 96), nn.Linear(dim * 2 + 96, 384), nn.GELU(),
                nn.Dropout(0.10), nn.Linear(384, 192), nn.GELU(), nn.Linear(192, 1))
            # ROOT score: a learned function of the snapshot's own statistics
            self.root = nn.Sequential(nn.Linear(3, 64), nn.GELU(),
                                      nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

        def forward(self, a_in, a_off, r_in, r_off, num):
            va, vr = self.ea(a_in, a_off), self.er(r_in, r_off)
            return self.head(torch.cat([va, vr, self.num(num)], -1)).squeeze(-1)

        def root_score(self, feats):
            return self.root(feats).squeeze(-1)
    return Net()


def pack(torch, ds):
    a_in, a_off, r_in, r_off = [], [], [], []
    for d in ds:
        for a, r in zip(d["A"], d["R"]):
            a_off.append(len(a_in)); a_in += a
            r_off.append(len(r_in)); r_in += r
    num = np.vstack([d["NUM"] for d in ds])
    return (torch.tensor(a_in, dtype=torch.long), torch.tensor(a_off, dtype=torch.long),
            torch.tensor(r_in, dtype=torch.long), torch.tensor(r_off, dtype=torch.long),
            torch.tensor(num))


def root_feats(torch, d):
    """Per-snapshot features for the ROOT head: relative size and rank."""
    s = d["sizes"]
    z = (s - s.mean()) / (s.std() + 1e-6)
    rank = np.argsort(np.argsort(s)) / max(1, len(s) - 1)
    inv_n = np.full(len(s), 1.0 / len(s), dtype=np.float32)
    return torch.tensor(np.stack([z, rank, inv_n], axis=1).astype(np.float32))


def item_logits(torch, net, d, device):
    """(n, n+1) matrix: logits[v, u] = score that u is v's parent; column n is ROOT."""
    a_in, a_off, r_in, r_off, num = [t.to(device) for t in pack(torch, [d])]
    flat = net(a_in, a_off, r_in, r_off, num)
    n = d["n"]
    M = torch.full((n, n + 1), -1e4, device=device)
    pu = torch.tensor([p[0] for p in d["pairs"]], device=device)
    pv = torch.tensor([p[1] for p in d["pairs"]], device=device)
    M = M.index_put((pv, pu), flat)
    M[:, n] = net.root_score(root_feats(torch, d).to(device))
    return M


def train(data, epochs=12, lr=2e-3, dim=128, seed=0, device=None,
          batch_items=6, log=print, val=None):
    import torch, torch.nn as nn
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else
                     ("mps" if torch.backends.mps.is_available() else "cpu"))
    net = make_model(torch, nn, dim=dim).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * (len(data) // batch_items + 1)))
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        order = list(range(len(data))); random.shuffle(order)
        tot = nb = 0
        for k in range(0, len(order), batch_items):
            chunk = [data[i] for i in order[k:k + batch_items]]
            opt.zero_grad()
            loss = 0.0
            for d in chunk:
                M = item_logits(torch, net, d, dev)
                loss = loss + lossf(M, torch.tensor(d["target"], device=dev))
            loss = loss / len(chunk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
            tot += float(loss.detach()); nb += 1
        msg = f"  epoch {ep+1}/{epochs}  loss={tot/max(1,nb):.4f}  {time.time()-t0:.0f}s"
        if val is not None and (ep + 1) % 3 == 0:
            msg += f"  val_parent_acc={evaluate(net, val, dev):.4f}"
        log(msg)
    return net, dev


# ------------------------------------------------------------------ decode
def decode_matching(M, max_roots=6):
    """M: (n, n+1) score matrix, column n is ROOT. Returns parent index per
    snapshot (n == ROOT). Each real parent may take at most one child, so this
    is an assignment problem; ROOT is duplicated into `max_roots` columns."""
    from scipy.optimize import linear_sum_assignment
    n = M.shape[0]
    k = min(max_roots, n)
    C = np.concatenate([M[:, :n], np.tile(M[:, n:n + 1], (1, k))], axis=1)
    np.fill_diagonal(C[:, :n], -1e6)
    rows, cols = linear_sum_assignment(-C)
    parent = {}
    for r, c in zip(rows, cols):
        parent[int(r)] = int(c) if c < n else "ROOT"
    return break_cycles(parent, M, n)


def break_cycles(parent, M, n):
    """Detach the weakest edge of any cycle to ROOT."""
    for _ in range(n + 1):
        cyc = _find_cycle(parent, n)
        if not cyc:
            break
        weakest = min(cyc, key=lambda v: M[v, parent[v]] if parent[v] != "ROOT" else 1e9)
        parent[weakest] = "ROOT"
    return parent


def _find_cycle(parent, n):
    state = {}
    for start in range(n):
        path, node = [], start
        while True:
            st = state.get(node)
            if st == 1:
                break
            if st == 0:
                return path[path.index(node):]
            state[node] = 0
            path.append(node)
            nxt = parent.get(node, "ROOT")
            if nxt == "ROOT":
                break
            node = nxt
        for v in path:
            state[v] = 1
    return None


def decode_greedy(M):
    n = M.shape[0]
    parent = {}
    for v in range(n):
        row = M[v].copy()
        row[v] = -1e9
        j = int(np.argmax(row))
        parent[v] = "ROOT" if j == n else j
    return break_cycles(parent, M, n)


def predict(net, data, device, decoder=decode_matching):
    import torch
    net.eval()
    out = {}
    with torch.no_grad():
        for d in data:
            M = item_logits(torch, net, d, device).cpu().numpy().astype(np.float64)
            par = decoder(M)
            out[d["item_id"]] = {
                d["sids"][v]: ("ROOT" if p == "ROOT" else d["sids"][p])
                for v, p in par.items()}
    return out


def evaluate(net, data, device, decoder=decode_matching):
    preds = predict(net, data, device, decoder)
    ok = tot = 0
    for d in data:
        p = preds[d["item_id"]]
        for i, sid in enumerate(d["sids"]):
            tgt = d["target"][i]
            true = "ROOT" if tgt == d["n"] else d["sids"][tgt]
            ok += int(p[sid] == true); tot += 1
    return ok / max(1, tot)
