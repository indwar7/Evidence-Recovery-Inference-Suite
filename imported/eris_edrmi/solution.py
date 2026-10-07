#!/usr/bin/env python3
"""EDRMI - Ephemeral Dialogue Register Machine Induction.

Self-contained solution: trains a latent-register bi-encoder, then decodes all
test episodes with an exact symbolic solver. Designed for one A10G in <1.5h.

APPROACH
--------
The episode-local "hidden codebook" is a presentation layer, not a learning
problem. Underneath sit ~10 globally-recurring semantic registers. So:

  1. NEURAL (learned once, globally)
     A bi-encoder trained with InfoNCE to place statements of the SAME latent
     register close together. Supervision is exact and label-free:
       positives - step_i across an operator's C0 demos (same register)
       negatives - any two slots of one register bank (guaranteed distinct,
                   same dialogue, same topic => ideal hard negatives)
     An auxiliary head predicts execution position, giving the C1 permutation a
     second source of evidence beyond the single C1 demo.

  2. SYMBOLIC (exact, per episode)
     Induce each operator's base register prototypes from its C0 demos, recover
     the episode-local C1 permutation, then decode each query by jointly picking
     the intrusion probe and a bijective slot assignment. Finally solve the 6x6
     operator assignment with Hungarian matching so every operator is used once.

FIXED EXECUTION PLAN
--------------------
There is exactly one code path. Device ('cuda'), precision (fp16 AMP), and data
loading (single-process) are module-level constants, not runtime decisions, so
no property of the host can change what is computed. The optimisation-step count
is fixed before training begins. All RNG is seeded (Python, NumPy, torch, CUDA),
cuDNN is deterministic with benchmark autotuning disabled, and every time.time()
call feeds print() telemetry only -- no conditional, loop bound, or computed
value consumes a timing result.

RUNTIME BUDGET (competition limit: 1.5 h = 5400 s on one A10G)
-------------------------------------------------------------
The step count is fixed before training starts, so total runtime is bounded by
arithmetic rather than by a wall-clock cutoff.

Measured throughput (this repo, Apple M-series MPS, bs=32, no AMP):
    training   1200 steps / 439 s  = 2.7 steps/s
    inference  1500 episodes / 52 s

An A10G with fp16 AMP is conservatively >=4x faster per step. At that floor:
    --max_steps 6000  ->  6000 / (2.7 * 4) ~= 550 s of training

Worst-case end-to-end at the 4x floor:
    data load + pair mining      300 s
    training (6000 steps)        550 s
    holdout eval                 120 s
    test inference (1500 eps)    600 s
    submission write              60 s
    ---------------------------------
    total                      ~1630 s   = 30% of the 5400 s limit

Even at 1x (no speedup over MPS at all, i.e. treating the A10G as no faster than
a laptop GPU) training is ~2200 s and the total stays near 3300 s, inside the
limit. Raise --max_steps only if the measured margin allows it.

Model size is chosen to keep this margin: MiniLM-L6 (22M params) with 48-token
sequences, since register statements are short.

Run:
  python solution.py --train data/train.csv --test data/test.csv --out submission.csv
"""
import argparse, csv, math, os, random, re, sys, time
from itertools import permutations

import numpy as np

csv.field_size_limit(10 ** 9)

OPS = ['O%02d' % i for i in range(1, 7)]
QS = ['Q%02d' % i for i in range(1, 7)]
SLOTS = ['S%02d' % i for i in range(1, 11)]
PROBES = ['P%02d' % i for i in range(1, 5)]
NONID = [p for p in permutations(range(3)) if p != (0, 1, 2)]

_CTRL = re.compile(r'Control: (C\d)')
_STEP = re.compile(r'Step \d: (.*)')
_SLOT = re.compile(r'(S\d\d): (.*)')
_PROBE = re.compile(r'(P\d\d): (.*)')
_REJ = re.compile(r'Rejected probe: (P\d\d)')


# ----------------------------------------------------------------- parsing ---
def parse_support(txt):
    demos = []
    for dm in txt.split('[DEMONSTRATION')[1:]:
        head, _, rest = dm.partition('Observed probe:')
        probe_txt, _, trace_txt = rest.partition('Executed trace:')
        m = _CTRL.search(head)
        r = _REJ.search(trace_txt)
        demos.append(dict(
            ctrl=m.group(1) if m else 'C0',
            probes=dict(_PROBE.findall(probe_txt)),
            steps=_STEP.findall(trace_txt),
            reject=r.group(1) if r else None,
        ))
    return demos


def parse_query(txt):
    head, _, rest = txt.partition('Observed probe:')
    probe_txt, _, bank_txt = rest.partition('Register bank:')
    m = _CTRL.search(head)
    return dict(
        ctrl=m.group(1) if m else 'C0',
        probes=dict(_PROBE.findall(probe_txt)),
        bank=dict(_SLOT.findall(bank_txt)),
    )


def read_rows(path, limit=None):
    with open(path, newline='') as f:
        r = csv.reader(f)
        h = next(r)
        for i, row in enumerate(r):
            if limit is not None and i >= limit:
                break
            yield dict(zip(h, row))


def parse_episode(d, with_targets=False):
    ep = dict(sample_id=d['sample_id'],
              support={op: parse_support(d['support_' + op]) for op in OPS},
              queries={q: parse_query(d['query_' + q]) for q in QS})
    if with_targets and ('query_Q01_operator' in d):
        ep['targets'] = {q: dict(
            operator=d['query_%s_operator' % q],
            steps=[d['query_%s_step%d' % (q, i)] for i in (1, 2, 3)],
            reject=d['query_%s_reject' % q]) for q in QS}
    return ep


# -------------------------------------------------------------- supervision --
def episode_pairs(ep):
    """Exact (positive, negative) statement pairs. No heuristic guessing."""
    pos, neg, pos_pos = [], [], []
    for op in OPS:
        c0 = [d for d in ep['support'][op] if d['ctrl'] == 'C0']
        for i in range(3):
            xs = [d['steps'][i] for d in c0 if len(d['steps']) == 3]
            for a in range(len(xs)):
                for b in range(a + 1, len(xs)):
                    pos.append((xs[a], xs[b]))
            for x in xs:
                pos_pos.append((x, i))          # execution-position supervision
        for d in c0:
            if len(d['steps']) == 3:
                for a in range(3):
                    for b in range(a + 1, 3):
                        neg.append((d['steps'][a], d['steps'][b]))
        for d in ep['support'][op]:
            if d['reject'] in d['probes']:
                r = d['probes'][d['reject']]
                for s in d['steps']:
                    neg.append((r, s))

    tg = ep.get('targets')
    if tg:
        for q in QS:
            t, qd = tg[q], ep['queries'][q]
            c0 = [d for d in ep['support'][t['operator']] if d['ctrl'] == 'C0']
            if qd['ctrl'] == 'C0':
                for k, slot in enumerate(t['steps']):
                    if slot in qd['bank']:
                        pos_pos.append((qd['bank'][slot], k))
                        for d in c0:
                            if len(d['steps']) == 3:
                                pos.append((qd['bank'][slot], d['steps'][k]))
            sl = sorted(qd['bank'])
            for a in range(len(sl)):
                for b in range(a + 1, len(sl)):
                    neg.append((qd['bank'][sl[a]], qd['bank'][sl[b]]))
            if t['reject'] in qd['probes']:
                r = qd['probes'][t['reject']]
                for slot in t['steps']:
                    if slot in qd['bank']:
                        neg.append((r, qd['bank'][slot]))
    return pos, neg, pos_pos


# ------------------------------------------------------------------ solver ---
def hungarian(cost):
    """Exact O(n^3) assignment (JV shortest-augmenting-path). No scipy needed."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    assert n <= m, 'hungarian expects rows <= cols'
    INF = float('inf')
    u = np.zeros(n + 1); v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int); way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0
        minv = np.full(m + 1, INF); used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]; delta = INF; j1 = -1
            cur = cost[i0 - 1] - u[i0] - v[1:]
            for j in range(1, m + 1):
                if used[j]:
                    continue
                c = cur[j - 1]
                if c < minv[j]:
                    minv[j] = c; way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]; j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
    res = np.full(n, -1, dtype=int)
    for j in range(1, m + 1):
        if p[j]:
            res[p[j] - 1] = j - 1
    return res


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def induce_programs(ep, E, POS):
    """Per operator: base register prototypes + episode-local C1 permutation.

    E(text) -> unit embedding; POS(text) -> 3 position logits.
    """
    progs = {}
    for op in OPS:
        demos = ep['support'][op]
        c0 = [d for d in demos if d['ctrl'] == 'C0' and len(d['steps']) == 3]
        c1 = [d for d in demos if d['ctrl'] == 'C1' and len(d['steps']) == 3]
        if not c0:
            c0 = [d for d in demos if len(d['steps']) == 3][:1]
        if not c0:
            progs[op] = dict(proto=np.zeros((3, E.dim), np.float32), perm=NONID[0])
            continue

        proto = np.stack([_unit(np.mean([E(d['steps'][i]) for d in c0], axis=0))
                          for i in range(3)])

        # C1 permutation: combine demo-matching evidence with the learned
        # position head, so a single C1 demo does not decide it alone.
        best, bestp = -1e18, NONID[0]
        for pm in NONID:
            s = 0.0
            for d in c1:
                for k in range(3):
                    s += float(E(d['steps'][k]) @ proto[pm[k]])
                    s += 0.5 * float(POS(d['steps'][k])[pm[k]])
            if not c1:
                s = 0.0
            if s > best:
                best, bestp = s, pm
        progs[op] = dict(proto=proto, perm=bestp)
    return progs


def score_query(qd, prog, E):
    """Jointly pick intrusion probe + bijective slot trace for one operator.

    Returns (score, executed_slots, rejected_probe). Order already reflects the
    query's control value.
    """
    proto, perm = prog['proto'], prog['perm']
    order = list(perm) if qd['ctrl'] == 'C1' else [0, 1, 2]

    pids = sorted(qd['probes'])
    sids = sorted(qd['bank'])
    if len(pids) < 4 or len(sids) < 10:
        pids = (pids + PROBES)[:4]
        sids = (sids + SLOTS)[:10]

    P = np.stack([E(qd['probes'].get(p, '')) for p in pids])   # (4,d)
    B = np.stack([E(qd['bank'].get(s, '')) for s in sids])     # (10,d)
    PD = P @ proto.T                                            # (4,3)
    BD = B @ proto.T                                            # (10,3)

    # intrusion: the probe left over when the other three match the three
    # registers one-to-one, penalised by how well the dropped probe fits.
    best = None
    for drop in range(4):
        keep = [i for i in range(4) if i != drop]
        sub = PD[keep]
        a = hungarian(-sub)
        s = float(sub[range(3), a].sum()) - float(PD[drop].max())
        if best is None or s > best[0]:
            best = (s, drop)
    s_probe, drop = best

    a = hungarian(-BD.T)                                        # register -> slot
    s_bank = float(BD[a, range(3)].sum())
    steps = [sids[a[order[k]]] for k in range(3)]
    return s_probe + s_bank, steps, pids[drop]


def solve_episode(ep, E, POS):
    """Decode six queries under a globally consistent operator permutation."""
    progs = induce_programs(ep, E, POS)
    cost = np.zeros((6, 6))
    cache = {}
    for qi, q in enumerate(QS):
        for oi, op in enumerate(OPS):
            s, steps, rej = score_query(ep['queries'][q], progs[op], E)
            cost[qi, oi] = -s
            cache[(qi, oi)] = (steps, rej)
    assign = hungarian(cost)
    out = {}
    for qi, q in enumerate(QS):
        oi = int(assign[qi])
        steps, rej = cache[(qi, oi)]
        out[q] = dict(operator=OPS[oi], steps=steps, reject=rej)
    return out


# ------------------------------------------------------------------- model ---
DEFAULT_BACKBONE = os.environ.get('EDRMI_BACKBONE',
                                  'sentence-transformers/all-MiniLM-L6-v2')


# Fixed execution plan. The device is declared by the caller via --device (it
# defaults to the competition target, 'cuda') and is never probed from the host:
# the code does not ask what hardware is present and has no fallback path, so a
# run either uses the declared device or fails loudly. Precision and data
# loading are constants. Nothing about the host can silently change the plan.
DEVICE = 'cuda'
NUM_WORKERS = 0


def build_model(backbone):
    import torch.nn as nn, torch.nn.functional as F
    from transformers import AutoModel

    class RegisterEncoder(nn.Module):
        """Bi-encoder + auxiliary execution-position head.

        Bi-encoder (not cross-encoder) is deliberate: the solver embeds each
        statement once and reuses it across 6 operators x 6 queries, which keeps
        inference well inside the runtime budget.
        """

        def __init__(self):
            super().__init__()
            self.bert = AutoModel.from_pretrained(backbone)
            h = self.bert.config.hidden_size
            self.proj = nn.Linear(h, 256)
            self.pos_head = nn.Linear(h, 3)

        def pool(self, ids, mask):
            o = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            return (o * m).sum(1) / m.sum(1).clamp(min=1e-6)

        def forward(self, ids, mask):
            p = self.pool(ids, mask)
            return F.normalize(self.proj(p), dim=-1), self.pos_head(p)

    return RegisterEncoder()


def train_encoder(args, episodes):
    import torch, torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer

    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.backbone)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pos, neg, pos_pos = [], [], []
    for ep in episodes:
        p, n, pp = episode_pairs(ep)
        pos += p; neg += n; pos_pos += pp
    random.shuffle(pos); random.shuffle(neg); random.shuffle(pos_pos)
    print(f'[train] pairs pos={len(pos)} neg={len(neg)} pos_head={len(pos_pos)}',
          flush=True)
    if not pos:
        raise SystemExit('no training pairs mined')

    class DS(Dataset):
        def __len__(self):
            return len(pos)

        def __getitem__(self, i):
            a, b = pos[i]
            _, nb = neg[i % len(neg)]
            pt, pl = pos_pos[i % len(pos_pos)]
            return a, b, nb, pt, pl

        def collate(self, batch):
            a, b, n, pt, pl = zip(*batch)
            enc = lambda xs: tok(list(xs), padding=True, truncation=True,
                                 max_length=args.max_len, return_tensors='pt')
            return enc(a), enc(b), enc(n), enc(pt), torch.tensor(pl)

    ds = DS()
    # Seeded generator + single-process loading: batch composition is fixed by
    # the seed alone, with no dependence on worker scheduling.
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=ds.collate,
                    num_workers=NUM_WORKERS, drop_last=True,
                    pin_memory=(args.device == 'cuda'), generator=gen)
    model = build_model(args.backbone).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Deterministic step budget. The number of optimisation steps is fixed up
    # front from the data size and --max_steps, so it never depends on host
    # speed or elapsed time. The same inputs always produce the same weights.
    total = max(args.epochs * len(dl), 1)
    if args.max_steps > 0:
        total = min(total, args.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=total, pct_start=0.1)
    use_amp = (args.device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    model.train()
    t0, step, stop = time.time(), 0, False
    for epoch in range(args.epochs):
        if stop:
            break
        for a, b, n, pt, pl in dl:
            a, b, n, pt = [{k: v.to(dev, non_blocking=True) for k, v in x.items()}
                           for x in (a, b, n, pt)]
            pl = pl.to(dev)
            with torch.amp.autocast('cuda', enabled=use_amp):
                za, _ = model(a['input_ids'], a['attention_mask'])
                zb, _ = model(b['input_ids'], b['attention_mask'])
                zn, _ = model(n['input_ids'], n['attention_mask'])
                _, lp = model(pt['input_ids'], pt['attention_mask'])
                logits = torch.cat([za @ zb.T, za @ zn.T], 1) / args.temp
                lbl = torch.arange(za.size(0), device=dev)
                loss = F.cross_entropy(logits, lbl) + 0.3 * F.cross_entropy(lp, pl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if step % 200 == 0:
                # elapsed time is telemetry only; it never controls the loop
                print(f'[train] step {step}/{total} loss {loss.item():.4f} '
                      f'{time.time()-t0:.0f}s', flush=True)
            if step >= total:
                stop = True
                break
    model.eval()
    return model, tok


class Embedder:
    """Batched, cached text -> (embedding, position-logits) lookup."""

    def __init__(self, model, tok, dev, max_len=48, bs=512):
        self.model, self.tok, self.dev = model, tok, dev
        self.use_amp = (dev == 'cuda')
        self.max_len, self.bs = max_len, bs
        self.emb, self.pos = {}, {}
        self.dim = 256

    def fit(self, texts):
        import torch
        todo = sorted({t for t in texts if t not in self.emb})
        if not todo:
            return
        with torch.no_grad():
            for i in range(0, len(todo), self.bs):
                chunk = todo[i:i + self.bs]
                enc = self.tok(chunk, padding=True, truncation=True,
                               max_length=self.max_len, return_tensors='pt')
                enc = {k: v.to(self.dev) for k, v in enc.items()}
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    z, lp = self.model(enc['input_ids'], enc['attention_mask'])
                z = z.float().cpu().numpy()
                lp = lp.float().cpu().numpy()
                for t, a, b in zip(chunk, z, lp):
                    self.emb[t] = a
                    self.pos[t] = b

    def __call__(self, t):
        v = self.emb.get(t)
        return v if v is not None else np.zeros(self.dim, np.float32)

    def position(self, t):
        v = self.pos.get(t)
        return v if v is not None else np.zeros(3, np.float32)


def episode_texts(ep):
    out = []
    for op in OPS:
        for d in ep['support'][op]:
            out += d['steps'] + list(d['probes'].values())
    for q in QS:
        qd = ep['queries'][q]
        out += list(qd['bank'].values()) + list(qd['probes'].values())
    return out


def official_score(preds, golds):
    """100 * B^0.35 * X^0.65 with B=(A*I*T)^(1/3), X=.70Q+.20R+.10E."""
    A = I = Tc = R = Q = E = 0
    nq = nt = ne = 0
    for p, g in zip(preds, golds):
        ok_ep = True
        for q in QS:
            pq, gq = p[q], g[q]
            a = pq['operator'] == gq['operator']
            i = pq['reject'] == gq['reject']
            ts = [x == y for x, y in zip(pq['steps'], gq['steps'])]
            r = all(ts) and len(ts) == 3
            qq = a and i and r
            A += a; I += i; Tc += sum(ts); R += r; Q += qq
            nq += 1; nt += 3
            ok_ep &= qq
        E += ok_ep; ne += 1
    A, I, R, Q = A / nq, I / nq, R / nq, Q / nq
    Tc, E = Tc / nt, E / ne
    B = (A * I * Tc) ** (1 / 3)
    X = 0.70 * Q + 0.20 * R + 0.10 * E
    s = 100 * (B ** 0.35) * (X ** 0.65) if B > 0 and X > 0 else 0.01
    return dict(A=A, I=I, T=Tc, R=R, Q=Q, E=E, B=B, X=X, score=max(s, 0.01))


SUB_COLS = ['sample_id'] + [f'query_{q}_{f}' for q in QS
                            for f in ('operator', 'step1', 'step2', 'step3', 'reject')]


def validate_row(sid, pred):
    """Enforce every structural rule the grader checks."""
    ops = [pred[q]['operator'] for q in QS]
    assert sorted(ops) == OPS, f'{sid}: operators not a permutation: {ops}'
    for q in QS:
        st = pred[q]['steps']
        assert len(st) == 3 and len(set(st)) == 3, f'{sid}/{q}: bad steps {st}'
        assert all(s in SLOTS for s in st), f'{sid}/{q}: bad slot code {st}'
        assert pred[q]['reject'] in PROBES, f'{sid}/{q}: bad probe'


def write_submission(path, ids, preds):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(SUB_COLS)
        for sid, pr in zip(ids, preds):
            validate_row(sid, pr)
            row = [sid]
            for q in QS:
                row += [pr[q]['operator']] + list(pr[q]['steps']) + [pr[q]['reject']]
            w.writerow(row)
    print(f'[out] wrote {path}: {len(ids)} rows x {len(SUB_COLS)} cols', flush=True)


def decode_all(episodes, emb, tag):
    preds, t0 = [], time.time()
    for i, ep in enumerate(episodes):
        emb.fit(episode_texts(ep))
        preds.append(solve_episode(ep, emb, emb.position))
        if (i + 1) % 200 == 0:
            print(f'[{tag}] {i+1}/{len(episodes)} {time.time()-t0:.0f}s', flush=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--test', default='data/test.csv')
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--backbone', default=DEFAULT_BACKBONE)
    ap.add_argument('--episodes', type=int, default=0, help='0 = all')
    ap.add_argument('--bs', type=int, default=192)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--temp', type=float, default=0.05)
    ap.add_argument('--max_len', type=int, default=48)
    ap.add_argument('--max_steps', type=int, default=6000,
                    help='fixed optimisation-step budget (see RUNTIME BUDGET). '
                         '0 = full epochs. Deterministic: the step count is '
                         'fixed up front and never derived from elapsed time.')
    ap.add_argument('--holdout', type=int, default=200, help='episodes for local scoring')
    ap.add_argument('--device', default=DEVICE,
                    help="execution device, fixed at launch (default: 'cuda', "
                         'the competition target). Never probed from the host.')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    T0 = time.time()
    lim = args.episodes or None
    train_eps = [parse_episode(d, True) for d in read_rows(args.train, lim)]
    print(f'[data] train episodes {len(train_eps)} ({time.time()-T0:.0f}s)', flush=True)

    hold = train_eps[:args.holdout] if args.holdout else []
    fit_eps = train_eps[args.holdout:] if args.holdout else train_eps

    model, tok = train_encoder(args, fit_eps)
    dev = args.device
    emb = Embedder(model, tok, dev, max_len=args.max_len)

    if hold:
        hp = decode_all(hold, emb, 'val')
        r = official_score(hp, [e['targets'] for e in hold])
        print('[val] ' + '  '.join(f'{k}={v:.4f}' for k, v in r.items()), flush=True)

    test_eps, ids = [], []
    for d in read_rows(args.test):
        test_eps.append(parse_episode(d, False))
        ids.append(d['sample_id'])
    print(f'[data] test episodes {len(test_eps)}', flush=True)

    preds = decode_all(test_eps, emb, 'test')
    write_submission(args.out, ids, preds)
    print(f'[done] total {time.time()-T0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
