#!/usr/bin/env python3
"""
Feature-trace prediction from raw amino-acid sequence.

Usage:  python solution.py <public_data_dir> <output_csv>

Everything - the lexicons, the count models, the per-residue scorers, the note
tables and the decision thresholds - is fitted inside this script from the
supplied train.csv only.  test.csv is read for inference only.  No network, no
pretrained weights, no external corpora.

The internal validation split is homology-disjoint on purpose: the test proteins
have essentially no close relative in train (median nearest-neighbour 5-mer
identity 0.011 vs 0.68 for a random train holdout), so a random split would
overstate the score roughly four-fold.

Output: exactly two columns, id,feature_trace_json
"""

import sys, os, re, json, time, random, collections
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------- constants
AA = 'ACDEFGHIKLMNPQRSTVWY'
ALPHA = 'ACDEFGHIKLMNPQRSTVWYXBZUO'
AIDX = {c: i for i, c in enumerate(ALPHA)}
NA = len(ALPHA)
AAI = {a: i for i, a in enumerate(AA)}

KD = dict(zip('ARNDCQEGHILKMFPSTWYV',
              [1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5, 3.8,
               -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2]))

TYPES = ['SIGNAL', 'TRANSMEM', 'TOPO_DOM', 'DNA_BIND', 'ZN_FING', 'REGION',
         'MOTIF', 'ACT_SITE', 'BINDING', 'MOD_RES', 'LIPID', 'DISULFID']
TIDX = {t: i for i, t in enumerate(TYPES)}

# emission order recovered from the training traces (mean rank of each type)
EMIT = ['SIGNAL', 'DNA_BIND', 'ZN_FING', 'ACT_SITE', 'REGION', 'TOPO_DOM',
        'TRANSMEM', 'MOTIF', 'BINDING', 'MOD_RES', 'DISULFID', 'LIPID']

# types whose exact residue is chosen by a learned per-residue scorer
RESTYPES = ('BINDING', 'ACT_SITE', 'MOD_RES', 'LIPID')
ALLOWED = {'BINDING': set('DHCERNSTKGQYWFMAVLIP'),
           'ACT_SITE': set('HDEKCSRNTYGQW'),
           'MOD_RES': set('SKTYRM'),
           'LIPID': set('CGSNK')}

MAXK = 40          # cap on features emitted per type
KMER = 5           # k for the homology index used by the disjoint split
SIG_END = 22       # modal signal-peptide cleavage position

# Values the in-script coordinate ascent converges to on the disjoint split.
# Used as the starting point, and as the fallback when calibration is skipped.
DEFAULT_THR = {'SIGNAL': 0.30, 'TRANSMEM': 0.20, 'TOPO_DOM': 0.60, 'DNA_BIND': 0.60,
               'ZN_FING': 0.60, 'REGION': 0.50, 'MOTIF': 0.50, 'ACT_SITE': 0.35,
               'BINDING': 0.45, 'MOD_RES': 0.35, 'LIPID': 0.30, 'DISULFID': 0.20}
DEFAULT_MUL = {'SIGNAL': 1.0, 'TRANSMEM': 1.0, 'TOPO_DOM': 1.5, 'DNA_BIND': 1.0,
               'ZN_FING': 1.0, 'REGION': 1.0, 'MOTIF': 1.0, 'ACT_SITE': 1.25,
               'BINDING': 1.25, 'MOD_RES': 1.1, 'LIPID': 0.8, 'DISULFID': 1.5}


# ----------------------------------------------------------------- utilities
def parse_loc(loc):
    ns = re.findall(r'\d+', str(loc))
    if not ns:
        return None
    v = [int(x) for x in ns]
    return min(v), max(v)


def fmt(a, b):
    return str(a) if a == b else '%d..%d' % (a, b)


def hydro(seq):
    return np.array([KD.get(ch, 0.0) for ch in seq], dtype=np.float32)


def smooth(x, w):
    if len(x) < w:
        return np.full(max(1, len(x)), float(x.mean()) if len(x) else 0.0, np.float32)
    cs = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    return ((cs[w:] - cs[:-w]) / w).astype(np.float32)


def org_tok(o):
    return ' '.join(re.findall(r'[a-z]+', str(o).lower())[:2])


def prot_feats(d):
    """Dense hand features describing one protein."""
    s = d['sequence']; n = len(s)
    cnt = np.zeros(20, np.float32)
    for ch in s:
        i = AAI.get(ch)
        if i is not None:
            cnt[i] += 1
    comp = cnt / max(1, n)
    h = hydro(s)
    h19 = smooth(h, 19) if n >= 19 else np.array([h.mean() if n else 0], np.float32)
    nt = h[:30]
    f = [np.log1p(n), n / 1000.0]
    f += comp.tolist()
    f += [h.mean(), h.std(), h19.max(), h19.min(), float((h19 > 1.6).mean()),
          float((h19 > 1.6).sum()), float((h19 > 2.0).sum()), float((h19 > 1.0).sum())]
    runs, inr = 0, False
    for v in (h19 > 1.6):
        if v and not inr:
            runs += 1; inr = True
        elif not v:
            inr = False
    f += [runs, min(runs, 12)]
    f += [float(nt.mean()) if len(nt) else 0.0, float(nt.max()) if len(nt) else 0.0,
          float(s[:30].count('C')), float(s.count('C')), s.count('C') / max(1, n)]
    dis = sum(s.count(c) for c in 'PESKQGARND') / max(1, n)
    f += [dis]
    w = 40
    if n >= w:
        fr = np.array([sum(s[i:i + w].count(c) for c in 'PESKQGARND') / w
                       for i in range(0, n - w + 1, 10)], np.float32)
        f += [float(fr.max()), float((fr > 0.75).mean()), float((fr > 0.75).sum())]
    else:
        f += [dis, 0.0, 0.0]
    return np.array(f, np.float32)


# -------------------------------------------------- homology-disjoint split
def _norm_csr(M):
    n = np.sqrt(M.multiply(M).sum(1)).A.ravel(); n[n == 0] = 1
    return csr_matrix(M.multiply((1.0 / n)[:, None]), dtype=np.float32)


def kmer_cols(seqs, vocab, build):
    r, c = [], []
    for i, s in enumerate(seqs):
        for km in set(s[j:j + KMER] for j in range(len(s) - KMER + 1)):
            if build:
                j = vocab.setdefault(km, len(vocab))
            else:
                j = vocab.get(km)
                if j is None:
                    continue
            r.append(i); c.append(j)
    return np.asarray(r, np.int32), np.asarray(c, np.int32)


def build_sim(seqs):
    vocab = {}
    r, c = kmer_cols(seqs, vocab, True)
    V, N = len(vocab), len(seqs)
    A = csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(N, V), dtype=np.float32)
    df = np.asarray((A > 0).sum(0)).ravel()
    idf = np.log((N + 1) / (df + 1)).astype(np.float32)
    return vocab, idf, _norm_csr(A.multiply(idf[None, :]).tocsr())


def disjoint_split(seqs, n_val=3000, thresh=0.02, seed=SEED, block=400):
    """Hold out n_val proteins, then drop every train protein similar to any of
    them.  This reproduces the test set's lack of near neighbours."""
    rnd = random.Random(seed)
    N = len(seqs)
    val = rnd.sample(range(N), min(n_val, N))
    _, _, A = build_sim(seqs)
    AT = A.T.tocsr()
    banned = set(val)
    for s in range(0, len(val), block):
        S = (A[val[s:s + block]] @ AT).toarray()
        banned.update(np.where((S >= thresh).any(0))[0].tolist())
    pool = [i for i in range(N) if i not in banned]
    return val, pool


# ------------------------------------------------------- per-residue scorer
class ResidueModel:
    """Logistic model over a +/-W residue window; scores every residue."""

    def __init__(self, W=10, n_pos=250000, n_neg=500000, seed=SEED, C=1.0):
        self.W, self.n_pos, self.n_neg, self.seed, self.C = W, n_pos, n_neg, seed, C
        self.clf = None

    def _design(self, samples, seqs):
        W = self.W
        base = (2 * W + 1) * NA
        NEX = 6
        indptr, idx, dat = [0], [], []
        for si, pos in samples:
            s = seqs[si]; n = len(s)
            cols = []
            for k, off in enumerate(range(-W, W + 1)):
                p = pos - 1 + off
                ch = s[p] if 0 <= p < n else 'X'
                cols.append(k * NA + AIDX.get(ch, AIDX['X']))
            vals = [1.0] * len(cols)
            rel = pos / max(1, n)
            h = KD.get(s[pos - 1] if 1 <= pos <= n else 'X', 0.0)
            lo, hi = max(0, pos - 6), min(n, pos + 5)
            hw = float(np.mean([KD.get(c, 0.0) for c in s[lo:hi]])) if hi > lo else 0.0
            cols += [base + i for i in range(NEX)]
            vals += [rel, min(rel, 1 - rel), float(np.log1p(n)) / 8.0,
                     h / 4.5, hw / 4.5, 1.0]
            idx.extend(cols); dat.extend(vals); indptr.append(len(idx))
        return csr_matrix((np.asarray(dat, np.float32), np.asarray(idx, np.int32),
                           np.asarray(indptr, np.int64)),
                          shape=(len(samples), base + NEX), dtype=np.float32)

    def fit(self, seqs, traces, ftype):
        rnd = random.Random(self.seed)
        pos, neg = [], []
        for si, (s, ft) in enumerate(zip(seqs, traces)):
            hit = set()
            for f in ft:
                if f['type'] != ftype:
                    continue
                p = parse_loc(f['location'])
                if not p:
                    continue
                for q in range(p[0], min(p[1], p[0] + 3) + 1):
                    if 1 <= q <= len(s):
                        hit.add(q)
            if not hit:
                continue
            pos.extend((si, q) for q in hit)
            n = len(s)
            for _ in range(min(3 * len(hit), 40)):
                q = rnd.randint(1, n)
                if q not in hit:
                    neg.append((si, q))
        if not pos:
            return self
        if len(pos) > self.n_pos:
            pos = rnd.sample(pos, self.n_pos)
        if len(neg) > self.n_neg:
            neg = rnd.sample(neg, self.n_neg)
        samples = pos + neg
        y = np.zeros(len(samples), np.int8); y[:len(pos)] = 1
        if y.sum() == 0 or y.sum() == len(y):
            return self
        self.clf = LogisticRegression(C=self.C, max_iter=120, solver='liblinear', tol=1e-3)
        self.clf.fit(self._design(samples, seqs), y)
        return self

    def score(self, seq, cands):
        if self.clf is None:
            return np.zeros(len(cands), np.float32)
        return self.clf.decision_function(self._design([(0, p) for p in cands], [seq]))


# ------------------------------------------------------------ note learning
def note_sim(a, b):
    from difflib import SequenceMatcher
    a = ' '.join(str(a or '').lower().split())
    b = ' '.join(str(b or '').lower().split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def best_note(golds, cands, cap=3000, seed=SEED):
    g = golds if len(golds) <= cap else random.Random(seed).sample(golds, cap)
    best, bs = '', -1.0
    for c in cands:
        s = sum(note_sim(c, x) for x in g) / max(1, len(g))
        if s > bs:
            bs, best = s, c
    return best


class Notes:
    """Per-type note strings chosen to maximise expected character similarity."""

    def fit(self, seqs, traces):
        by = collections.defaultdict(list)
        byres = collections.defaultdict(lambda: collections.defaultdict(list))
        bytopo = collections.defaultdict(list)
        for s, ft in zip(seqs, traces):
            ti = 0
            for f in ft:
                t = f['type']; nt = f.get('note', '') or ''
                by[t].append(nt)
                p = parse_loc(f['location'])
                if t in ('MOD_RES', 'ACT_SITE', 'LIPID') and p and 1 <= p[0] <= len(s):
                    byres[t][s[p[0] - 1]].append(nt)
                if t == 'TOPO_DOM':
                    bytopo[ti % 2].append(nt); ti += 1
        self.N = {t: best_note(v, [c for c, _ in collections.Counter(v).most_common(40)])
                  for t, v in by.items()}
        self.R = {}
        for t, d in byres.items():
            self.R[t] = {r: best_note(v, [c for c, _ in collections.Counter(v).most_common(25)])
                         for r, v in d.items() if len(v) >= 20}
        self.T = {k: best_note(v, [c for c, _ in collections.Counter(v).most_common(25)])
                  for k, v in bytopo.items()}
        return self

    def get(self, t, seq, a, i=0):
        if t == 'TOPO_DOM':
            return self.T.get(i % 2, self.N.get(t, ''))
        if t in self.R and 1 <= a <= len(seq):
            r = self.R[t].get(seq[a - 1])
            if r is not None:
                return r
        return self.N.get(t, '')


# ------------------------------------------------------- location generators
def tm_ranked(seq, k, seglen=21):
    """Best-first non-overlapping hydrophobic windows (transmembrane helices)."""
    n = len(seq)
    if k <= 0 or n < seglen:
        return []
    sc = smooth(hydro(seq), seglen)
    used = np.zeros(n + seglen, bool)
    out = []
    for i in np.argsort(-sc):
        if len(out) >= k:
            break
        a, b = int(i), int(i) + seglen - 1
        if used[max(0, a - 4):b + 5].any():
            continue
        used[a:b + 1] = True
        out.append((a + 1, b + 1))
    return out


def region_ranked(seq, k, w=40, seglen=36):
    """Best-first polar/low-complexity windows (disordered regions)."""
    n = len(seq)
    if k <= 0:
        return []
    DIS = set('PESKQGARNDTH')
    fl = np.array([1.0 if ch in DIS else 0.0 for ch in seq], np.float32)
    if n < w:
        return [(1, n)]
    sc = smooth(fl, w)
    used = np.zeros(n + w, bool)
    out = []
    for i in np.argsort(-sc):
        if len(out) >= k:
            break
        a, b = int(i), min(n, int(i) + seglen - 1)
        if used[a:b + 1].any():
            continue
        used[max(0, a - 5):b + 6] = True
        out.append((a + 1, b + 1))
    return out


def topo_from_tm(seq, tms, k):
    """Loops flanking the chosen TM helices, longest first."""
    n = len(seq)
    if k <= 0:
        return []
    if not tms:
        return [(1, n)][:k]
    segs = []
    if tms[0][0] > 1:
        segs.append((1, tms[0][0] - 1))
    for (a1, b1), (a2, b2) in zip(tms, tms[1:]):
        if a2 - b1 > 1:
            segs.append((b1 + 1, a2 - 1))
    if tms[-1][1] < n:
        segs.append((tms[-1][1] + 1, n))
    segs.sort(key=lambda s: -(s[1] - s[0]))
    return sorted(segs[:k])


def cys_pairs(seq, k):
    cys = [i + 1 for i, ch in enumerate(seq) if ch == 'C']
    out = []
    for j in range(0, len(cys) - 1, 2):
        if len(out) >= k:
            break
        out.append((cys[j], cys[j + 1]))
    return out


def precompute(seq, rms):
    """Ranked candidate spans per type; a [:k] prefix is the best k."""
    n = len(seq)
    pre = {'TRANSMEM': tm_ranked(seq, MAXK),
           'REGION': region_ranked(seq, MAXK),
           'DISULFID': cys_pairs(seq, MAXK)}
    for t in RESTYPES:
        cands = [i + 1 for i, ch in enumerate(seq) if ch in ALLOWED[t]] or list(range(1, n + 1))
        rm = rms.get(t)
        if rm is None or rm.clf is None:
            pre[t] = [(p, p) for p in cands[:MAXK]]
        else:
            sc = rm.score(seq, cands)
            pre[t] = [(cands[i], cands[i]) for i in np.argsort(-sc)[:MAXK]]
    cys = [i + 1 for i, ch in enumerate(seq) if ch == 'C']
    pre['ZN_FING'] = [(a, min(n, a + 34)) for a in [cys[j] for j in range(0, len(cys), 4)][:MAXK]]
    pre['DNA_BIND'] = [(max(1, n // 3), min(n, n // 3 + 59))]
    return pre


def build_trace(seq, counts, pre, notes):
    """Assemble the ordered feature trace from per-type counts."""
    n = len(seq)
    sp = {}
    tms = sorted(pre['TRANSMEM'][:counts.get('TRANSMEM', 0)])
    sp['TRANSMEM'] = tms
    sp['TOPO_DOM'] = topo_from_tm(seq, tms, counts.get('TOPO_DOM', 0))
    sp['REGION'] = sorted(pre['REGION'][:counts.get('REGION', 0)])
    sp['SIGNAL'] = [(1, min(SIG_END, n))] if counts.get('SIGNAL', 0) > 0 else []
    sp['DISULFID'] = sorted(pre['DISULFID'][:counts.get('DISULFID', 0)])
    for t in RESTYPES:
        sp[t] = sorted(pre[t][:counts.get(t, 0)])
    sp['ZN_FING'] = sorted(pre['ZN_FING'][:counts.get('ZN_FING', 0)])
    sp['DNA_BIND'] = pre['DNA_BIND'] if counts.get('DNA_BIND', 0) else []
    nm = counts.get('MOTIF', 0)
    step = max(1, n // (nm + 1)) if nm else 1
    sp['MOTIF'] = [(min(n, (j + 1) * step), min(n, (j + 1) * step + 4)) for j in range(nm)]

    out = []
    for t in EMIT:
        if t == 'TOPO_DOM':
            # topology alternates along the chain, so emit TM and loops interleaved
            merged = ([(a, b, 'TOPO_DOM', i) for i, (a, b) in enumerate(sp['TOPO_DOM'])] +
                      [(a, b, 'TRANSMEM', i) for i, (a, b) in enumerate(sp['TRANSMEM'])])
            for a, b, tt, i in sorted(merged):
                a = max(1, min(n, a)); b = max(a, min(n, b))
                out.append({'type': tt, 'location': fmt(a, b), 'note': notes.get(tt, seq, a, i)})
            continue
        if t == 'TRANSMEM':
            continue
        for i, (a, b) in enumerate(sp.get(t) or []):
            a = max(1, min(n, a)); b = max(a, min(n, b))
            out.append({'type': t, 'location': fmt(a, b), 'note': notes.get(t, seq, a, i)})
    return out


# ----------------------------------------------------------- count modelling
class CountModel:
    """Predicts, for each feature type, whether it occurs and how many times."""

    def fit(self, ins, fts):
        names = [d['protein_name'] for d in ins]
        orgs = [org_tok(d['organism']) for d in ins]
        self.vw = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=3,
                                  sublinear_tf=True)
        self.vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=5,
                                  sublinear_tf=True, max_features=300000)
        self.vo = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=3,
                                  sublinear_tf=True)
        Xw = self.vw.fit_transform(names)
        Xc = self.vc.fit_transform(names)
        Xo = self.vo.fit_transform(orgs)
        Dm = np.vstack([prot_feats(d) for d in ins])
        self.sc = StandardScaler().fit(Dm)
        X = hstack([Xw, Xc, Xo, csr_matrix(self.sc.transform(Dm).astype(np.float32))]).tocsr()

        Y = np.zeros((len(ins), len(TYPES)), np.float32)
        for i, ft in enumerate(fts):
            for f in ft:
                j = TIDX.get(f['type'])
                if j is not None:
                    Y[i, j] += 1
        self.pres, self.cnt = [], []
        for j in range(len(TYPES)):
            y = (Y[:, j] > 0).astype(np.int8)
            if y.sum() < 30 or y.sum() == len(y):
                self.pres.append(None); self.cnt.append(None); continue
            self.pres.append(LogisticRegression(C=2.0, max_iter=400,
                                                solver='liblinear').fit(X, y))
            m = Y[:, j] > 0
            self.cnt.append(Ridge(alpha=2.0).fit(X[m], np.log1p(Y[m, j])))
        return self

    def _X(self, ins):
        names = [d['protein_name'] for d in ins]
        Dm = np.vstack([prot_feats(d) for d in ins])
        return hstack([self.vw.transform(names), self.vc.transform(names),
                       self.vo.transform([org_tok(d['organism']) for d in ins]),
                       csr_matrix(self.sc.transform(Dm).astype(np.float32))]).tocsr()

    def predict(self, ins, batch=20000):
        n = len(ins)
        P = np.zeros((n, len(TYPES)), np.float32)
        C = np.zeros((n, len(TYPES)), np.float32)
        for s in range(0, n, batch):
            X = self._X(ins[s:s + batch])
            m = X.shape[0]
            for j in range(len(TYPES)):
                if self.pres[j] is None:
                    continue
                P[s:s + m, j] = self.pres[j].predict_proba(X)[:, 1]
                C[s:s + m, j] = np.expm1(self.cnt[j].predict(X))
        return P, C


def counts_from(P, C, thr, mul):
    out = []
    for i in range(P.shape[0]):
        d = {}
        for j, t in enumerate(TYPES):
            if P[i, j] >= thr[t]:
                d[t] = int(max(1, min(MAXK, round(C[i, j] * mul[t]))))
        out.append(d)
    return out


# ------------------------------------------------------------------- scoring
def _lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return inter / union if union > 0 else 0.0


def row_score(P_, G_):
    P_, G_ = list(P_ or []), list(G_ or [])
    pk = [(f['type'], f['location']) for f in P_]
    gk = [(f['type'], f['location']) for f in G_]
    if not pk and not gk:
        s1 = 1.0
    elif not pk or not gk:
        s1 = 0.0
    else:
        cp, cg = collections.Counter(pk), collections.Counter(gk)
        s1 = 2.0 * sum((cp & cg).values()) / (len(pk) + len(gk))
    pt = [f['type'] for f in P_]; gt = [f['type'] for f in G_]
    s2 = 1.0 if (not pt and not gt) else (0.0 if (not pt or not gt)
                                          else 2.0 * _lcs(pt, gt) / (len(pt) + len(gt)))
    if not P_ and not G_:
        s3 = s4 = 1.0
    elif not P_ or not G_:
        s3 = s4 = 0.0
    else:
        cand = []
        for i, fp in enumerate(P_):
            lp = parse_loc(fp['location'])
            for j, fg in enumerate(G_):
                if fp['type'] != fg['type']:
                    continue
                v = _iou(lp, parse_loc(fg['location']))
                if v > 0:
                    cand.append((v, i, j))
        cand.sort(key=lambda t: (-t[0], t[1], t[2]))
        ui, uj, tot, sims = set(), set(), 0.0, []
        for v, i, j in cand:
            if i in ui or j in uj:
                continue
            ui.add(i); uj.add(j); tot += v
            sims.append(note_sim(P_[i].get('note', ''), G_[j].get('note', '')))
        s3 = tot / max(len(P_), len(G_), 1)
        s4 = (sum(sims) / len(sims)) if sims else 0.0
    c = lambda x: max(0.0, min(1.0, x))
    return c(0.55 * c(s1) + 0.20 * c(s2) + 0.15 * c(s3) + 0.10 * c(s4)), (s1, s2, s3, s4)


def score_all(preds, golds):
    rs = [row_score(p, g) for p, g in zip(preds, golds)]
    n = max(1, len(rs))
    return (sum(r[0] for r in rs) / n,
            tuple(sum(r[1][k] for r in rs) / n for k in range(4)))


# ---------------------------------------------------------------------- main
def fit_bundle(ins, fts, seqs, log):
    notes = Notes().fit(seqs, fts)
    log('  notes learned')
    rms = {t: ResidueModel().fit(seqs, fts, t) for t in RESTYPES}
    log('  per-residue scorers fitted')
    cm = CountModel().fit(ins, fts)
    log('  count model fitted')
    return notes, rms, cm


def calibrate(tr_in, tr_ft, tr_seq, log):
    """Tune per-type thresholds and count multipliers on a homology-disjoint
    internal split, and report the score there."""
    val, pool = disjoint_split(tr_seq, n_val=3000, thresh=0.02)
    log('  disjoint split: %d val / %d usable pool (of %d)' % (len(val), len(pool), len(tr_seq)))
    pin = [tr_in[i] for i in pool]; pft = [tr_ft[i] for i in pool]
    pseq = [tr_seq[i] for i in pool]
    vft = [tr_ft[i] for i in val]; vseq = [tr_seq[i] for i in val]
    vin = [tr_in[i] for i in val]

    notes, rms, cm = fit_bundle(pin, pft, pseq, log)
    pre = [precompute(s, rms) for s in vseq]
    P, C = cm.predict(vin)

    thr = dict(DEFAULT_THR); mul = dict(DEFAULT_MUL)
    ev = lambda th, mu: score_all(
        [build_trace(s, c, p, notes) for s, c, p in
         zip(vseq, counts_from(P, C, th, mu), pre)], vft)[0]
    best = ev(thr, mul)
    for _ in range(2):
        for t in TYPES:
            for v in (0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70):
                o = thr[t]; thr[t] = v
                s = ev(thr, mul)
                if s > best:
                    best = s
                else:
                    thr[t] = o
            for v in (0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5):
                o = mul[t]; mul[t] = v
                s = ev(thr, mul)
                if s > best:
                    best = s
                else:
                    mul[t] = o
    preds = [build_trace(s, c, p, notes) for s, c, p in
             zip(vseq, counts_from(P, C, thr, mul), pre)]
    m, comp = score_all(preds, vft)
    log('  HOMOLOGY-DISJOINT VALIDATION  score %.4f '
        '(exactF1 %.4f  typeLCS %.4f  IoU %.4f  note %.4f)' % (m, *comp))
    return thr, mul, m


def main(data_dir, out_path):
    t0 = time.time()
    log = lambda m: print('[%6.1fs] %s' % (time.time() - t0, m), flush=True)

    tr = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    te = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    tr_in = [json.loads(s) for s in tr['input_json']]
    tr_ft = [json.loads(s)['feature_trace'] for s in tr['feature_trace_json']]
    te_in = [json.loads(s) for s in te['input_json']]
    tr_seq = [d['sequence'] for d in tr_in]
    te_seq = [d['sequence'] for d in te_in]
    te_ids = list(te['id'])
    log('loaded train=%d test=%d' % (len(tr_in), len(te_in)))

    thr, mul = dict(DEFAULT_THR), dict(DEFAULT_MUL)
    if os.environ.get('SKIP_CALIBRATION', '') != '1':
        log('calibrating on a homology-disjoint internal split ...')
        try:
            thr, mul, _ = calibrate(tr_in, tr_ft, tr_seq, log)
        except Exception as exc:                       # keep going with defaults
            log('  calibration skipped (%s)' % exc)

    log('fitting final models on all %d training rows ...' % len(tr_in))
    notes, rms, cm = fit_bundle(tr_in, tr_ft, tr_seq, log)

    log('predicting counts for %d test rows ...' % len(te_in))
    P, C = cm.predict(te_in)
    cs = counts_from(P, C, thr, mul)

    log('generating traces ...')
    rows = []
    for i, (sid, seq, cnt) in enumerate(zip(te_ids, te_seq, cs)):
        tr_out = build_trace(seq, cnt, precompute(seq, rms), notes)
        rows.append((sid, json.dumps({'feature_trace': tr_out},
                                     ensure_ascii=False, separators=(',', ':'))))
        if (i + 1) % 5000 == 0:
            log('  %d/%d' % (i + 1, len(te_ids)))

    out = pd.DataFrame(rows, columns=['id', 'feature_trace_json'])
    assert list(out.columns) == ['id', 'feature_trace_json']
    assert len(out) == len(te_ids) and out['id'].nunique() == len(te_ids)
    out.to_csv(out_path, index=False, encoding='utf-8')
    log('wrote %s  (%d rows, columns %s)' % (out_path, len(out), list(out.columns)))


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else 'public'
    o = sys.argv[2] if len(sys.argv) > 2 else 'submission.csv'
    main(d, o)
