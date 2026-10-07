#!/usr/bin/env python3
"""
Anonymous model-card routing: recover MODEL>TASK@EVIDENCE ledgers.

Usage:  python solution.py <public_data_dir> <output_submission_csv>

Approach
--------
The dataset ships every source triple of profiles as two independently shuffled
candidate views.  Grouping the two views of a triple gives two structural wins:

  * evidence : the true evidence of a profile must live in BOTH views' evidence
    banks, so the 7-way bank collapses to the intersection (almost always
    exactly the 3 true evidences).  The remaining job is a 3x3 permutation.
  * task     : likewise the 9-way task bank collapses to the ~6 element
    intersection of the two views.

The evidence fragment is the literal textual continuation of the truncated card
fragment, so the evidence route is scored with a stupid-backoff word n-gram LM
and a character LM evaluated at the prefix/suffix junction, plus punctuation,
quote-balance, SDLC-phase and lexical-overlap features, combined by a learned
pairwise ranker.

The gold capability is masked out of the profile text as "<CAPABILITY>", so the
task route uses a fill-the-blank LM score over that mask, a 32-way TF-IDF text
classifier over profile+evidence, and literal-mention features.

Both routes are decoded jointly per group by enumerating evidence permutations
and, for each, the best one-to-one task assignment.
"""
import sys, os, re, json, math, itertools, collections
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# optional sklearn; pure-numpy fallbacks keep the script runnable without it
# --------------------------------------------------------------------------
try:
    from sklearn.linear_model import LogisticRegression as _SkLR
    from sklearn.ensemble import HistGradientBoostingClassifier as _SkGB
    from sklearn.feature_extraction.text import TfidfVectorizer as _SkTfidf
    from scipy.sparse import hstack as _sp_hstack
    HAVE_SK = True
except Exception:                                            # pragma: no cover
    HAVE_SK = False

SEED = 0

# ==========================================================================
# text utilities
# ==========================================================================
CF = '|| card fragment:'
TOKRE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")
WORDRE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
PHASE = re.compile(r'\b(SI|SQA|SD|SM|SR)\b|\bSoftware (Implementation|Quality Assurance|Design|Maintenance|Requirement)')
PHMAP = {'SI': 'SI', 'SQA': 'SQA', 'SD': 'SD', 'SM': 'SM', 'SR': 'SR',
         'Implementation': 'SI', 'Quality Assurance': 'SQA', 'Design': 'SD',
         'Maintenance': 'SM', 'Requirement': 'SR'}
MASK = '<CAPABILITY>'
MASKS = ('<MODEL>', MASK, '<NUM>')


def card(t):
    i = t.find(CF)
    return t[i + len(CF):].strip() if i >= 0 else t.strip()


def tok(s):
    return TOKRE.findall(s.lower())


def phases(s):
    out = set()
    for a, b in PHASE.findall(s):
        out.add(PHMAP[a or b])
    return out


def norm(s):
    return ' ' + re.sub(r'[^a-z0-9 ]', ' ', s.lower()) + ' '


class NGramLM:
    """Word-level stupid-backoff language model."""

    def __init__(self, texts, maxn=5):
        self.maxn = maxn
        self.cnt = [collections.Counter() for _ in range(maxn + 1)]
        for t in texts:
            w = tok(t)
            for n in range(1, maxn + 1):
                c = self.cnt[n]
                for i in range(len(w) - n + 1):
                    c[tuple(w[i:i + n])] += 1
        self.tot1 = max(1, sum(self.cnt[1].values()))

    def logp(self, ctx, nxt):
        f = 1.0
        for k in range(min(len(ctx), self.maxn - 1), -1, -1):
            c = ctx[len(ctx) - k:] if k > 0 else ()
            num = self.cnt[k + 1].get(c + (nxt,), 0)
            if num:
                den = self.cnt[k].get(c, 0) if k > 0 else self.tot1
                if den:
                    return math.log(f * num / den)
            f *= 0.4
        return math.log(f * 0.4 / self.tot1)

    def seq(self, left, right, J):
        s, n = 0.0, 0
        for j in range(min(J, len(right))):
            ctx = tuple((left + right[:j])[-(self.maxn - 1):])
            s += self.logp(ctx, right[j])
            n += 1
        return s, n


class CharLM:
    """Character n-gram model used to score the prefix/suffix junction."""

    def __init__(self, texts, n=6):
        self.n = n
        self.c = collections.Counter()
        self.p = collections.Counter()
        for t in texts:
            s = ' ' + t.lower() + ' '
            for i in range(len(s) - n + 1):
                g = s[i:i + n]
                self.c[g] += 1
                self.p[g[:-1]] += 1

    def score(self, left, right, J=12):
        s, n = 0.0, self.n
        L = left[-(n - 1):]
        for j in range(min(J, len(right))):
            g = (L + right[:j + 1])[-n:]
            num = self.c.get(g, 0)
            den = self.p.get(g[:-1], 0)
            s += math.log((num + 0.05) / (den + 0.05 * 80))
        return s


class Idf:
    def __init__(self, texts):
        df = collections.Counter()
        for t in texts:
            for w in set(WORDRE.findall(t.lower())):
                df[w] += 1
        N = len(texts)
        self.idf = {w: math.log((N + 1) / (c + 1)) + 1 for w, c in df.items()}
        self.default = math.log(N + 1) + 1

    def vec(self, t):
        c = collections.Counter(WORDRE.findall(t.lower()))
        v = {w: (1 + math.log(n)) * self.idf.get(w, self.default) for w, n in c.items()}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / nrm for w, x in v.items()}


def cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b[w] for w, x in a.items() if w in b)


def charset(s, n=4):
    s = re.sub(r'\s+', ' ', s.lower())
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jac(a, b):
    if not a or not b:
        return 0.0
    i = len(a & b)
    return i / (len(a) + len(b) - i)


# ==========================================================================
# pure-numpy fallback learners (used only when sklearn is unavailable)
# ==========================================================================
class NpLogReg:
    def __init__(self, C=1.0, iters=600, lr=0.5):
        self.C, self.iters, self.lr = C, iters, lr

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        X = np.hstack([X, np.ones((n, 1))])
        w = np.zeros(d + 1)
        m1 = np.zeros(d + 1)
        m2 = np.zeros(d + 1)
        lam = 1.0 / max(self.C, 1e-6)
        for t in range(1, self.iters + 1):
            z = X @ w
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = X.T @ (p - y) / n
            g[:-1] += lam * w[:-1] / n
            m1 = 0.9 * m1 + 0.1 * g
            m2 = 0.999 * m2 + 0.001 * g * g
            w -= self.lr * (m1 / (1 - 0.9 ** t)) / (np.sqrt(m2 / (1 - 0.999 ** t)) + 1e-8)
        self.w = w
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.hstack([X, np.ones((X.shape[0], 1))]) @ self.w


def augment(block):
    """Append candidate-set-centred copies of each feature (listwise context)."""
    B = np.asarray(block, dtype=np.float64)
    return np.hstack([B, B - B.mean(0, keepdims=True)])


def fit_binary(X, y, kind='lr'):
    X = np.asarray(X, dtype=np.float64)
    if kind == 'gb' and HAVE_SK:
        m = _SkGB(max_iter=400, learning_rate=0.06, max_leaf_nodes=15,
                  l2_regularization=1.0, random_state=SEED).fit(X, y)

        def score(Z):
            p = np.clip(m.predict_proba(np.asarray(Z, dtype=np.float64))[:, 1], 1e-9, 1 - 1e-9)
            return np.log(p / (1 - p))
        return score
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    m = (_SkLR(C=1.0, max_iter=4000) if HAVE_SK else NpLogReg(C=1.0)).fit(Z, y)

    def score(W):
        return m.decision_function((np.asarray(W, dtype=np.float64) - mu) / sd)
    return score


# ==========================================================================
# feature extraction
# ==========================================================================
def prep_profile(t, idf):
    c = card(t)
    return dict(full=t, raw=c, toks=tok(c), ph=phases(t), cs=charset(c), vec=idf.vec(t))


def prep_ev(t, idf):
    return dict(raw=t, toks=tok(t), ph=phases(t), cs=charset(t), vec=idf.vec(t))


def ev_features(pf, ev, lm, clm):
    f = []
    pt, et = pf['toks'], ev['toks']
    for J in (1, 2, 3, 5, 8):
        s, n = lm.seq(pt, et, J)
        b, _ = lm.seq([], et, J)
        f.append(s / max(n, 1))
        f.append((s - b) / max(n, 1))
    f.append(clm.score(pf['raw'], ev['raw'], 8) / 8.0)
    f.append((clm.score(pf['raw'], ev['raw'], 16) - clm.score('', ev['raw'], 16)) / 16.0)
    f.append(clm.score(pf['raw'], ev['raw'], 3) / 3.0)
    # backward direction: how well the prefix tail is predicted from a merged context
    for J in (2, 4):
        rs, rn = lm.seq(et[::-1], pt[::-1][:J], J)
        rb, _ = lm.seq([], pt[::-1][:J], J)
        f.append(rs / max(rn, 1))
        f.append((rs - rb) / max(rn, 1))

    lc = pf['raw'][-1] if pf['raw'] else ' '
    fc = ev['raw'][0] if ev['raw'] else ' '
    endsent = 1.0 if lc in '.!?' or (len(pf['raw']) > 1 and pf['raw'][-2] in '.!?' and lc == '"') else 0.0
    up = 1.0 if fc.isupper() else 0.0
    lo = 1.0 if fc.islower() else 0.0
    f += [endsent, up, lo, endsent * up, endsent * lo, (1 - endsent) * lo,
          (1 - endsent) * up, 1.0 if lc == ',' else 0.0, 1.0 if lc == '"' else 0.0]
    for ch in ('"', "'", '`'):
        f.append(1.0 if (pf['raw'].count(ch) + ev['raw'].count(ch)) % 2 == 0 else 0.0)
    f.append(float((pf['raw'].count('(') + ev['raw'].count('(')) -
                   (pf['raw'].count(')') + ev['raw'].count(')'))))
    f.append(1.0 if ev['raw'].rstrip().endswith('"') else 0.0)

    a, b = pf['ph'], ev['ph']
    f += [1.0 if (a & b) else 0.0, 1.0 if (a and b and not (a & b)) else 0.0,
          1.0 if not b else 0.0, float(len(a & b))]
    for m in MASKS:
        f.append(1.0 if (m in pf['raw']) == (m in ev['raw']) else 0.0)
        f.append(1.0 if (m in pf['raw'] and m in ev['raw']) else 0.0)
    f.append(cos(pf['vec'], ev['vec']))
    f.append(jac(pf['cs'], ev['cs']))
    f.append(len(et) / 50.0)
    f.append(len(pt) / 50.0)
    return f


def subst_score(ptxt, d, lm):
    """Fill-the-blank LM score for candidate `d` at the <CAPABILITY> mask."""
    if MASK not in ptxt:
        return 0.0, 0.0
    tot, n = 0.0, 0
    dt = tok(d)
    for m in re.finditer(re.escape(MASK), ptxt):
        pos = m.start()
        left = tok(ptxt[:pos])[-4:]
        right = tok(ptxt[pos + len(MASK):])[:4]
        s, c = lm.seq(left, dt + right, len(dt) + len(right))
        tot += s / max(c, 1)
        n += 1
    return tot / max(n, 1), 1.0


def task_features(p_txt, e_txt, d, logprob_row, cidx, prior, tot_prior, ntask, lm, idf, dvec):
    pl, el, dl = norm(p_txt), norm(e_txt), norm(d).strip()
    f = [logprob_row[cidx[d]] if d in cidx else -12.0]
    f.append(1.0 if ' ' + dl + ' ' in pl else 0.0)
    f.append(1.0 if ' ' + dl in pl else 0.0)
    f.append(1.0 if ' ' + dl + ' ' in el else 0.0)
    f.append(1.0 if ' ' + dl in el else 0.0)
    dw, pw, ew = set(dl.split()), set(pl.split()), set(el.split())
    f.append(len(dw & pw) / max(len(dw), 1))
    f.append(len(dw & ew) / max(len(dw), 1))
    f.append(math.log((prior.get(d, 0) + 1) / (tot_prior + ntask)))
    f.append(1.0 if MASK in p_txt else 0.0)
    s, h = subst_score(p_txt, d, lm)
    f += [s, h, s * h]
    f.append(cos(dvec, idf.vec(p_txt)))
    f.append(cos(dvec, idf.vec(e_txt)))
    f.append(float(len(dw)))
    return f


# ==========================================================================
# dataset plumbing
# ==========================================================================
def parse(df, labelled):
    """Return list of per-case dicts."""
    out = []
    for _, r in df.iterrows():
        c = dict(case_id=r.case_id,
                 profiles=json.loads(r.profiles),
                 tasks=json.loads(r.task_bank),
                 evid=json.loads(r.evidence_bank))
        if labelled:
            g = {}
            for a in str(r.target_ledger).split(';'):
                m, rest = a.split('>')
                t, e = rest.split('@')
                g[m] = (t, e)
            c['gold'] = g
        out.append(c)
    return out


def group_cases(cases):
    """Group the (exactly two) shuffled views of each source profile triple."""
    g = collections.defaultdict(list)
    for i, c in enumerate(cases):
        g[frozenset(p['text'] for p in c['profiles'])].append(i)
    return list(g.values())


def build_groups(cases, labelled):
    """One record per source triple, with cross-view reduced candidate sets."""
    recs = []
    for gi, idxs in enumerate(group_cases(cases)):
        ev_sets = [{e['text'] for e in cases[i]['evid']} for i in idxs]
        tk_sets = [{t['description'] for t in cases[i]['tasks']} for i in idxs]
        ei = set.intersection(*ev_sets)
        ti = set.intersection(*tk_sets)
        ev_cand = sorted(ei) if len(ei) >= 3 else sorted(set.union(*ev_sets))
        tk_cand = sorted(ti) if len(ti) >= 3 else sorted(set.union(*tk_sets))
        i0 = idxs[0]
        profs = [p['text'] for p in cases[i0]['profiles']]
        rec = dict(gi=gi, idxs=idxs, profs=profs, ev_cand=ev_cand, tk_cand=tk_cand)
        if labelled:
            emap = {e['evidence']: e['text'] for e in cases[i0]['evid']}
            tmap = {t['task']: t['description'] for t in cases[i0]['tasks']}
            pm = {p['model']: p['text'] for p in cases[i0]['profiles']}
            rec['gold_ev'] = {pm[m]: emap[e] for m, (t, e) in cases[i0]['gold'].items()}
            rec['gold_tk'] = {pm[m]: tmap[t] for m, (t, e) in cases[i0]['gold'].items()}
        recs.append(rec)
    return recs


# ==========================================================================
# model
# ==========================================================================
class Router:
    def fit_resources(self, all_cases):
        texts = []
        for c in all_cases:
            for p in c['profiles']:
                texts.append(card(p['text']))
            for e in c['evid']:
                texts.append(e['text'])
        texts = list(dict.fromkeys(texts))
        self.lm = NGramLM(texts, maxn=6)
        self.clm = CharLM(texts)
        self.idf = Idf(texts)
        self.tasks = sorted({t['description'] for c in all_cases for t in c['tasks']})
        self.dvec = {d: self.idf.vec(d) for d in self.tasks}
        self._pc, self._ec = {}, {}

    def P(self, t):
        if t not in self._pc:
            self._pc[t] = prep_profile(t, self.idf)
        return self._pc[t]

    def E(self, t):
        if t not in self._ec:
            self._ec[t] = prep_ev(t, self.idf)
        return self._ec[t]

    # -------------------- evidence route --------------------
    def _ev_block(self, r):
        blocks = []
        for p in r['profs']:
            pf = self.P(p)
            blocks.append(augment([ev_features(pf, self.E(c), self.lm, self.clm)
                                   for c in r['ev_cand']]))
        return np.vstack(blocks)

    def fit_evidence(self, recs):
        X, y = [], []
        for r in recs:
            X.append(self._ev_block(r))
            for p in r['profs']:
                for c in r['ev_cand']:
                    y.append(1 if r['gold_ev'].get(p) == c else 0)
        X = np.vstack(X)
        y = np.array(y)
        gb = fit_binary(X, y, 'gb')
        lr = fit_binary(X, y, 'lr')
        self.ev_score = lambda Z: 0.65 * gb(Z) + 0.35 * lr(Z)

    def evidence_matrix(self, r):
        s = self.ev_score(self._ev_block(r))
        return np.array(s).reshape(len(r['profs']), len(r['ev_cand']))

    # -------------------- task route --------------------
    def fit_task_classifier(self, recs, ev_for):
        docs = [p + ' || ' + ev_for(r, p) for r in recs for p in r['profs']]
        lab = [r['gold_tk'][p] for r in recs for p in r['profs']]
        if HAVE_SK:
            self.vw = _SkTfidf(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
            self.vc = _SkTfidf(analyzer='char_wb', ngram_range=(3, 5), min_df=3,
                               sublinear_tf=True, max_features=200000)
            A = _sp_hstack([self.vw.fit_transform(docs), self.vc.fit_transform(docs)]).tocsr()
            self.tclf = _SkLR(C=4, max_iter=3000).fit(A, lab)
            self.cidx = {c: i for i, c in enumerate(self.tclf.classes_)}
        else:
            self.tclf = None
            self.cidx = {c: i for i, c in enumerate(self.tasks)}
            self._centroid = {}
            byl = collections.defaultdict(list)
            for d, l in zip(docs, lab):
                byl[l].append(d)
            for l, ds in byl.items():
                v = collections.Counter()
                for d in ds:
                    for w, x in self.idf.vec(d).items():
                        v[w] += x
                nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
                self._centroid[l] = {w: x / nrm for w, x in v.items()}
        self.prior = collections.Counter(lab)
        self.tot_prior = sum(self.prior.values())

    def task_logprobs(self, docs):
        if HAVE_SK:
            B = _sp_hstack([self.vw.transform(docs), self.vc.transform(docs)]).tocsr()
            return self.tclf.predict_log_proba(B)
        out = np.full((len(docs), len(self.tasks)), -12.0)
        for i, d in enumerate(docs):
            v = self.idf.vec(d)
            for j, l in enumerate(self.tasks):
                out[i, j] = 6.0 * cos(v, self._centroid.get(l, {}))
            out[i] -= np.log(np.exp(out[i] - out[i].max()).sum()) + out[i].max()
        return out

    def task_rows(self, r, docs_lp, prof_idx, p, e_txt, lp_row):
        return [task_features(p, e_txt, d, lp_row, self.cidx, self.prior,
                              self.tot_prior, len(self.tasks), self.lm, self.idf,
                              self.dvec[d]) for d in r['tk_cand']]

    def fit_task(self, recs, ev_for):
        X, y = [], []
        docs, keys = [], []
        for r in recs:
            for p in r['profs']:
                docs.append(p + ' || ' + ev_for(r, p))
                keys.append((r, p))
        LP = self.task_logprobs(docs)
        for i, (r, p) in enumerate(keys):
            e = ev_for(r, p)
            blk = []
            for j, d in enumerate(r['tk_cand']):
                blk.append(task_features(p, e, d, LP[i], self.cidx, self.prior,
                                         self.tot_prior, len(self.tasks), self.lm,
                                         self.idf, self.dvec[d]))
                y.append(1 if r['gold_tk'][p] == d else 0)
            X.append(augment(blk))
        X = np.vstack(X)
        self.tk_score = fit_binary(X, np.array(y), 'lr')

    # -------------------- joint decoding --------------------
    def decode(self, r):
        profs = r['profs']
        EV = self.evidence_matrix(r)                      # 3 x |ev_cand|
        nE, nT = len(r['ev_cand']), len(r['tk_cand'])
        # task scores for every (profile, evidence-hypothesis, task) triple
        docs, key = [], []
        for i, p in enumerate(profs):
            for k, ec in enumerate(r['ev_cand']):
                docs.append(p + ' || ' + ec)
                key.append((i, k))
        LP = self.task_logprobs(docs)
        TK = np.zeros((len(profs), nE, nT))
        rows, idxs = [], []
        for n, (i, k) in enumerate(key):
            p, ec = profs[i], r['ev_cand'][k]
            blk = []
            for j, d in enumerate(r['tk_cand']):
                blk.append(task_features(p, ec, d, LP[n], self.cidx, self.prior,
                                         self.tot_prior, len(self.tasks), self.lm,
                                         self.idf, self.dvec[d]))
                idxs.append((i, k, j))
            rows.append(augment(blk))
        sc = self.tk_score(np.vstack(rows))
        for (i, k, j), v in zip(idxs, sc):
            TK[i, k, j] = v
        best, bs = None, -1e18
        for pe in itertools.permutations(range(nE), 3):
            base = sum(EV[i, pe[i]] for i in range(3))
            bt, bts = None, -1e18
            for pt in itertools.permutations(range(nT), 3):
                v = sum(TK[i, pe[i], pt[i]] for i in range(3))
                if v > bts:
                    bts, bt = v, pt
            if base + bts > bs:
                bs, best = base + bts, (pe, bt)
        pe, pt = best
        return {profs[i]: (r['tk_cand'][pt[i]], r['ev_cand'][pe[i]]) for i in range(3)}


# ==========================================================================
# owner-aware local validation
# ==========================================================================
# The test partition holds out complete model owners, so a random split would
# be optimistic.  Owner identity is not recoverable (and recovering publisher
# identity is prohibited), so we use a deliberately conservative proxy: profiles
# whose *card templates* are near-duplicates are assumed to possibly share an
# owner and are forced into the same fold.  Over-merging is the safe direction
# here -- it can only make validation stricter, never leakier.
OWNER_TAU = 0.70


def _profile_sim_pairs(profs, tau):
    """Yield near-duplicate profile index pairs above cosine `tau`."""
    if HAVE_SK:
        V = _SkTfidf(sublinear_tf=True, ngram_range=(1, 2), min_df=1)
        X = V.fit_transform(profs)
        M = (X @ X.T).toarray()
        np.fill_diagonal(M, 0.0)
        ii, jj = np.where(M >= tau)
        for a, b in zip(ii.tolist(), jj.tolist()):
            if a < b:
                yield a, b
        return
    idf = Idf(profs)
    vecs = [idf.vec(p) for p in profs]
    inv = collections.defaultdict(list)
    for i, v in enumerate(vecs):
        for w in sorted(v, key=v.get, reverse=True)[:25]:
            inv[w].append(i)
    seen = set()
    for w, lst in inv.items():
        if len(lst) > 60:
            continue
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                k = (lst[a], lst[b])
                if k in seen:
                    continue
                seen.add(k)
                if cos(vecs[k[0]], vecs[k[1]]) >= tau:
                    yield k


def owner_proxy_clusters(recs, tau=OWNER_TAU):
    """Map group id -> owner-proxy cluster id via near-duplicate card templates."""
    profs = sorted({p for r in recs for p in r['profs']})
    pi = {p: i for i, p in enumerate(profs)}
    par = list(range(len(profs)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a, b in _profile_sim_pairs(profs, tau):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb

    gpar = {r['gi']: r['gi'] for r in recs}

    def gfind(x):
        while gpar[x] != x:
            gpar[x] = gpar[gpar[x]]
            x = gpar[x]
        return x

    byc = collections.defaultdict(set)
    for r in recs:
        for p in r['profs']:
            byc[find(pi[p])].add(r['gi'])
    for _, gs in byc.items():
        gs = sorted(gs)
        for x in gs[1:]:
            a, b = gfind(gs[0]), gfind(x)
            if a != b:
                gpar[a] = b
    return {r['gi']: gfind(r['gi']) for r in recs}


def owner_aware_cv(tr_cases, all_cases, tr_recs, n_folds=3, tau=OWNER_TAU):
    """Grouped CV where every owner-proxy cluster sits entirely in one fold."""
    cl = owner_proxy_clusters(tr_recs, tau)
    by = collections.defaultdict(list)
    for r in tr_recs:
        by[cl[r['gi']]].append(r)
    # greedy largest-first packing keeps folds balanced despite skewed clusters
    order = sorted(by.values(), key=len, reverse=True)
    folds = [[] for _ in range(n_folds)]
    for blk in order:
        folds[min(range(n_folds), key=lambda i: len(folds[i]))].extend(blk)

    TPt = FPt = TPe = FPe = ex = ncase = 0
    for fi in range(n_folds):
        va = {r['gi'] for r in folds[fi]}
        trr = [r for r in tr_recs if r['gi'] not in va]
        var = [r for r in tr_recs if r['gi'] in va]
        if not trr or not var:
            continue
        R = Router()
        R.fit_resources(all_cases)
        R.fit_evidence(trr)
        R.fit_task_classifier(trr, lambda r, p: r['gold_ev'].get(p, ''))
        R.fit_task(trr, lambda r, p: r['gold_ev'].get(p, ''))
        for r in var:
            a = R.decode(r)
            for i in r['idxs']:
                c = tr_cases[i]
                ncase += 1
                ok = True
                tmap = {t['task']: t['description'] for t in c['tasks']}
                emap = {e['evidence']: e['text'] for e in c['evid']}
                for p in c['profiles']:
                    gt, ge = c['gold'][p['model']]
                    pt, pe = a[p['text']]
                    if pt == tmap[gt]:
                        TPt += 1
                    else:
                        FPt += 1
                        ok = False
                    if pe == emap[ge]:
                        TPe += 1
                    else:
                        FPe += 1
                        ok = False
                ex += ok
    f1t = 2 * TPt / max(2 * TPt + 2 * FPt, 1)
    f1e = 2 * TPe / max(2 * TPe + 2 * FPe, 1)
    exr = ex / max(ncase, 1)
    return dict(task_f1=f1t, evidence_f1=f1e, exact=exr,
                score=0.45 * f1t + 0.30 * f1e + 0.25 * exr,
                n_clusters=len(by), n_cases=ncase)


def ledger_for_case(case, assign):
    """Render the group-level assignment with this view's local tokens."""
    tmap = {t['description']: t['task'] for t in case['tasks']}
    emap = {e['text']: e['evidence'] for e in case['evid']}
    atoms = []
    for p in case['profiles']:
        d, ev = assign[p['text']]
        atoms.append('%s>%s@%s' % (p['model'], tmap[d], emap[ev]))
    atoms.sort(key=lambda a: a.split('>')[0])
    return ';'.join(atoms)


# ==========================================================================
def main(data_dir, out_path):
    tr = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    te = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    tr_cases = parse(tr, True)
    te_cases = parse(te, False)

    R = Router()
    R.fit_resources(tr_cases + te_cases)
    tr_recs = build_groups(tr_cases, True)
    te_recs = build_groups(te_cases, False)

    # ---- owner-aware local validation on public training cases -------------
    # Reported for transparency; never allowed to break the prediction path.
    if os.environ.get('SKIP_LOCAL_CV', '') != '1':
        try:
            m = owner_aware_cv(tr_cases, tr_cases + te_cases, tr_recs, n_folds=3)
            print('owner-aware CV (%d owner-proxy clusters, %d cases): '
                  'TaskF1 %.4f  EvidenceF1 %.4f  Exact %.4f  SCORE %.4f'
                  % (m['n_clusters'], m['n_cases'], m['task_f1'],
                     m['evidence_f1'], m['exact'], m['score']))
        except Exception as exc:                                 # pragma: no cover
            print('owner-aware CV skipped (%s)' % exc)

    # ---- final fit on all public training cases ----------------------------
    R.fit_evidence(tr_recs)
    R.fit_task_classifier(tr_recs, lambda r, p: r['gold_ev'].get(p, ''))
    R.fit_task(tr_recs, lambda r, p: r['gold_ev'].get(p, ''))

    preds = {}
    for r in te_recs:
        assign = R.decode(r)
        for i in r['idxs']:
            preds[te_cases[i]['case_id']] = ledger_for_case(te_cases[i], assign)

    sub_path = os.path.join(data_dir, 'sample_submission.csv')
    ids = pd.read_csv(sub_path)['case_id'].tolist() if os.path.exists(sub_path) \
        else [c['case_id'] for c in te_cases]
    out = pd.DataFrame({'case_id': ids,
                        'predicted_ledger': [preds.get(i, '') for i in ids]})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    out.to_csv(out_path, index=False)
    print('wrote %s  (%d rows)' % (out_path, len(out)))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
