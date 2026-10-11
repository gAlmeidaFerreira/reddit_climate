"""
Transition tensors p[s, i, k] from the events tables (Stages 4-7 of the algorithm).

    s = own class, i = influencer category (l dyadic, g mean x variance, c composition), k = output class

The same functions serve both tensors; the only difference is the name of the influencer column:
    cfg.COL_L  -> 3 influencer categories (the parent's class)
    cfg.COL_G  -> 2m influencer categories (mean level x variance level of the group)
    cfg.COL_C  -> composition cells of the group (class counts or rounded shares, see events.composition_cells)

Estimator (cfg.TAU is the number of pseudo-events):
    p[s, i, :] = (N[s, i, :] + TAU * q[s, :]) / (R[s, i] + TAU),    q[s, k] = share of k among all events with own class s
    TAU = 0 gives Kozitsin's raw proportions; slices shrink toward the global row of the same (s, i) instead.

Usage:
    import events, tensors
    res = events.build_events()
    dy = tensors.fit_tensor(res["dyadic"], "l")
    gr = tensors.fit_tensor(res["group"], "g")
    dy["summary"]; dy["perm_s"]["p_value"]
    tensors.slice_tensors(res["dyadic"], "l")
    tensors.choose_tau(res["dyadic"], "l")

    python tensors.py            # self-test on synthetic data with a known tensor
"""
import warnings

import numpy as np
import pandas as pd

import config as cfg
import events

EPS = 1e-9          # the number of classes m is read from cfg.N_CLASSES at call time (cfg.set_classes)


# =======================================================================================
# Counting
# =======================================================================================
_EXTRA = {}          # derived influencer columns registered by compare_summaries: name -> (n, labels, lean array)


def register_influencer(name, n, labels, lean_of=None):
    """Make a derived column usable as an influencer in every function of this module."""
    _EXTRA[name] = (int(n), list(labels), lean_of)


def n_influencer(infl):
    """Number of influencer categories for the column `infl`."""
    if infl in _EXTRA:
        return _EXTRA[infl][0]
    if infl == cfg.COL_L:
        return cfg.N_CLASSES
    if infl in (cfg.COL_G, cfg.COL_H):
        return cfg.N_G
    if infl == cfg.COL_LEAN:
        return cfg.N_CLASSES
    if infl == cfg.COL_C:
        return len(events.composition_cells())
    raise ValueError(f"influencer column must be one of l, g, h, lean, c or a registered column, got {infl!r}")


def influencer_labels(infl):
    if infl in _EXTRA:
        return _EXTRA[infl][1]
    return events.influencer_info(infl)[1]


def lean(infl):
    """Class towards which each influencer category points: l itself; g: its mean level; c: the rounded
    mean class position of the composition. Used for the 'toward' column."""
    if infl in _EXTRA:
        if _EXTRA[infl][2] is None:
            raise ValueError(f"no lean registered for {infl!r}")
        return np.asarray(_EXTRA[infl][2])
    if infl in (cfg.COL_L, cfg.COL_LEAN):
        return np.arange(cfg.N_CLASSES)
    if infl in (cfg.COL_G, cfg.COL_H):
        return np.arange(cfg.N_G) // 2
    parts = np.array(events.composition_cells()["parts"].tolist(), dtype=float)
    pos = (parts * np.arange(cfg.N_CLASSES)).sum(1) / parts.sum(1)
    return np.floor(np.round(pos, 6) + 0.5).astype(int)


def usable(ev, infl):
    """Events with s, the influencer and k all present."""
    return ev.dropna(subset=[cfg.COL_S, infl, cfg.COL_K])


def _cells(d, infl):
    """Integer arrays s, i, k and the flat cell code s * (n_i * cfg.N_CLASSES) + i * cfg.N_CLASSES + k."""
    n_i = n_influencer(infl)
    s, i, k = (d[c].to_numpy(dtype=int) for c in (cfg.COL_S, infl, cfg.COL_K))
    return s, i, k, s * (n_i * cfg.N_CLASSES) + i * cfg.N_CLASSES + k


def count(ev, infl):
    """N[s, i, k]: number of events in each cell."""
    n_i = n_influencer(infl)
    code = _cells(usable(ev, infl), infl)[3]
    return np.bincount(code, minlength=cfg.N_CLASSES * n_i * cfg.N_CLASSES).reshape(cfg.N_CLASSES, n_i, cfg.N_CLASSES).astype(float)


# =======================================================================================
# Probabilities
# =======================================================================================
def pooled_prior(N):
    """q[s, k] (same for every influencer category): the share of k among all events with own class s.
    Works on a stack of tensors: the last three axes are (s, i, k)."""
    num = N.sum(-2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = num / num.sum(-1, keepdims=True)
    return np.broadcast_to(q, N.shape)


def shrink(N, prior, tau):
    """(N + tau * prior) / (R + tau), row by row. Rows with R + tau = 0 come out as NaN."""
    R = N.sum(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (N + tau * prior) / (R + tau)


def probabilities(N, tau=None):
    """Global estimator: shrink every row toward q[s, :]. tau = 0 gives the raw proportions."""
    tau = cfg.TAU if tau is None else tau
    return shrink(N, pooled_prior(N), tau)


# =======================================================================================
# User-level bootstrap
# =======================================================================================
def _boot_counts(users, code, ncell, B, seed):
    """B resamples of the USERS (with replacement). Returns B x ncell count matrix.
    A[u, c] = events of user u in cell c; a resample is a weight vector w over users, counts = w @ A."""
    rng = np.random.default_rng(seed)
    A = (pd.crosstab(np.asarray(users), code).reindex(columns=range(ncell), fill_value=0)
         .to_numpy(dtype=float))
    n = A.shape[0]
    out = np.empty((B, ncell))
    for b in range(B):
        out[b] = np.bincount(rng.integers(0, n, n), minlength=n) @ A
    return out


def bootstrap(ev, infl, B=None, tau=None, seed=None, prior=None):
    """B probability tensors, one per user-resample, shape (B, cfg.N_CLASSES, n_i, cfg.N_CLASSES).
    prior=None shrinks each replicate toward its own q; pass a tensor to shrink toward a fixed prior (slices)."""
    B, seed = cfg.B if B is None else B, cfg.SEED if seed is None else seed
    tau = cfg.TAU if tau is None else tau
    d = usable(ev, infl)
    n_i = n_influencer(infl)
    code = _cells(d, infl)[3]
    Nb = _boot_counts(d["user"].to_numpy(), code, cfg.N_CLASSES * n_i * cfg.N_CLASSES, B, seed).reshape(B, cfg.N_CLASSES, n_i, cfg.N_CLASSES)
    return shrink(Nb, pooled_prior(Nb) if prior is None else prior, tau)


# =======================================================================================
# Summary table
# =======================================================================================
def _intervals(Ps):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(Ps, cfg.CI, axis=0)           # (2, ...)


def summary(N, Ps, infl, P=None, tau=None):
    """One row per (own class, influencer category).
    stay / up / down: probability of keeping the class, moving to a higher / lower class.
    toward: the move in the direction of the influencer's class (for g: the group's mean level);
            undefined (NaN) when the influencer's class equals the own class.
    *_lo / *_hi: percentile interval over the bootstrap replicates Ps (NaN when Ps is None)."""
    tau = cfg.TAU if tau is None else tau
    n_i = n_influencer(infl)
    labels = influencer_labels(infl)
    P = probabilities(N, tau) if P is None else P
    R = N.sum(-1)

    leans = lean(infl)

    def moves(T, s, i):
        c = leans[i]
        up, down = T[..., s, i, s + 1:].sum(-1), T[..., s, i, :s].sum(-1)
        toward = up if c > s else down if c < s else np.full_like(up, np.nan)
        return {"stay": T[..., s, i, s], "up": up, "down": down, "toward": toward}

    ci = None if Ps is None else _intervals(Ps)
    rows = []
    for s in range(cfg.N_CLASSES):
        for i in range(n_i):
            n = int(R[s, i])
            prior_w = tau / (n + tau) if n + tau > 0 else np.nan
            flags = [name for name, bad in (("thin", n < cfg.MIN_ROW_EVENTS),
                                            ("prior", prior_w > cfg.MAX_PRIOR_WEIGHT)) if bad]
            row = {"s": cfg.CLASS_LABELS[s], infl: labels[i], "n_events": n,
                   "prior_weight": round(prior_w, 3) if prior_w == prior_w else np.nan,
                   "flag": ",".join(flags)}
            for k in range(cfg.N_CLASSES):
                row[f"p_{cfg.CLASS_LABELS[k]}"] = P[s, i, k]
            est = moves(P, s, i)
            for name, val in est.items():
                row[name] = val
                if ci is None:
                    row[name + "_lo"] = row[name + "_hi"] = np.nan
                else:
                    lo, hi = moves(ci[0], s, i)[name], moves(ci[1], s, i)[name]
                    row[name + "_lo"], row[name + "_hi"] = lo, hi
            rows.append(row)
    return pd.DataFrame(rows)


# =======================================================================================
# Null test: does the influencer tell us anything about k once s is known?
# =======================================================================================
def cmi_bits(N):
    """I(K; influencer | S) in bits from a count tensor (0 = the influencer is irrelevant)."""
    n = N.sum()
    if n == 0:
        return 0.0
    p = N / n
    p_s = p.sum((1, 2), keepdims=True)
    p_si = p.sum(2, keepdims=True)
    p_sk = p.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p * p_s / (p_si * p_sk)), 0.0)
    return float(terms.sum())


def entropy_k_given_s(N):
    """H(K | S) in bits: the uncertainty about k that the own class alone leaves."""
    n = N.sum()
    p_sk = N.sum(1) / n
    p_s = p_sk.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.where(p_sk > 0, p_sk * np.log2(p_sk / p_s), 0.0).sum())


def permutation_test(ev, infl, strata=(cfg.COL_S,), n_perm=None, seed=None):
    """Shuffle the influencer among events that share the same `strata` columns and recompute
    I(K; influencer | S). strata=("s",) is the basic null; ("s", "thread") also keeps the thread fixed,
    so that only within-thread differences in the influencer can explain k.
    Returns dict: cmi, null (array), p_value, cmi_excess (cmi - mean null), share (cmi / H(K|S))."""
    n_perm, seed = cfg.N_PERM if n_perm is None else n_perm, cfg.SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    d = usable(ev, infl)
    n_i = n_influencer(infl)
    s, i, k, _ = _cells(d, infl)
    strata_code = d.groupby(list(strata), dropna=False, sort=False).ngroup().to_numpy()
    base = np.argsort(strata_code, kind="stable")
    size = cfg.N_CLASSES * n_i * cfg.N_CLASSES

    def cmi_of(i_):
        return cmi_bits(np.bincount(s * (n_i * cfg.N_CLASSES) + i_ * cfg.N_CLASSES + k, minlength=size).reshape(cfg.N_CLASSES, n_i, cfg.N_CLASSES).astype(float))

    obs = cmi_of(i)
    null = np.empty(n_perm)
    for b in range(n_perm):
        order = np.lexsort((rng.random(len(i)), strata_code))      # random order inside each stratum
        shuffled = np.empty_like(i)
        shuffled[base] = i[order]
        null[b] = cmi_of(shuffled)
    h = entropy_k_given_s(count(ev, infl))
    return {"strata": tuple(strata), "cmi": obs, "null": null,
            "p_value": (1 + int((null >= obs).sum())) / (n_perm + 1),
            "cmi_excess": obs - float(null.mean()), "share": obs / h if h > 0 else np.nan}


# =======================================================================================
# Slices
# =======================================================================================
def _quarter(ev):
    return ev["t_reply"].dt.to_period(cfg.SLICE_FREQ).astype(str)


def slice_tensors(ev, infl, B=None, seed=None):
    """Tensors for each slice listed in cfg.SLICE_BY, shrunk toward the global row of the same (s, i)
    with strength cfg.TAU_SLICE. Slice-level intervals come from a user bootstrap inside the slice
    (B = 0 skips them). Returns one long table: slice, subreddit, quarter + the columns of summary()."""
    B = cfg.B if B is None else B
    ev = usable(ev, infl).assign(quarter=lambda d: _quarter(d))
    P_global = probabilities(count(ev, infl), cfg.TAU)

    parts = []
    if "subreddit" in cfg.SLICE_BY:
        parts += [("subreddit", sr, None, d) for sr, d in ev.groupby("subreddit")]
    if "quarter" in cfg.SLICE_BY:
        parts += [("quarter", None, q, d) for q, d in ev.groupby("quarter")]
    if "subreddit" in cfg.SLICE_BY and "quarter" in cfg.SLICE_BY:
        for sr in cfg.SLICE_PAIR_SUBREDDITS:
            parts += [("subreddit x quarter", sr, q, d) for q, d in ev[ev["subreddit"] == sr].groupby("quarter")]

    tables = []
    for name, sr, q, d in parts:
        N = count(d, infl)
        P = shrink(N, P_global, cfg.TAU_SLICE)
        Ps = bootstrap(d, infl, B, cfg.TAU_SLICE, seed, prior=P_global) if B > 0 else None
        t = summary(N, Ps, infl, P=P, tau=cfg.TAU_SLICE)
        t.insert(0, "quarter", q)
        t.insert(0, "subreddit", sr)
        t.insert(0, "slice", name)
        tables.append(t)
    return pd.concat(tables, ignore_index=True)


# =======================================================================================
# Choosing TAU by held-out log-loss
# =======================================================================================
def choose_tau(ev, infl, taus=(0, 5, 10, 25, 50, 100, 200, 500), n_folds=5, seed=None):
    """Cross-validation by USER: fit on the other folds, score the held-out events with the mean
    log-loss in bits (lower is better). The extra row 'no influencer' predicts k from s alone, so the
    gap to it is what the influencer adds. tau = 0 can give infinite loss when a held-out event falls
    in a cell with no training events (floored at EPS)."""
    seed = cfg.SEED if seed is None else seed
    d = usable(ev, infl)
    s, i, k, _ = _cells(d, infl)
    users = pd.factorize(d["user"])[0]
    fold_of_user = np.random.default_rng(seed).integers(0, n_folds, users.max() + 1)
    fold = fold_of_user[users]
    n_i = n_influencer(infl)
    losses = {t: [] for t in (*taus, "no influencer")}
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        N = np.bincount(s[tr] * (n_i * cfg.N_CLASSES) + i[tr] * cfg.N_CLASSES + k[tr], minlength=cfg.N_CLASSES * n_i * cfg.N_CLASSES).reshape(cfg.N_CLASSES, n_i, cfg.N_CLASSES).astype(float)
        for t in losses:
            P = pooled_prior(N) if t == "no influencer" else probabilities(N, t)
            p = P[s[te], i[te], k[te]]
            losses[t].append(-np.log2(np.clip(np.nan_to_num(p, nan=0.0), EPS, 1)).mean())
    return pd.DataFrame({"tau": list(losses), "logloss_bits": [np.mean(v) for v in losses.values()],
                         "fold_sd": [np.std(v) for v in losses.values()]})


# =======================================================================================
# Which summary of the group captures the influence on the individual? (held-out comparison)
# =======================================================================================
def _size_bin(n_members):
    return np.searchsorted(np.array(cfg.TEST_SIZE_EDGES), np.asarray(n_members), side="right")


def _dname(w):
    return f"D({w:g})"


def _derived_table(ev):
    """Events with every summary that will be compared, on the SAME events, as integer columns.
    none (1 category), size (group-size class), lean (mean level), lean_size, g (mean x variance),
    h (mean x opposed pairs), gh (mean x variance x opposed pairs) and c (composition, when present)."""
    m, nb = cfg.N_CLASSES, len(cfg.TEST_SIZE_EDGES) + 1
    need = [cfg.COL_S, cfg.COL_K, cfg.COL_G, cfg.COL_H, cfg.COL_LEAN, "n_members", "opp", "mild", "var", "user"]
    d = ev.dropna(subset=need).copy()
    for c in need[:-4]:
        d[c] = d[c].astype(int)
    target = float((d["var"] >= cfg.VAR_CUT).mean())               # every 'high' label keeps this share of the events
    d["mlev"] = (d["mild"] >= events.matched_cut(d["mild"], target)).astype(int)
    for w in cfg.DISAGREE_WEIGHTS:
        D = d["opp"] + w * d["mild"]
        d[_dname(w)] = 2 * d[cfg.COL_LEAN] + (D >= events.matched_cut(D, target)).astype(int)
    d["none"] = 0
    d["size"] = _size_bin(d["n_members"])
    d["vlev"] = d[cfg.COL_G] % 2
    d["olev"] = d[cfg.COL_H] % 2
    d["lean_size"] = d[cfg.COL_LEAN] * nb + d["size"]
    d["gh"] = d[cfg.COL_G] * 2 + d["olev"]
    lean = np.arange(m)
    spec = {"none": (1, ["no group information"], [0]),
            "size": (nb, [f"size class {i}" for i in range(nb)], [0] * nb),
            cfg.COL_LEAN: (m, list(cfg.MEAN_LABELS), lean),
            "lean_size": (m * nb, [f"{a} | size {i}" for a in cfg.MEAN_LABELS for i in range(nb)], np.repeat(lean, nb)),
            cfg.COL_G: (cfg.N_G, list(cfg.G_LABELS), np.arange(cfg.N_G) // 2),
            cfg.COL_H: (cfg.N_G, list(cfg.H_LABELS), np.arange(cfg.N_G) // 2),
            "gh": (4 * m, [f"{g} + {o}" for g in cfg.G_LABELS for o in cfg.OPP_LABELS], np.arange(4 * m) // 4),
            "vlev": (2, list(cfg.VAR_LABELS), [0, 0]), "olev": (2, list(cfg.OPP_LABELS), [0, 0]),
            "mlev": (2, ["low-mild", "high-mild"], [0, 0])}
    for w in cfg.DISAGREE_WEIGHTS:
        spec[_dname(w)] = (cfg.N_G, [f"{a}-{b}" for a in cfg.MEAN_LABELS for b in ("low", "high")], np.arange(cfg.N_G) // 2)
    if cfg.COL_C in d and d[cfg.COL_C].notna().all() and len(d):
        d[cfg.COL_C] = d[cfg.COL_C].astype(int)
        reg = events.composition_cells()
        spec[cfg.COL_C] = (len(reg), reg["label"].tolist(), lean_c())
    _EXTRA.clear()                                                  # sizes depend on m; built-in columns resolve themselves
    builtin = (cfg.COL_G, cfg.COL_H, cfg.COL_LEAN, cfg.COL_C)
    for name, (n, labels, ln) in spec.items():
        if name not in builtin:
            register_influencer(name, n, labels, ln)
    return d, [k for k in spec if k not in ("vlev", "olev", "mlev")]


def lean_c():
    parts = np.array(events.composition_cells()["parts"].tolist(), dtype=float)
    return np.floor(np.round((parts * np.arange(cfg.N_CLASSES)).sum(1) / parts.sum(1), 6) + 0.5).astype(int)


def _heldout_loss(d, infl, taus, n_folds, seed):
    """Per-event log-loss (bits) of tensor p[s, i, k] fitted on the other user folds. tau is chosen inside
    the training folds (inner user folds), so the choice never sees the held-out events. The folds are the
    same for every summary, which makes the losses of two summaries paired."""
    s, i, k = (d[c].to_numpy(dtype=int) for c in (cfg.COL_S, infl, cfg.COL_K))
    n_i, m = n_influencer(infl), cfg.N_CLASSES
    code = s * (n_i * m) + i * m + k
    users = pd.factorize(d["user"])[0]
    nu = users.max() + 1
    outer = np.random.default_rng(seed).integers(0, n_folds, nu)[users]

    def counts(mask):
        return np.bincount(code[mask], minlength=m * n_i * m).reshape(m, n_i, m).astype(float)

    def loss_of(P, idx):
        p = P[s[idx], i[idx], k[idx]]
        return -np.log2(np.clip(np.nan_to_num(p, nan=0.0), EPS, 1))

    loss, chosen = np.empty(len(d)), []
    for f in range(n_folds):
        train, test = outer != f, outer == f
        inner = np.random.default_rng(seed + 1 + f).integers(0, n_folds, nu)[users]
        total = dict.fromkeys(taus, 0.0)
        for g in range(n_folds):
            N = counts(train & (inner != g))
            val = np.flatnonzero(train & (inner == g))
            for t in taus:
                total[t] += loss_of(probabilities(N, t), val).sum()
        best = min(total, key=total.get)
        chosen.append(best)
        loss[test] = loss_of(probabilities(counts(train), best), np.flatnonzero(test))
    return loss, chosen


def _paired(diff, users, B, seed):
    """Mean of per-event differences with a user-cluster bootstrap interval."""
    u = pd.factorize(users)[0]
    sums, cnt = np.bincount(u, weights=diff), np.bincount(u).astype(float)
    rng = np.random.default_rng(seed)
    n = len(sums)
    boots = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n)
        boots[b] = (w * sums).sum() / (w * cnt).sum()
    lo, hi = np.percentile(boots, cfg.CI)
    return float(sums.sum() / cnt.sum()), float(lo), float(hi)


def compare_summaries(ev, ref=None, n_perm=None, seed=None, tests=True):
    """Which summary of the group tells most about the replier's new opinion k?
    Works on events that carry g, h, lean and n_members (the group or composition table), restricted to the
    rows where every summary exists, so all of them are scored on exactly the same events.
    table: held-out log-loss (bits per event, user folds, tau chosen inside the training folds), the gain
           over 'none' and the paired difference to the reference summary `ref` (default g) with a user-bootstrap
           interval; positive 'better_than_ref' means lower loss than the reference.
    added: does the second coordinate add information beyond the other one? Permutation test of
           I(K; coordinate | s, lean, size [, other coordinate]) with the shuffle inside those strata;
           cmi_excess = observed - mean of the null (bits per event).
    family: D(w) = O + w * M for the weights of cfg.DISAGREE_WEIGHTS (lean level x high/low level of D(w), the cut
           matched to the same share of 'high' events as VAR_CUT): is it worth counting mild disagreement,
           and with which weight? better_than_w0 > 0 means that adding the mild pairs lowers the loss."""
    seed = cfg.SEED if seed is None else seed
    ref = cfg.COL_G if ref is None else ref
    d, names = _derived_table(ev)
    if len(d) == 0:
        raise ValueError("no event has all the columns needed for the comparison")
    losses, taus = {}, {}
    for name in names:
        losses[name], taus[name] = _heldout_loss(d, name, cfg.COMPARE_TAUS, cfg.COMPARE_FOLDS, seed)
    base, rows = losses["none"], []
    for name in names:
        diff = losses[ref] - losses[name]
        mean, lo, hi = _paired(diff, d["user"].to_numpy(), cfg.COMPARE_B, seed)
        rows.append({"summary": name, "categories": n_influencer(name), "logloss_bits": losses[name].mean(),
                     "gain_vs_none": base.mean() - losses[name].mean(), "median_tau": float(np.median(taus[name])),
                     "better_than_ref": mean, "ref_lo": lo, "ref_hi": hi})
    table = pd.DataFrame(rows).set_index("summary")
    fam = table.loc[[_dname(w) for w in cfg.DISAGREE_WEIGHTS], ["logloss_bits", "gain_vs_none", "better_than_ref", "ref_lo", "ref_hi"]].copy()
    fam.index = pd.Index(list(cfg.DISAGREE_WEIGHTS), name="w (weight of a mild pair)")
    fam["better_than_w0"] = fam["logloss_bits"].iloc[0] - fam["logloss_bits"]
    out = {"table": table, "family": fam, "n_events": len(d), "n_users": d["user"].nunique(), "ref": ref, "losses": losses}
    if tests:
        base_strata = (cfg.COL_S, cfg.COL_LEAN, "size")
        spec = [("opposed pairs | s, lean, size", "olev", base_strata),
                ("variance      | s, lean, size", "vlev", base_strata),
                ("opposed pairs | s, lean, size, variance", "olev", base_strata + ("vlev",)),
                ("variance      | s, lean, size, opposed pairs", "vlev", base_strata + ("olev",)),
                ("mild pairs    | s, lean, size, opposed pairs", "mlev", base_strata + ("olev",)),
                ("opposed pairs | s, lean, size, mild pairs", "olev", base_strata + ("mlev",))]
        rows = []
        for label, col, strata in spec:
            r = permutation_test(d, col, strata, n_perm, seed)
            rows.append({"coordinate | held fixed": label, "cmi_bits": r["cmi"], "null_mean": float(r["null"].mean()),
                         "cmi_excess": r["cmi_excess"], "p_value": r["p_value"]})
        out["added"] = pd.DataFrame(rows).set_index("coordinate | held fixed")
    return out


# =======================================================================================
# Everything for one tensor
# =======================================================================================
def fit_tensor(ev, infl, B=None, n_perm=None, tau=None, seed=None):
    """Counts, probabilities, bootstrap, summary table and both permutation nulls for one tensor."""
    tau = cfg.TAU if tau is None else tau
    N = count(ev, infl)
    Ps = bootstrap(ev, infl, B, tau, seed)
    return {"N": N, "P": probabilities(N, tau), "boot": Ps, "summary": summary(N, Ps, infl, tau=tau),
            "perm_s": permutation_test(ev, infl, (cfg.COL_S,), n_perm, seed),
            "perm_s_thread": permutation_test(ev, infl, (cfg.COL_S, "thread"), n_perm, seed)}


# =======================================================================================
# Self-test on synthetic events with a known tensor
# =======================================================================================
def _synthetic_events(n=30000, n_users=600, dependent=True, seed=3, with_comp=False):
    rng = np.random.default_rng(seed)
    P_true = np.empty((cfg.N_CLASSES, cfg.N_CLASSES, cfg.N_CLASSES))
    for s in range(cfg.N_CLASSES):
        for l in range(cfg.N_CLASSES):
            w = np.full(cfg.N_CLASSES, .1)
            w[s] += 1.0
            if dependent:
                w[l] += 0.8                                        # pulled toward the influencer
            P_true[s, l] = w / w.sum()
    s, l = rng.integers(0, cfg.N_CLASSES, n), rng.integers(0, cfg.N_CLASSES, n)
    k = np.array([rng.choice(cfg.N_CLASSES, p=P_true[a, b]) for a, b in zip(s, l)])
    g = 2 * l + rng.integers(0, 2, n)                              # 2m categories, mean level = l
    c = None
    if with_comp:                                                  # group of 3: majority class = l, so the lean of c is l
        reg = events.composition_cells()
        cells = {tuple(p): i for p, i in zip(reg["parts"], reg["c"])}
        c = []
        for a in l:
            cnt = np.zeros(cfg.N_CLASSES, int)
            cnt[a] = 2
            cnt[rng.integers(0, cfg.N_CLASSES)] += 1
            c.append(cells[tuple(cnt)])
    ev = pd.DataFrame({"user": rng.integers(0, n_users, n), "thread": rng.integers(0, 3000, n),
                       "subreddit": rng.choice(["r/2", "r/4"], n, p=[.7, .3]),
                       "t_reply": pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 700, n), "D"),
                       cfg.COL_S: s, cfg.COL_L: l, cfg.COL_G: g, cfg.COL_C: c if c is not None else pd.NA, cfg.COL_K: k})
    for col in (cfg.COL_S, cfg.COL_L, cfg.COL_G, cfg.COL_C, cfg.COL_K):
        ev[col] = ev[col].astype("Int64")
    return ev, P_true


def _synthetic_groups(rule, n=40000, n_users=800, seed=11):
    """Events whose groups are built from raw scores, so that variance and opposed pairs are both computed the
    way events.py computes them. The new class k depends on s and the lean in both rules; in addition
        rule = 'opposed' : a group with many opposed pairs pushes k to the neutral class
        rule = 'variance': a group with high variance pushes k to the neutral class (opposed pairs play no role)
        rule = 'mild'    : a group with many mildly disagreeing pairs pushes k to the neutral class"""
    rng = np.random.default_rng(seed)
    m = cfg.N_CLASSES
    cuts = list(cfg.CUTS)
    edges = [0.0] + cuts + [1.0]
    size = rng.choice([2, 3, 4, 5, 6, 8], n)
    kind = rng.random(n)
    rows = []
    for a in range(n):
        if kind[a] < 0.3:                                          # two camps (plus a few neutrals)
            cls = rng.choice([0, m - 1] + ([1] if m == 3 else []), size[a], p=[.45, .45, .1] if m == 3 else [.5, .5])
        else:                                                      # ordinary group
            cls = rng.choice(m, size[a], p=[.5, .27, .23] if m == 3 else [.5, .5])
        x = rng.uniform(np.take(edges[:-1], cls), np.take(edges[1:], cls))
        rows.append((len(x), x.mean(), x.var(ddof=1) if len(x) > 1 else np.nan, np.bincount(cls, minlength=m)))
    g_n = np.array([r[0] for r in rows])
    d = pd.DataFrame({"g_n": g_n, "g_mu": [r[1] for r in rows], "g_var": [r[2] for r in rows]})
    d["g_opp"], d["g_mild"] = events._pair_stats(np.array([r[3] for r in rows]))
    cut = events.opp_cut_for(d)
    ev = events._format(
        pd.DataFrame({"thread": rng.integers(0, 4000, n), "subreddit": "r/2", "user": rng.integers(0, n_users, n),
                      "t_reply": pd.Timestamp("2021-06-01"), "gap": 1.0, "parent_kind": "comment",
                      cfg.COL_S: rng.integers(0, m, n), cfg.COL_K: 0}),
        **events._group_cols(d, cut))
    s, lean = ev[cfg.COL_S].to_numpy(int), ev[cfg.COL_LEAN].to_numpy(int)
    if rule == "opposed":
        trigger = ev[cfg.COL_H].to_numpy(int) % 2
    elif rule == "variance":
        trigger = ev[cfg.COL_G].to_numpy(int) % 2
    else:
        trigger = (ev["mild"] >= events.matched_cut(ev["mild"], float((ev["var"] >= cfg.VAR_CUT).mean()))).to_numpy(int)
    k = np.empty(n, int)
    for a in range(n):
        w = np.full(m, .15)
        w[s[a]] += 1.0
        w[lean[a]] += 0.6
        if trigger[a]:
            w[1 if m == 3 else 0] += 1.6
        k[a] = rng.choice(m, p=w / w.sum())
    ev[cfg.COL_K] = pd.array(k, dtype="Int64")
    return ev


def _selftest_one(verbose=True):
    ev, P_true = _synthetic_events()

    # counting against a plain loop
    N = count(ev, cfg.COL_L)
    N_loop = np.zeros((cfg.N_CLASSES, cfg.N_CLASSES, cfg.N_CLASSES))
    for a, b, c in zip(ev[cfg.COL_S], ev[cfg.COL_L], ev[cfg.COL_K]):
        N_loop[a, b, c] += 1
    assert (N == N_loop).all() and N.sum() == len(ev)

    # tau = 0 is the raw proportion, every row sums to 1, large tau collapses to q
    P0 = probabilities(N, 0)
    assert np.allclose(P0, N / N.sum(-1, keepdims=True)) and np.allclose(P0.sum(-1), 1)
    assert np.allclose(probabilities(N, 1e9), pooled_prior(N), atol=1e-4)
    assert np.allclose(probabilities(N, 50).sum(-1), 1)

    # bootstrap: shape, and the true tensor lies inside the 95% interval for most cells
    Ps = bootstrap(ev, cfg.COL_L, B=200, tau=0)
    assert Ps.shape == (200, cfg.N_CLASSES, cfg.N_CLASSES, cfg.N_CLASSES)
    lo, hi = np.percentile(Ps, [2.5, 97.5], axis=0)
    coverage = float(((P_true >= lo) & (P_true <= hi)).mean())
    assert coverage >= 0.8, f"bootstrap coverage too low: {coverage:.2f}"

    # permutation null: small p-value when k depends on the influencer, large when it does not
    dep = permutation_test(ev, cfg.COL_L, n_perm=100)
    ev0, _ = _synthetic_events(dependent=False)
    ind = permutation_test(ev0, cfg.COL_L, n_perm=100)
    assert dep["p_value"] < 0.02 and ind["p_value"] > 0.02, (dep["p_value"], ind["p_value"])
    assert dep["cmi"] > ind["cmi"]

    # the group tensor (6 categories) and the summary table
    g = fit_tensor(ev, cfg.COL_G, B=50, n_perm=20)
    assert g["N"].shape == (cfg.N_CLASSES, cfg.N_G, cfg.N_CLASSES) and len(g["summary"]) == cfg.N_CLASSES * cfg.N_G
    # in the data the group's mean level IS l, so the true 'toward' probability is known
    sm = g["summary"].copy()
    sm["c"] = [i // 2 for i in np.tile(np.arange(cfg.N_G), cfg.N_CLASSES)]
    sm["s_idx"] = np.repeat(np.arange(cfg.N_CLASSES), cfg.N_G)
    true = [P_true[a, c, a + 1:].sum() if c > a else P_true[a, c, :a].sum() if c < a else np.nan
            for a, c in zip(sm["s_idx"], sm["c"])]
    assert np.allclose(sm["toward"].to_numpy(float), np.array(true, float), atol=0.03, equal_nan=True), \
        "estimated 'toward' differs from the true one"

    # the composition tensor: shape, counts against a loop, the null rejects, 'toward' is defined by the lean
    evc, _ = _synthetic_events(with_comp=True)
    cc = fit_tensor(evc, cfg.COL_C, B=30, n_perm=50)
    n_c = n_influencer(cfg.COL_C)
    assert cc["N"].shape == (cfg.N_CLASSES, n_c, cfg.N_CLASSES) and len(cc["summary"]) == cfg.N_CLASSES * n_c
    Nl = np.zeros_like(cc["N"])
    for a, b, k_ in zip(evc[cfg.COL_S], evc[cfg.COL_C], evc[cfg.COL_K]):
        Nl[a, b, k_] += 1
    assert (cc["N"] == Nl).all()
    assert cc["perm_s"]["p_value"] < 0.05, cc["perm_s"]["p_value"]
    assert set(lean(cfg.COL_C)) <= set(range(cfg.N_CLASSES)) and len(lean(cfg.COL_C)) == n_c

    # which summary captures the influence? two worlds, in which the truth is known
    if cfg.N_CLASSES == 3:
        res_ = {}
        for rule in ("opposed", "variance", "mild"):
            evg = _synthetic_groups(rule)
            cmp_ = compare_summaries(evg, n_perm=60)
            res_[rule] = cmp_
            tb = cmp_["table"]
            assert tb.loc["none", "logloss_bits"] > tb.loc["lean", "logloss_bits"] > 0
            assert abs(tb.loc["none", "gain_vs_none"]) < 1e-12
        a, b = res_["opposed"]["table"], res_["variance"]["table"]
        assert a.loc["h", "logloss_bits"] < a.loc["g", "logloss_bits"], "h should win when opposed pairs drive k"
        assert b.loc["g", "logloss_bits"] < b.loc["h", "logloss_bits"], "g should win when the variance drives k"
        ad_a, ad_b = res_["opposed"]["added"], res_["variance"]["added"]
        assert ad_a.iloc[2]["p_value"] < 0.05, "opposed pairs should add information beyond the variance"
        assert ad_b.iloc[3]["p_value"] < 0.05, "the variance should add information beyond opposed pairs"
        assert "better_than_ref" in a.columns and a.loc["g", "better_than_ref"] == 0.0
        fa, fm = res_["opposed"]["family"], res_["mild"]["family"]
        assert fa["logloss_bits"].iloc[0] < fa["logloss_bits"].iloc[-1], "w = 0 should beat w = 1 when only opposed pairs matter"
        assert fm["logloss_bits"].iloc[-1] < fm["logloss_bits"].iloc[0], "w = 1 should beat w = 0 when mild pairs drive k"
        assert res_["mild"]["added"].iloc[4]["p_value"] < 0.05, "mild pairs should add information beyond opposed pairs"
        assert res_["opposed"]["added"].iloc[4]["p_value"] > 0.05, "mild pairs should add nothing when only opposed pairs matter"

    # slices: counts add up to the total and quarters cover the whole period
    sl = slice_tensors(ev, cfg.COL_L, B=20)
    for name in ("subreddit", "quarter"):
        tot = sl[sl["slice"] == name].groupby(["s", cfg.COL_L], sort=False)["n_events"].sum()
        assert (tot.values == N.sum(-1).reshape(-1)).all(), f"slice counts of '{name}' do not add up"

    # held-out log-loss: the influencer helps, and some shrinkage is not worse than none
    ct = choose_tau(ev, cfg.COL_L, taus=(0, 50), n_folds=4).set_index("tau")["logloss_bits"]
    assert ct[50] < ct["no influencer"], "the influencer should reduce held-out log-loss"
    assert ct[50] <= ct[0] + 1e-3

    if verbose:
        print(f"selftest m={cfg.N_CLASSES} OK: bootstrap coverage {coverage:.2f}; dependent p={dep['p_value']:.3f}, "
              f"independent p={ind['p_value']:.3f}; held-out log-loss tau=50 {ct[50]:.4f} vs "
              f"no influencer {ct['no influencer']:.4f} bits")
    return True


def selftest(verbose=True):
    """Run the self-test for m = 3 and m = 2 and restore the configuration."""
    old = cfg.N_CLASSES
    try:
        for m in (3, 2):
            cfg.set_classes(m)
            _selftest_one(verbose)
    finally:
        cfg.set_classes(old)
    return True


if __name__ == "__main__":
    selftest()