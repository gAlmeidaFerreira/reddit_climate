"""
Events table: one row per reply that can be used to estimate a transition (Stages 1-3 of the algorithm).

    raw CSVs -> items -> replies (+ parent) -> own opinion s -> group statistics -> events

Three tables come out, with the same columns (cfg.EVENT_COLS):
    dyadic  events: influencer l (class of the parent's score), g and c are empty
    group   events: influencer g (2*m categories from mean and variance of raw scores), l and c are empty
    comp    events: influencer c (composition of the group: class counts, or rounded shares) AND g,
            so the two summaries can be compared on exactly the same replies
The number of classes m (2 or 3) is cfg.N_CLASSES; change it with cfg.set_classes(m) before building.

Rules that protect against leakage:
    * own opinion s = class of the replier's last scored item strictly BEFORE the parent's timestamp
    * group members are strictly earlier than the reply (ties excluded) and written by other authors
    * the reply itself and any item of the replier are never part of the group
    * the group mean and variance use raw scores; they are binned once, at the end

Usage:
    import events
    res = events.build_events()            # reads data/ (cache or CSVs) according to config.py
    res["dyadic"], res["group"], res["comp"], res["funnel_*"]
    events.compare_fill(res)               # how many events per (s, influencer) row for l, g and c
    events.save_events(res)

    python events.py                       # runs the brute-force self-test on synthetic data
"""
import re
import warnings

import numpy as np
import pandas as pd

import config as cfg

PARENT_KEY = ["subreddit", "parent_kind", "parent_item"]


# =======================================================================================
# 0. Loading
# =======================================================================================
def load_raw(use_cache=True):
    """Return (threads, submissions) as raw DataFrames.
    Uses data/cache/{threads,submissions}.parquet when present, otherwise reads every CSV."""
    cache_t, cache_s = cfg.DATA_CACHE / "threads.parquet", cfg.DATA_CACHE / "submissions.parquet"
    if use_cache and cache_t.exists() and cache_s.exists():
        return pd.read_parquet(cache_t), pd.read_parquet(cache_s)
    out = []
    for folder in (cfg.THREADS_DIR, cfg.SUBMISSIONS_DIR):
        files = sorted(folder.rglob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no CSV files under {folder}; extract the zip into data/raw/ "
                                    f"or fix the paths in config.py")
        out.append(pd.concat((pd.read_csv(f, dtype=str) for f in files), ignore_index=True))
    cfg.DATA_CACHE.mkdir(parents=True, exist_ok=True)
    threads, submissions = out
    for df, path in ((threads, cache_t), (submissions, cache_s)):
        df[cfg.RAW_SCORE] = pd.to_numeric(df[cfg.RAW_SCORE], errors="coerce")   # "NI" -> NaN
        df.to_parquet(path)
    return threads, submissions


def to_class(x, cuts):
    """Raw score -> class 0 .. len(cuts) (NaN stays NaN).
    x < cuts[0]: 0 ; cuts[0] <= x <= cuts[1]: 1 ; x > cuts[1]: 2 (one cut: x < cuts[0] -> 0, else 1)."""
    x = np.asarray(x, dtype=float)
    cls = (x >= cuts[0]).astype(float)
    for c in cuts[1:]:
        cls = cls + (x > c)
    return np.where(np.isnan(x), np.nan, cls)


def count_cols():
    """Names of the per-class member counts of a group: g_c0 .. g_c{m-1}."""
    return [f"g_c{j}" for j in range(cfg.N_CLASSES)]


# =======================================================================================
# 1. Items: every comment and every submission, one common format
# =======================================================================================
def build_items(threads, submissions):
    """columns: subreddit, item_id, kind, thread, parent_item, parent_kind, user, t, x"""
    prefix = re.compile(f"^({re.escape(cfg.PARENT_PREFIX_COMMENT)}|{re.escape(cfg.PARENT_PREFIX_SUBMISSION)})")

    def time(s):
        return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert(None)   # naive UTC

    def thread_id(s):
        return s.astype(object).str.replace(f"^{re.escape(cfg.PARENT_PREFIX_SUBMISSION)}", "", regex=True)

    pid = threads[cfg.RAW_PARENT_ID].astype(object)
    parent_kind = np.where(pid.str.startswith(cfg.PARENT_PREFIX_COMMENT, na=False), "comment",
                  np.where(pid.str.startswith(cfg.PARENT_PREFIX_SUBMISSION, na=False), "submission", None))
    com = pd.DataFrame({
        "subreddit": threads[cfg.RAW_SUBREDDIT_ID].to_numpy(),
        "item_id": threads[cfg.RAW_COMMENT_ID].to_numpy(),
        "kind": "comment",
        "thread": thread_id(threads[cfg.RAW_SUBMISSION_ID]).to_numpy(),
        "parent_item": pid.str.replace(prefix, "", regex=True).to_numpy(),
        "parent_kind": parent_kind,
        "user": threads[cfg.RAW_AUTHOR_ID].to_numpy(),
        "t": time(threads[cfg.RAW_TIME]).to_numpy(),
        "x": pd.to_numeric(threads[cfg.RAW_SCORE], errors="coerce").to_numpy()})
    sub = pd.DataFrame({
        "subreddit": submissions[cfg.RAW_SUBREDDIT_ID].to_numpy(),
        "item_id": submissions[cfg.RAW_SUBMISSION_ID].to_numpy(),
        "kind": "submission",
        "thread": thread_id(submissions[cfg.RAW_SUBMISSION_ID]).to_numpy(),
        "parent_item": None, "parent_kind": None,
        "user": submissions[cfg.RAW_AUTHOR_ID].to_numpy(),
        "t": time(submissions[cfg.RAW_TIME]).to_numpy(),
        "x": pd.to_numeric(submissions[cfg.RAW_SCORE], errors="coerce").to_numpy()})
    return pd.concat([sub, com], ignore_index=True)


# =======================================================================================
# 2. Replies: every comment with its parent attached
# =======================================================================================
def build_replies(items):
    """One row per comment. Adds t_reply, parent_user, t_parent, x_parent, parent_found."""
    key = ["subreddit", "parent_item", "parent_kind"]
    parents = (items[["subreddit", "item_id", "kind", "user", "t", "x"]]
               .rename(columns={"item_id": "parent_item", "kind": "parent_kind", "user": "parent_user",
                                "t": "t_parent", "x": "x_parent"})
               .drop_duplicates(key))                  # a duplicated key would multiply reply rows
    com = items[items["kind"] == "comment"].rename(columns={"t": "t_reply"})
    rep = com.merge(parents, on=key, how="left").reset_index(drop=True)
    assert len(rep) == len(com), "parent merge changed the number of replies"
    rep["parent_found"] = rep["t_parent"].notna()
    return rep


# =======================================================================================
# 3. Own opinion s (and the output k)
# =======================================================================================
def add_own_opinion(rep, items):
    """s = class of the replier's last scored item strictly before the parent's timestamp.
    Adds x_prev, t_prev, gap (days between that item and the parent), s and k."""
    hist = (items.loc[items["user"].notna() & items["x"].notna(), ["user", "t", "x"]]
            .rename(columns={"t": "t_prev", "x": "x_prev"}).sort_values("t_prev"))
    ok = rep["user"].notna() & rep["t_parent"].notna()
    left = rep.loc[ok, ["user", "t_parent"]].assign(_row=rep.index[ok]).sort_values("t_parent")
    got = pd.merge_asof(left, hist, left_on="t_parent", right_on="t_prev", by="user",
                        allow_exact_matches=False).set_index("_row")
    rep = rep.copy()
    rep["x_prev"], rep["t_prev"] = got["x_prev"], got["t_prev"]
    rep["gap"] = (rep["t_parent"] - rep["t_prev"]).dt.total_seconds() / 86400
    rep[cfg.COL_S] = to_class(rep["x_prev"], cfg.CUTS)
    rep[cfg.COL_K] = to_class(rep["x"], cfg.CUTS)
    return rep


# =======================================================================================
# 4. Group statistics of every reply (raw scores; binning happens later)
# =======================================================================================
def _pool(rep):
    """Scored replies with a known author and a parent reference: they are both the focal
    replies and the possible siblings inside a parent's group."""
    ok = rep["x"].notna() & rep["user"].notna() & rep["parent_item"].notna() & rep["parent_kind"].notna()
    cols = PARENT_KEY + ["user", "t_reply", "x", "parent_found", "parent_user", "t_parent", "x_parent"]
    return rep.loc[ok, cols].copy()


def _strictly_before(p, grp, v):
    """For each row: sum of v over the rows of the same `grp` with a STRICTLY earlier t_reply.
    `p` must be sorted by (_pg, t_reply). Rows with equal timestamps are not counted."""
    exclusive = p.groupby(grp)[v].cumsum() - p[v]              # includes earlier rows of a tie block
    return exclusive.groupby([p[grp], p["t_reply"]]).transform("first")   # value at the start of the block


def _group_fast(rep):
    """Vectorised: members = parent (if scored, other author) + earlier scored siblings by other authors.
    Returns g_n, g_mu, g_var (population variance) and the class counts g_c0 .. g_c{m-1}."""
    p = _pool(rep)
    p["_pg"] = p.groupby(PARENT_KEY, sort=False).ngroup()
    p = p.sort_values(["_pg", "t_reply"], kind="stable")
    p["_pu"] = p.groupby(["_pg", "user"], sort=False).ngroup()
    cls = to_class(p["x"], cfg.CUTS)
    cols = {"v0": 1.0, "v1": p["x"], "v2": p["x"] ** 2}
    for j, name in enumerate(count_cols()):
        cols[name] = (cls == j).astype(float)
    for name, v in cols.items():
        p[name] = v
    tot = {name: (_strictly_before(p, "_pg", name) - _strictly_before(p, "_pu", name)).to_numpy()
           for name in cols}                                       # all earlier - replier's own
    from_parent = (p["parent_found"] & p["x_parent"].notna() & (p["parent_user"] != p["user"])).to_numpy()
    xp = p["x_parent"].fillna(0.0).to_numpy()
    cp = to_class(p["x_parent"], cfg.CUTS)
    n = tot["v0"] + from_parent
    s1 = tot["v1"] + from_parent * xp
    s2 = tot["v2"] + from_parent * xp ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(n > 0, s1 / n, np.nan)
        var = np.where(n > 0, np.maximum(s2 / n - mu ** 2, 0.0), np.nan)
    out = {"g_n": n, "g_mu": mu, "g_var": var}
    for j, name in enumerate(count_cols()):
        out[name] = tot[name] + from_parent * (cp == j)
    return pd.DataFrame(out, index=p.index)


def _group_slow(rep):
    """Plain loop, one reply at a time. Reference implementation for the test, and the only path
    that supports cfg.AUTHORS_ONCE and cfg.M_MAX."""
    p = _pool(rep)
    out = {}
    for _, grp in p.groupby(PARENT_KEY, sort=False):
        t = grp["t_reply"].to_numpy().astype("datetime64[ns]").astype("int64")
        u = grp["user"].to_numpy(object)
        x = grp["x"].to_numpy(float)
        for i, idx in enumerate(grp.index):
            members = []                                           # (author, time, score)
            xp, pu = grp["x_parent"].iat[i], grp["parent_user"].iat[i]
            if grp["parent_found"].iat[i] and not np.isnan(xp) and pu != u[i]:
                members.append((pu, grp["t_parent"].iat[i].value, xp))
            m = (t < t[i]) & (u != u[i])
            members += list(zip(u[m], t[m], x[m]))
            if cfg.AUTHORS_ONCE:                                   # latest item per author
                best = {}
                for a, tt, xx in members:
                    a = "__missing__" if pd.isna(a) else a
                    if a not in best or tt > best[a][1]:
                        best[a] = (a, tt, xx)
                members = list(best.values())
            if cfg.M_MAX is not None:                              # most recent M_MAX members
                members = sorted(members, key=lambda r: r[1], reverse=True)[:cfg.M_MAX]
            xs = np.array([r[2] for r in members], dtype=float)
            counts = np.bincount(to_class(xs, cfg.CUTS).astype(int), minlength=cfg.N_CLASSES) if len(xs) \
                else np.zeros(cfg.N_CLASSES, int)
            out[idx] = (len(xs), xs.mean() if len(xs) else np.nan, xs.var() if len(xs) else np.nan, *counts)
    return pd.DataFrame.from_dict(out, orient="index", columns=["g_n", "g_mu", "g_var"] + count_cols())


def add_group_stats(rep):
    """Adds g_n (scored members), g_mu, g_var (population variance, raw scores) and the class counts g_c*."""
    slow = cfg.AUTHORS_ONCE or cfg.M_MAX is not None
    stats = (_group_slow if slow else _group_fast)(rep)
    rep = rep.drop(columns=[c for c in stats.columns if c in rep.columns]).join(stats)
    for c in ["g_n"] + count_cols():
        rep[c] = rep[c].fillna(0)
    return rep


# =======================================================================================
# 4b. Composition cells: the registry of possible compositions and the assignment of every group to one
# =======================================================================================
def _compositions(total, m):
    """All ways to write `total` as an ordered sum of m non-negative integers."""
    if m == 1:
        yield (total,)
        return
    for a in range(total + 1):
        for rest in _compositions(total - a, m - 1):
            yield (a,) + rest


def _size_bins():
    """Group-size classes used by the 'shares' mode: [(lo, hi), ...], hi = None for the last open class."""
    edges = [cfg.MIN_MEMBERS] + list(cfg.COMP_SIZE_EDGES)
    return [(lo, hi - 1) for lo, hi in zip(edges, edges[1:])] + [(edges[-1], None)]


def composition_cells():
    """Registry of every possible cell c. Columns: c, bin, n_lo, n_hi, parts, label.
    'counts': one cell per composition of n members for n = MIN_MEMBERS .. COMP_MAX_SIZE (bin = n - MIN_MEMBERS).
    'shares': one cell per way to split COMP_GRID units among the m classes, times each group-size class."""
    m, rows = cfg.N_CLASSES, []
    if cfg.COMP_MODE == "counts":
        for n in range(cfg.MIN_MEMBERS, cfg.COMP_MAX_SIZE + 1):
            for comp in _compositions(n, m):
                rows.append((n - cfg.MIN_MEMBERS, n, n, comp, f"n={n}: " + "-".join(map(str, comp))))
    else:
        for b, (lo, hi) in enumerate(_size_bins()):
            tag = f"n={lo}" if hi == lo else (f"n={lo}+" if hi is None else f"n={lo}-{hi}")
            for comp in _compositions(cfg.COMP_GRID, m):
                rows.append((b, lo, hi, comp, f"{tag}: " + "-".join(map(str, comp)) + f" (/{cfg.COMP_GRID})"))
    reg = pd.DataFrame(rows, columns=["bin", "n_lo", "n_hi", "parts", "label"])
    reg.index.name = "c"
    return reg.reset_index()


def assign_cells(counts, seed=None):
    """Class counts of groups (array N x m) -> cell index c (float, NaN if the group is outside the registry).
    'counts': c identifies the exact counts (groups larger than COMP_MAX_SIZE or smaller than MIN_MEMBERS: NaN).
    'shares': the shares are rounded to multiples of 1/COMP_GRID by largest remainder; exact ties are broken at
              random (seeded), and the cell also records the group-size class."""
    counts = np.asarray(counts, dtype=np.int64)
    m, n = counts.shape[1], counts.sum(axis=1)
    reg = composition_cells()
    base = max(cfg.COMP_MAX_SIZE, cfg.COMP_GRID) + 1
    nb = int(reg["bin"].max()) + 1
    weights = base ** np.arange(m)
    code_of = lambda parts, bins: parts @ weights + bins * base ** m
    lookup = np.full(base ** m * nb, -1, dtype=np.int64)
    lookup[code_of(np.array(reg["parts"].tolist(), dtype=np.int64), reg["bin"].to_numpy())] = reg["c"].to_numpy()
    out = np.full(len(n), np.nan)
    if cfg.COMP_MODE == "counts":
        ok = (n >= cfg.MIN_MEMBERS) & (n <= cfg.COMP_MAX_SIZE)
        parts, bins = counts[ok], n[ok] - cfg.MIN_MEMBERS
    else:
        ok = n >= cfg.MIN_MEMBERS
        c, nn = counts[ok], n[ok][:, None]
        q, rem = (c * cfg.COMP_GRID) // nn, (c * cfg.COMP_GRID) % nn      # exact integer arithmetic
        deficit = cfg.COMP_GRID - q.sum(axis=1)
        rng = np.random.default_rng(cfg.SEED if seed is None else seed)
        rank = np.argsort(np.argsort(-(rem + rng.random(rem.shape)), axis=1), axis=1)
        parts = q + (rank < deficit[:, None])
        edges = np.array(cfg.COMP_SIZE_EDGES, dtype=np.int64)
        bins = np.searchsorted(edges, n[ok], side="right")
    code = lookup[code_of(parts, bins)]
    assert (code >= 0).all(), "a group fell outside the composition registry"
    out[ok] = code
    return out


# =======================================================================================
# 5. Funnel and events
# =======================================================================================
def _step(steps, name, df):
    steps.append((name, len(df)))
    return df


def funnel(steps):
    """Steps -> table with the number kept, the share of the previous step and the share of all comments."""
    f = pd.DataFrame(steps, columns=["step", "n"])
    f["share_of_previous"] = (f["n"] / f["n"].shift(1)).round(3)
    f["share_of_comments"] = (f["n"] / f["n"].iloc[0]).round(3)
    return f


def _common_filters(rep):
    steps = []
    d = _step(steps, "comments", rep)
    d = _step(steps, "reply is scored (not NI)", d[d["x"].notna()])
    d = _step(steps, "author known", d[d["user"].notna()])
    d = _step(steps, "parent present in the data", d[d["parent_found"]])
    lo, hi = pd.Timestamp(cfg.FIRST_DATE), pd.Timestamp(cfg.LAST_DATE) + pd.Timedelta(days=1)
    d = _step(steps, f"reply within {cfg.FIRST_DATE} .. {cfg.LAST_DATE}", d[(d["t_reply"] >= lo) & (d["t_reply"] < hi)])
    d = _step(steps, "replier has an earlier scored item (own opinion s)", d[d[cfg.COL_S].notna()])
    if cfg.MAX_GAP_DAYS is not None:
        d = _step(steps, f"own-opinion item at most {cfg.MAX_GAP_DAYS} days old", d[d["gap"] <= cfg.MAX_GAP_DAYS])
    return d, steps


def _format(d, **cols):
    """Final events table: EVENT_COLS in order, integer columns with NA allowed."""
    ev = pd.DataFrame({"thread": d["thread"], "subreddit": d["subreddit"], "user": d["user"],
                       "t_reply": d["t_reply"], "gap": d["gap"], "parent_kind": d["parent_kind"],
                       cfg.COL_S: d[cfg.COL_S], cfg.COL_K: d[cfg.COL_K]})
    for name, value in cols.items():
        ev[name] = value
    for c in (cfg.COL_S, cfg.COL_L, cfg.COL_G, cfg.COL_C, cfg.COL_K, "n_members"):
        ev[c] = ev[c].astype("Int64") if c in ev else pd.array([pd.NA] * len(ev), dtype="Int64")
    return ev.reindex(columns=list(cfg.EVENT_COLS))


def dyadic_events(d, steps):
    """Reply to a scored parent written by someone else: s, l (parent's class), k."""
    d = _step(steps, "parent is scored", d[d["x_parent"].notna()])
    d = _step(steps, "not a self-reply  ->  dyadic events", d[d["parent_user"] != d["user"]])
    return _format(d, **{cfg.COL_L: to_class(d["x_parent"], cfg.CUTS), "n_members": 1,
                         "mu": d["x_parent"], "var": 0.0})


def _g_category(d):
    """g = 2 * mean_level + var_level from the raw mean and variance."""
    return 2 * to_class(d["g_mu"], cfg.MEAN_CUTS) + (d["g_var"] >= cfg.VAR_CUT).astype(float)


def group_events(d, steps):
    """Reply with >= MIN_MEMBERS scored members before it: s, g, k with g = 2 * mean_level + var_level."""
    d = _step(steps, f"group has >= {cfg.MIN_MEMBERS} scored members  ->  group events", d[d["g_n"] >= cfg.MIN_MEMBERS])
    return _format(d, **{cfg.COL_G: _g_category(d), "n_members": d["g_n"], "mu": d["g_mu"], "var": d["g_var"]})


def composition_events(d, steps):
    """Same replies as group events (optionally capped in size) with the composition cell c next to g."""
    d = _step(steps, f"group has >= {cfg.MIN_MEMBERS} scored members", d[d["g_n"] >= cfg.MIN_MEMBERS])
    if cfg.COMP_MODE == "counts":
        d = _step(steps, f"group has at most {cfg.COMP_MAX_SIZE} members  ->  composition events",
                  d[d["g_n"] <= cfg.COMP_MAX_SIZE])
    else:
        steps.append(("composition events (shares: no size cap)", len(d)))
    cell = assign_cells(d[count_cols()].to_numpy())
    return _format(d, **{cfg.COL_C: cell, cfg.COL_G: _g_category(d), "n_members": d["g_n"],
                         "mu": d["g_mu"], "var": d["g_var"]})


def prepare(threads=None, submissions=None):
    """Raw tables -> (items, replies with s, k and group statistics, common-filtered replies, funnel steps)."""
    if threads is None or submissions is None:
        threads, submissions = load_raw()
    items = build_items(threads, submissions)
    rep = add_group_stats(add_own_opinion(build_replies(items), items))
    common, steps = _common_filters(rep)
    return items, rep, common, steps


def build_events(threads=None, submissions=None, verbose=True):
    """Whole pipeline. Returns dict with dyadic, group, comp, funnel_dyadic, funnel_group, funnel_comp, replies, items."""
    items, rep, common, steps = prepare(threads, submissions)
    steps_d, steps_g, steps_c = list(steps), list(steps), list(steps)
    res = {"items": items, "replies": rep,
           "dyadic": dyadic_events(common, steps_d), "group": group_events(common, steps_g),
           "comp": composition_events(common, steps_c)}
    res.update(funnel_dyadic=funnel(steps_d), funnel_group=funnel(steps_g), funnel_comp=funnel(steps_c))
    if verbose:
        for name in ("funnel_dyadic", "funnel_group", "funnel_comp"):
            print(f"\n=== {name} ===\n{res[name].to_string(index=False)}")
    return res


def save_events(res, folder=None):
    """Write the three events tables, the funnels, the cell registry and the parameters of this run."""
    folder = cfg.DATA_CACHE if folder is None else folder
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("dyadic", "group", "comp"):
        res[name].to_parquet(folder / f"events_{name}.parquet")
        res["funnel_" + name].to_csv(folder / f"funnel_{name}.csv", index=False)
    composition_cells().assign(parts=lambda r: r["parts"].astype(str)).to_csv(folder / "composition_cells.csv", index=False)
    cfg.save(folder / "config_used.json")


# =======================================================================================
# 6. Cell-fill diagnostics: how many events fall in each (s, influencer) row and (s, influencer, k) cell
# =======================================================================================
def influencer_info(col):
    """(number of categories, labels) of an influencer column."""
    if col == cfg.COL_L:
        return cfg.N_CLASSES, list(cfg.CLASS_LABELS)
    if col == cfg.COL_G:
        return cfg.N_G, list(cfg.G_LABELS)
    if col == cfg.COL_C:
        reg = composition_cells()
        return len(reg), reg["label"].tolist()
    raise ValueError(f"influencer column must be one of {cfg.COL_L!r}, {cfg.COL_G!r}, {cfg.COL_C!r}; got {col!r}")


def _fill_counts(ev, col):
    m, L = cfg.N_CLASSES, influencer_info(col)[0]
    d = ev.dropna(subset=[cfg.COL_S, col, cfg.COL_K])
    N = np.zeros((m, L, m), dtype=np.int64)
    np.add.at(N, (d[cfg.COL_S].to_numpy(int), d[col].to_numpy(int), d[cfg.COL_K].to_numpy(int)), 1)
    return N


def fill_table(ev, col=None):
    """One line per (s, influencer cell): number of events R, events per output class, and a low-fill flag.
    Empty cells are included (they are the point of the table)."""
    col = col or cfg.COL_C
    N = _fill_counts(ev, col)
    m, L = cfg.N_CLASSES, N.shape[1]
    labels = influencer_info(col)[1]
    t = pd.DataFrame({"s": np.repeat(np.arange(m), L), "cell": np.tile(np.arange(L), m)})
    t["s_label"] = [cfg.CLASS_LABELS[i] for i in t["s"]]
    t["label"] = [labels[i] for i in t["cell"]]
    t["R"] = N.sum(axis=2).reshape(-1)
    for k in range(m):
        t[f"n_k{k}"] = N[:, :, k].reshape(-1)
    t["low"] = t["R"] < cfg.MIN_ROW_EVENTS
    return t


def fill_summary(ev, col=None, thresholds=(10, 30, 100)):
    """One-row summary of how well an influencer summary is filled.
    rows = (s, influencer) pairs; 'events_in_rows_below_t' is the share of events that sit in rows with fewer than t events;
    prior_weight = event-weighted share of the estimate coming from the shrinkage prior TAU / (R + TAU);
    cmi_bias_bits = plug-in bias of I(K; influencer | S), roughly (K-1) * sum_s (L_s - 1) / (2 N ln 2)."""
    col = col or cfg.COL_C
    N = _fill_counts(ev, col)
    m, L = cfg.N_CLASSES, N.shape[1]
    R, tot = N.sum(axis=2), int(N.sum())
    out = {"influencer": col, "categories": L, "rows": m * L, "rows_nonempty": int((R > 0).sum()), "events": tot,
           "median_R": float(np.median(R[R > 0])) if tot else np.nan, "min_R_nonempty": int(R[R > 0].min()) if tot else 0,
           "cells_sk_zero": float((N == 0).mean())}
    for t in sorted(set(thresholds) | {cfg.MIN_ROW_EVENTS}):
        out[f"rows_below_{t}"] = float((R < t).mean())
        out[f"events_in_rows_below_{t}"] = float(R[R < t].sum() / tot) if tot else np.nan
    out["prior_weight"] = float((R * cfg.TAU / (R + cfg.TAU + 1e-12)).sum() / tot) if tot else np.nan
    Ls = (R > 0).sum(axis=1)
    out["cmi_bias_bits"] = float((m - 1) * np.maximum(Ls - 1, 0).sum() / (2 * tot * np.log(2))) if tot else np.nan
    return pd.DataFrame([out])


def compare_fill(res, thresholds=(10, 30, 100)):
    """Fill of the three summaries side by side: parent class (l), mean x variance (g), composition (c).
    'g (comp events)' is g on the same replies as c, the fair comparison with c."""
    parts = [("l  parent class", res["dyadic"], cfg.COL_L), ("g  mean x variance", res["group"], cfg.COL_G),
             ("g  (comp events)", res["comp"], cfg.COL_G), ("c  composition", res["comp"], cfg.COL_C)]
    out = pd.concat([fill_summary(ev, col, thresholds).assign(summary=name) for name, ev, col in parts], ignore_index=True)
    return out.set_index("summary")


def fill_by_size(ev):
    """Composition events by group-size class: possible cells, cells that were observed, events per possible cell,
    and the share of (s, cell) rows of that size class with fewer than MIN_ROW_EVENTS events."""
    reg = composition_cells()
    N = _fill_counts(ev, cfg.COL_C)
    R = N.sum(axis=2)                                              # (m, L)
    rows = []
    for b, g in reg.groupby("bin"):
        idx = g["c"].to_numpy()
        events = int(R[:, idx].sum())
        rows.append({"bin": b, "sizes": f"{g['n_lo'].iloc[0]}" + ("" if g["n_hi"].iloc[0] == g["n_lo"].iloc[0] else
                                                                  f"-{g['n_hi'].iloc[0]}" if g["n_hi"].iloc[0] else "+"),
                     "cells": len(idx), "cells_observed": int((R[:, idx].sum(axis=0) > 0).sum()), "events": events,
                     "events_per_cell": events / len(idx) / cfg.N_CLASSES,
                     "rows_below_min": float((R[:, idx] < cfg.MIN_ROW_EVENTS).mean())})
    t = pd.DataFrame(rows)
    t["share_of_events"] = t["events"] / t["events"].sum()
    return t


def scan_compositions(threads=None, submissions=None, ms=(2, 3), max_sizes=(3, 4, 5, 6), grids=(3, 4, 6), verbose=False):
    """Fill summary of the composition summary for several m, size caps ('counts') and grids ('shares').
    Rebuilds the group counts once per m and restores the configuration afterwards."""
    old = (cfg.N_CLASSES, cfg.COMP_MODE, cfg.COMP_MAX_SIZE, cfg.COMP_GRID)
    rows = []
    try:
        for m in ms:
            cfg.set_classes(m)
            _, _, common, steps = prepare(threads, submissions)
            options = [("counts", s, cfg.COMP_GRID) for s in max_sizes] + [("shares", cfg.COMP_MAX_SIZE, g) for g in grids]
            for mode, size, grid in options:
                cfg.COMP_MODE, cfg.COMP_MAX_SIZE, cfg.COMP_GRID = mode, size, grid
                if size < cfg.MIN_MEMBERS:
                    continue
                ev = composition_events(common, list(steps))
                row = fill_summary(ev, cfg.COL_C).iloc[0].to_dict()
                row.update(m=m, mode=mode, option=(f"n<={size}" if mode == "counts" else f"grid 1/{grid}"))
                rows.append(row)
                if verbose:
                    print(m, mode, size, grid, row["categories"], row["events"])
    finally:
        cfg.set_classes(old[0])
        cfg.COMP_MODE, cfg.COMP_MAX_SIZE, cfg.COMP_GRID = old[1:]
    cols = ["m", "mode", "option", "categories", "rows", "events", "median_R", "min_R_nonempty",
            f"rows_below_{cfg.MIN_ROW_EVENTS}", f"events_in_rows_below_{cfg.MIN_ROW_EVENTS}", "prior_weight", "cmi_bias_bits"]
    return pd.DataFrame(rows)[cols]


# =======================================================================================
# Self-test: vectorised group statistics and own opinion against brute force on synthetic data
# =======================================================================================
def _synthetic(seed=1, n_threads=12, n_comments=40, n_users=7):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2021-03-01")
    thr, sub = [], []
    for i in range(n_threads):
        sr, sid = f"r/{i % 2}", f"s{i}"
        t0 = int(rng.integers(0, 20))
        sub.append({cfg.RAW_SUBMISSION_ID: sid, cfg.RAW_SUBREDDIT_ID: sr, cfg.RAW_AUTHOR_ID: f"u{rng.integers(n_users)}",
                    cfg.RAW_TIME: base + pd.Timedelta(seconds=t0),
                    cfg.RAW_SCORE: rng.random() if rng.random() > .2 else "NI"})
        made = []                                                  # (comm_id, time) of this thread
        for j in range(n_comments):
            if made and rng.random() < .35:
                pcid, pt = made[rng.integers(len(made))]
                pid = cfg.PARENT_PREFIX_COMMENT + pcid
            else:
                pt, pid = t0, cfg.PARENT_PREFIX_SUBMISSION + sid
            t = max(pt, t0) + int(rng.integers(0, 6))              # few distinct seconds -> many ties
            cid = f"c{i}_{j}"
            made.append((cid, t))
            thr.append({cfg.RAW_COMMENT_ID: cid, cfg.RAW_SUBMISSION_ID: cfg.PARENT_PREFIX_SUBMISSION + sid,
                        cfg.RAW_PARENT_ID: pid, cfg.RAW_SUBREDDIT_ID: sr, cfg.RAW_AUTHOR_ID: f"u{rng.integers(n_users)}",
                        cfg.RAW_TIME: base + pd.Timedelta(seconds=t),
                        cfg.RAW_SCORE: rng.random() if rng.random() > .25 else "NI"})
    return pd.DataFrame(thr), pd.DataFrame(sub)


def _brute_class(x):
    """Independent class of a raw score (written without to_class)."""
    c = cfg.CUTS
    if x < c[0]:
        return 0
    return 1 if (len(c) == 1 or x <= c[1]) else 2


def _selftest_one(verbose=True):
    threads, submissions = _synthetic()
    items = build_items(threads, submissions)
    rep = add_own_opinion(build_replies(items), items)

    fast, slow = _group_fast(rep), _group_slow(rep)
    assert fast.index.equals(slow.index.sort_values()) or set(fast.index) == set(slow.index)
    f, s = fast.loc[slow.index], slow
    assert (f["g_n"] == s["g_n"]).all(), "member counts differ between fast and slow"
    assert np.allclose(f["g_mu"], s["g_mu"], equal_nan=True), "means differ"
    assert np.allclose(f["g_var"], s["g_var"], equal_nan=True, atol=1e-9), "variances differ"
    assert (f[count_cols()].to_numpy() == s[count_cols()].to_numpy()).all(), "class counts differ between fast and slow"

    # independent brute force straight from the raw tables (no code shared with the pipeline)
    raw_c = threads.assign(t=pd.to_datetime(threads[cfg.RAW_TIME]), x=pd.to_numeric(threads[cfg.RAW_SCORE], errors="coerce"))
    raw_s = submissions.assign(t=pd.to_datetime(submissions[cfg.RAW_TIME]), x=pd.to_numeric(submissions[cfg.RAW_SCORE], errors="coerce"))
    n_checked = n_with_members = 0
    for _, r in rep[rep["x"].notna() & rep["user"].notna()].iterrows():
        c = raw_c[(raw_c[cfg.RAW_COMMENT_ID] == r["item_id"]) & (raw_c[cfg.RAW_SUBREDDIT_ID] == r["subreddit"])].iloc[0]
        xs = []
        sib = raw_c[(raw_c[cfg.RAW_SUBREDDIT_ID] == r["subreddit"]) & (raw_c[cfg.RAW_PARENT_ID] == c[cfg.RAW_PARENT_ID])
                    & (raw_c["t"] < c["t"]) & (raw_c[cfg.RAW_AUTHOR_ID] != c[cfg.RAW_AUTHOR_ID]) & raw_c["x"].notna()]
        xs += list(sib["x"])
        pid = c[cfg.RAW_PARENT_ID]
        if pid.startswith(cfg.PARENT_PREFIX_SUBMISSION):
            pr = raw_s[(raw_s[cfg.RAW_SUBMISSION_ID] == pid[3:]) & (raw_s[cfg.RAW_SUBREDDIT_ID] == r["subreddit"])]
        else:
            pr = raw_c[(raw_c[cfg.RAW_COMMENT_ID] == pid[3:]) & (raw_c[cfg.RAW_SUBREDDIT_ID] == r["subreddit"])]
        if len(pr) and pd.notna(pr["x"].iloc[0]) and pr[cfg.RAW_AUTHOR_ID].iloc[0] != c[cfg.RAW_AUTHOR_ID]:
            xs.append(pr["x"].iloc[0])
        got = fast.loc[r.name] if r.name in fast.index else None
        assert got is not None
        assert int(got["g_n"]) == len(xs), f"{r['item_id']}: n {got['g_n']} vs brute {len(xs)}"
        if xs:
            n_with_members += 1
            assert np.isclose(got["g_mu"], np.mean(xs)) and np.isclose(got["g_var"], np.var(xs))
        brute = [sum(_brute_class(v) == j for v in xs) for j in range(cfg.N_CLASSES)]
        assert [int(got[c]) for c in count_cols()] == brute, f"{r['item_id']}: class counts {got[count_cols()].tolist()} vs {brute}"
        # own opinion: the time of the last scored item of the user strictly before the parent
        if len(pr):
            tp = pr["t"].iloc[0]
            mine = pd.concat([raw_c.loc[raw_c[cfg.RAW_AUTHOR_ID] == c[cfg.RAW_AUTHOR_ID], ["t", "x"]],
                              raw_s.loc[raw_s[cfg.RAW_AUTHOR_ID] == c[cfg.RAW_AUTHOR_ID], ["t", "x"]]])
            mine = mine[mine["x"].notna() & (mine["t"] < tp)]
            if len(mine):
                assert rep.loc[r.name, "t_prev"] == mine["t"].max(), f"{r['item_id']}: wrong own-opinion item"
            else:
                assert pd.isna(rep.loc[r.name, "t_prev"]), f"{r['item_id']}: own opinion should be missing"
        n_checked += 1

    # the full pipeline runs, and the options that force the slow path run too
    res = build_events(threads, submissions, verbose=False)
    for name in ("dyadic", "group", "comp"):
        assert set(cfg.EVENT_COLS) == set(res[name].columns)
    assert res["group"][cfg.COL_G].dropna().between(0, cfg.N_G - 1).all()
    assert res["dyadic"][cfg.COL_L].dropna().between(0, cfg.N_CLASSES - 1).all()
    old = (cfg.AUTHORS_ONCE, cfg.M_MAX)
    try:
        cfg.AUTHORS_ONCE, cfg.M_MAX = True, 3
        build_events(threads, submissions, verbose=False)
    finally:
        cfg.AUTHORS_ONCE, cfg.M_MAX = old

    # composition cells, both modes: the registry decodes to the brute-force counts of every event
    rep = res["replies"]
    saved = (cfg.COMP_MODE, cfg.COMP_MAX_SIZE, cfg.COMP_GRID, cfg.COMP_SIZE_EDGES)
    try:
        for mode, size, grid, edges in (("counts", 5, 4, ()), ("counts", 3, 4, ()), ("shares", 5, 4, ()),
                                        ("shares", 5, 6, (3, 5))):
            cfg.COMP_MODE, cfg.COMP_MAX_SIZE, cfg.COMP_GRID, cfg.COMP_SIZE_EDGES = mode, size, grid, edges
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cfg.validate()
            reg = composition_cells()
            if mode == "counts":
                from math import comb
                want = sum(comb(n + cfg.N_CLASSES - 1, cfg.N_CLASSES - 1) for n in range(cfg.MIN_MEMBERS, size + 1))
                assert len(reg) == want, f"registry has {len(reg)} cells, expected {want}"
            assert reg["label"].is_unique and (reg["c"] == np.arange(len(reg))).all()
            ev = build_events(threads, submissions, verbose=False)["comp"]
            assert ev[cfg.COL_C].notna().all() and len(ev) > 0
            cnt = rep.loc[ev.index, count_cols()].to_numpy().astype(int)
            parts = np.array(reg.loc[ev[cfg.COL_C].astype(int), "parts"].tolist())
            if mode == "counts":
                assert (parts == cnt).all(), "counts-mode cell does not decode to the group's class counts"
                assert (ev["n_members"].to_numpy() <= size).all()
            else:
                assert (parts.sum(axis=1) == grid).all(), "shares must add up to the grid"
                assert np.abs(parts / grid - cnt / cnt.sum(axis=1, keepdims=True)).max() <= 1 / grid + 1e-9
                lo = reg.loc[ev[cfg.COL_C].astype(int), "n_lo"].to_numpy()
                hi = reg.loc[ev[cfg.COL_C].astype(int), "n_hi"].fillna(10 ** 9).to_numpy()
                assert ((ev["n_members"].to_numpy() >= lo) & (ev["n_members"].to_numpy() <= hi)).all(), "wrong size class"
            # diagnostics run and are consistent with the events
            t = fill_table(ev, cfg.COL_C)
            assert t["R"].sum() == len(ev) and len(t) == cfg.N_CLASSES * len(reg)
            assert fill_by_size(ev)["events"].sum() == len(ev)
            fs = fill_summary(ev, cfg.COL_C)
            assert int(fs["events"].iloc[0]) == len(ev)
        compare_fill(build_events(threads, submissions, verbose=False))
    finally:
        cfg.COMP_MODE, cfg.COMP_MAX_SIZE, cfg.COMP_GRID, cfg.COMP_SIZE_EDGES = saved
    if verbose:
        print(f"selftest m={cfg.N_CLASSES} OK: {n_checked} replies checked against brute force "
              f"({n_with_members} with at least one group member); "
              f"{len(res['dyadic'])} dyadic, {len(res['group'])} group and {len(res['comp'])} composition events")
    return True


def selftest(verbose=True):
    """Run the whole self-test for m = 2 and m = 3 and restore the configuration."""
    old = cfg.N_CLASSES
    try:
        for m in (3, 2):
            cfg.set_classes(m)
            _selftest_one(verbose)
        # a size grid with enough synthetic data to see all rounding branches
        counts = np.array([[1, 1, 1], [2, 1, 0], [3, 0, 0], [1, 1, 0], [2, 2, 2]])
        cfg.set_classes(3)
        saved = (cfg.COMP_MODE, cfg.COMP_GRID, cfg.COMP_SIZE_EDGES)
        try:
            cfg.COMP_MODE, cfg.COMP_GRID, cfg.COMP_SIZE_EDGES = "shares", 4, ()
            a, b = assign_cells(counts, seed=1), assign_cells(counts, seed=1)
            assert (a == b).all(), "seeded rounding must be reproducible"
            reg = composition_cells()
            got = [reg.loc[int(c), "parts"] for c in a]
            assert got[1] == (3, 1, 0) and got[2] == (4, 0, 0) and sum(got[0]) == 4 and sum(got[3]) == 4
        finally:
            cfg.COMP_MODE, cfg.COMP_GRID, cfg.COMP_SIZE_EDGES = saved
        if verbose:
            print("selftest: shares rounding OK")
    finally:
        cfg.set_classes(old)
    return True


if __name__ == "__main__":
    selftest()