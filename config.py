"""
Parameters of the transition-tensor analysis (Stage 0 of the algorithm).

Notation used everywhere (text and code), as in Kozitsin:
    s  own opinion class of the replier        (0 against, 1 neutral, 2 favour)
    l  influencer opinion, dyadic case          (class of the parent's score)
    g  influencer opinion, group case           (one of 6 categories, see G_LABELS)
    k  output opinion class                     (class of the replier's reply)
    x  raw score in [0, 1] given by GPT-3.5     (NI = missing)

Rules for this file:
  * every threshold or option lives here and nowhere else (no numbers typed inside functions)
  * cut-points are chosen from histograms of the scores BEFORE looking at any transition
  * validate() runs at import time; to_dict()/save() store the parameters next to each output
"""
import json
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------------------
# Paths (the zip is expected to be extracted into data/raw/)
# ---------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw" / "filtered_subs_and_threads_anonymized"
THREADS_DIR = DATA_RAW / "threads"            # threads/subredditN_.../<submission>_scored.csv
SUBMISSIONS_DIR = DATA_RAW / "submissions"    # submissions/subredditN_.../subredditN_subs_anon.csv
DATA_CACHE = ROOT / "data" / "cache"          # optional saved tables
OUTPUTS = ROOT / "outputs"

# ---------------------------------------------------------------------------------------
# Raw column names (as in the released CSVs)
# ---------------------------------------------------------------------------------------
RAW_COMMENT_ID = "comm_id"
RAW_SUBMISSION_ID = "sub_id"
RAW_PARENT_ID = "parent_id"            # "t1_<comm_id>" (reply to a comment) or "t3_<sub_id>" (to the submission)
RAW_SUBREDDIT_ID = "subreddit_id"      # "r/0" ... "r/5"
RAW_AUTHOR_ID = "author_id"
RAW_TIME = "created_at"
RAW_SCORE = "predicted_stance_scores"  # 0..1, or "NI" (not related to climate change)
PARENT_PREFIX_COMMENT = "t1_"
PARENT_PREFIX_SUBMISSION = "t3_"

# Anonymised subreddits, matched to Table 1 of Antelmi et al. by number of submissions
# (181, 148, 2429, 385, 490, 2618). Verify against your data before relying on the names.
SUBREDDIT_NAMES = {"r/0": "bayarea", "r/1": "California", "r/2": "collapse",
                   "r/3": "news", "r/4": "politics", "r/5": "AskReddit"}

# ---------------------------------------------------------------------------------------
# Classes (s, l and k share the same classes).  N_CLASSES = m = 2 or 3
# ---------------------------------------------------------------------------------------
N_CLASSES = 3

# raw score -> class with increasing cuts c_0 < c_1 < ...:
#   x < c_0 -> 0 ; c_0 <= x <= c_1 -> 1 ; x > c_1 -> 2        (m = 2 uses one cut only)
# m = 2 means "against" versus "not against": a cut at 0.5 would split the spike of scores exactly 0.5.
CUTS_BY_M = {2: (0.35,), 3: (0.35, 0.65)}
CLASS_LABELS_BY_M = {2: ("against", "not-against"), 3: ("against", "neutral", "favour")}
CUTS_CHOSEN_FROM_HISTOGRAM = False     # set True once you have looked at the score histogram

# ---------------------------------------------------------------------------------------
# Group influencer g = 2 * mean_level + var_level   (statistics from RAW scores, binned once)
# ---------------------------------------------------------------------------------------
MEAN_CUTS_BY_M = {2: (0.35,), 3: (0.35, 0.65)}     # group mean of raw scores -> m levels (same scale as the cuts)
MEAN_LABELS_BY_M = {2: ("lean-against", "lean-not-against"), 3: ("lean-against", "centre", "lean-favour")}
VAR_UNBIASED = True                    # True: unbiased sample variance (population variance * n / (n - 1)), the average of
                                       #   (x_i - x_j)^2 / 2 over pairs; its mean does not depend on the group size n.
                                       # False: population variance (ddof = 0), which grows with n. Needs MIN_MEMBERS >= 2.
VAR_CUT = 0.125                        # variance of raw scores: below = consensus, else divided
                                       #   (largest value: 0.25 if population, 0.5 if unbiased: two scores 0 and 1)
VAR_CUT_CHOSEN_FROM_HISTOGRAM = False  # PLACEHOLDER value above: set from the histogram of group variances
VAR_LABELS = ("consensus", "divided")

# ---------------------------------------------------------------------------------------
# Pair statistics of the composition (alternative to the variance as the "spread" coordinate)
#   O = share of pairs of members that are OPPOSED (one in the first class, one in the last class)   unbiased in n
#   M = share of pairs that disagree mildly (different classes, not opposed)
#   h = 2 * mean_level + opp_level, with opp_level = (O >= OPP_CUT)   -> same 2m categories as g
# ---------------------------------------------------------------------------------------
OPP_CUT = None                         # None: matched to VAR_CUT, i.e. the cut of O that labels the same share of events
                                       #   "high" as VAR_CUT does for the variance (a fair comparison); or a number in (0, 1]
OPP_LABELS = ("low-opposition", "high-opposition")
DISAGREE_WEIGHTS = (0.0, 0.25, 0.5, 1.0)   # D(w) = O + w * M : weight of a mild pair relative to an opposed pair.
                                       #   0 = opposed pairs only ; 0.25 = variance of the class positions ;
                                       #   0.5 = mean absolute distance ; 1 = every disagreeing pair counts the same
TEST_SIZE_EDGES = (3, 4, 6)            # group-size classes of the comparison test: 2 | 3 | 4-5 | 6+
COMPARE_FOLDS = 5                      # user folds of the held-out comparison
COMPARE_TAUS = (0, 10, 25, 50, 100, 200, 500)   # tau chosen inside the training folds (nested), per summary
COMPARE_B = 500                        # user bootstrap for the paired differences

# ---------------------------------------------------------------------------------------
# Group influencer c = composition of the group (how many members in each class)
#   "counts": exact class counts, one cell per composition of a group of n members, n = MIN_MEMBERS..COMP_MAX_SIZE
#             (larger groups are dropped; set M_MAX = COMP_MAX_SIZE to keep their most recent members instead)
#   "shares": the shares of the classes rounded to multiples of 1 / COMP_GRID (largest remainder, random tie-break),
#             optionally split by group size with COMP_SIZE_EDGES, e.g. (3, 6, 11) -> sizes 2 | 3-5 | 6-10 | 11+
# ---------------------------------------------------------------------------------------
COMP_MODE = "counts"
COMP_MAX_SIZE = 5
COMP_GRID = 4
COMP_SIZE_EDGES = ()

# Who is in the group of a reply (the parent if scored and not written by the replier, plus
# earlier replies to the same parent by other authors); never the reply itself.
MIN_MEMBERS = 2          # minimum scored members; robustness check: 3
M_MAX = None             # keep only the most recent M_MAX members (None = no cap); check: 10
AUTHORS_ONCE = False     # True: each author counts once, with the latest score before the reply

# ---------------------------------------------------------------------------------------
# Own opinion s of the replier
# ---------------------------------------------------------------------------------------
MAX_GAP_DAYS = None      # drop events whose own-opinion item is older than this (None = no limit); check: 7

# ---------------------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------------------
TAU = 50                 # pseudo-events of shrinkage toward q[s, k]; TAU = 0 gives Kozitsin's raw proportions
TAU_SLICE = 50           # shrinkage of a slice row toward the global row (choose by held-out log-loss)
MIN_ROW_EVENTS = 300     # rows below this are flagged
MAX_PRIOR_WEIGHT = 0.3   # a row whose prior weight TAU / (R + TAU) exceeds this is flagged

# ---------------------------------------------------------------------------------------
# Uncertainty and null tests
# ---------------------------------------------------------------------------------------
B = 500                  # user-level bootstrap replicates
N_PERM = 200             # shuffles of the influencer for the null of I(K; influencer | S)
CI = (2.5, 97.5)         # percentiles reported
SEED = 0

# ---------------------------------------------------------------------------------------
# Slices (Stage 3b)
# ---------------------------------------------------------------------------------------
FIRST_DATE = "2020-07-01"      # events outside [FIRST_DATE, LAST_DATE] are dropped (about 10 in 2023)
LAST_DATE = "2022-12-31"
SLICE_FREQ = "Q"               # calendar quarters
SLICE_BY = ("subreddit", "quarter")   # slices to build; subreddit x quarter only for the largest subreddit
SLICE_PAIR_SUBREDDITS = ("r/2",)

# ---------------------------------------------------------------------------------------
# Events table: the interface between events.py and tensors.py
# ---------------------------------------------------------------------------------------
COL_S, COL_L, COL_G, COL_C, COL_K = "s", "l", "g", "c", "k"
COL_H, COL_LEAN = "h", "lean"          # h: mean level x opposed-pair level ; lean: mean level of the group alone
EVENT_COLS = ("user", "thread", "subreddit", "t_reply", "gap", "parent_kind",
              COL_S, COL_L, COL_G, COL_C, COL_H, COL_LEAN, COL_K, "n_members", "mu", "var", "opp", "mild")


def set_classes(m):
    """Select the number of opinion classes (2 or 3) and recompute every derived constant.
    Call it before building events; tensors.py and events.py read these values at call time."""
    global N_CLASSES, CLASS_LABELS, CUTS, MEAN_CUTS, MEAN_LABELS, G_LABELS, N_G, H_LABELS, N_H
    if m not in CUTS_BY_M:
        raise ValueError(f"N_CLASSES must be one of {sorted(CUTS_BY_M)}, got {m}")
    N_CLASSES = m
    CLASS_LABELS = CLASS_LABELS_BY_M[m]
    CUTS = CUTS_BY_M[m]
    MEAN_CUTS = MEAN_CUTS_BY_M[m]
    MEAN_LABELS = MEAN_LABELS_BY_M[m]
    G_LABELS = tuple(f"{a}-{b}" for a in MEAN_LABELS for b in VAR_LABELS)    # index g = 2 * mean_level + var_level
    N_G = len(G_LABELS)
    H_LABELS = tuple(f"{a}-{b}" for a in MEAN_LABELS for b in OPP_LABELS)           # index h = 2 * mean_level + opp_level
    N_H = len(H_LABELS)


def validate():
    """Fail early on inconsistent parameters; warn about placeholders."""
    for name, cuts in (("CUTS", CUTS), ("MEAN_CUTS", MEAN_CUTS)):
        ok = len(cuts) == N_CLASSES - 1 and all(0 <= a < b <= 1 for a, b in zip(cuts, cuts[1:])) \
            and all(0 <= c <= 1 for c in cuts)
        if not ok:
            raise ValueError(f"{name} must be {N_CLASSES - 1} increasing values in [0, 1], got {cuts}")
    if len(CLASS_LABELS) != N_CLASSES or len(MEAN_LABELS) != N_CLASSES:
        raise ValueError("CLASS_LABELS and MEAN_LABELS need one label per class")
    var_max = 0.5 if VAR_UNBIASED else 0.25
    if not 0 < VAR_CUT < var_max:
        raise ValueError(f"VAR_CUT must lie in (0, {var_max}) for VAR_UNBIASED = {VAR_UNBIASED} "
                         f"(the largest possible variance), got {VAR_CUT}")
    if VAR_UNBIASED and MIN_MEMBERS < 2:
        raise ValueError("VAR_UNBIASED needs MIN_MEMBERS >= 2 (the unbiased variance of one score is undefined)")
    if MIN_MEMBERS < 1 or (M_MAX is not None and M_MAX < MIN_MEMBERS):
        raise ValueError("MIN_MEMBERS must be >= 1 and M_MAX (if set) >= MIN_MEMBERS")
    if TAU < 0 or TAU_SLICE < 0 or B < 1 or N_PERM < 1:
        raise ValueError("TAU, TAU_SLICE must be >= 0 and B, N_PERM >= 1")
    if not 0 < MAX_PRIOR_WEIGHT < 1:
        raise ValueError("MAX_PRIOR_WEIGHT must lie in (0, 1)")
    if MAX_GAP_DAYS is not None and MAX_GAP_DAYS <= 0:
        raise ValueError("MAX_GAP_DAYS must be positive or None")
    if len(G_LABELS) != N_G or len(set(G_LABELS)) != N_G or N_G != 2 * N_CLASSES:
        raise ValueError("G_LABELS must contain 2 * N_CLASSES distinct categories")
    if OPP_CUT is not None and not 0 < OPP_CUT <= 1:
        raise ValueError(f"OPP_CUT must be None or lie in (0, 1], got {OPP_CUT}")
    if not DISAGREE_WEIGHTS or any(w < 0 for w in DISAGREE_WEIGHTS):
        raise ValueError("DISAGREE_WEIGHTS must be a non-empty tuple of non-negative numbers")
    if len(OPP_LABELS) != 2 or N_H != 2 * N_CLASSES:
        raise ValueError("OPP_LABELS needs two labels")
    if any(b <= a for a, b in zip(TEST_SIZE_EDGES, TEST_SIZE_EDGES[1:])) or any(e <= MIN_MEMBERS for e in TEST_SIZE_EDGES):
        raise ValueError("TEST_SIZE_EDGES must be increasing and larger than MIN_MEMBERS")
    if COMPARE_FOLDS < 2 or COMPARE_B < 1 or not COMPARE_TAUS:
        raise ValueError("COMPARE_FOLDS >= 2, COMPARE_B >= 1 and a non-empty COMPARE_TAUS are needed")
    if COMP_MODE not in ("counts", "shares"):
        raise ValueError(f"COMP_MODE must be 'counts' or 'shares', got {COMP_MODE!r}")
    if COMP_MAX_SIZE < max(MIN_MEMBERS, 1) or COMP_GRID < 1:
        raise ValueError("COMP_MAX_SIZE must be >= MIN_MEMBERS and COMP_GRID >= 1")
    edges = tuple(COMP_SIZE_EDGES)
    if any(b <= a for a, b in zip(edges, edges[1:])) or any(e <= MIN_MEMBERS for e in edges):
        raise ValueError("COMP_SIZE_EDGES must be increasing and larger than MIN_MEMBERS")
    if not CUTS_CHOSEN_FROM_HISTOGRAM:
        warnings.warn("CUTS are placeholders: choose them from the score histogram before looking at "
                      "transitions, then set CUTS_CHOSEN_FROM_HISTOGRAM = True.", stacklevel=2)
    if not VAR_CUT_CHOSEN_FROM_HISTOGRAM:
        warnings.warn("VAR_CUT is a placeholder: choose it from the histogram of group variances before "
                      "looking at transitions, then set VAR_CUT_CHOSEN_FROM_HISTOGRAM = True.", stacklevel=2)


def to_dict():
    """All parameters (upper-case names) as a JSON-friendly dict."""
    out = {}
    for name, value in globals().items():
        if name.isupper() and not name.startswith("_"):
            out[name] = str(value) if isinstance(value, Path) else value
    return out


def save(path):
    """Write the parameters used by a run next to its outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(), indent=2, default=list))


set_classes(N_CLASSES)
validate()