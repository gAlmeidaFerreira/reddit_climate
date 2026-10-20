"""
Cuts that are chosen from the data, before any transition is looked at.

The opposition cut turns the share of opposed pairs O into a two-level label, so that h = 2 * lean + opposition
level has the same categories as g = 2 * lean + variance level. By default it is matched to VAR_CUT: it labels
(as closely as the discrete values of O allow) the same share of groups "high" as VAR_CUT does for the variance,
so g and h split the events in the same proportion and differ only in WHICH groups they call divided.
The new opinion class is never used, so the cut cannot be tuned toward a result.
"""
import numpy as np

import config as cfg


def matched_cut(values, target_share):
    """The cut such that the share of values >= cut is as close as possible to target_share (ties between values
    can make an exact match impossible). Candidate cuts are the distinct POSITIVE values (a cut at 0 would call
    every group 'high'). Missing values are ignored. Returns infinity if no value is positive."""
    sorted_values = np.sort(np.asarray(values, dtype=float))
    sorted_values = sorted_values[~np.isnan(sorted_values)]
    candidate_cuts = np.unique(sorted_values[sorted_values > 0])
    if len(candidate_cuts) == 0:
        return float(np.inf)
    share_at_or_above = 1.0 - np.searchsorted(sorted_values, candidate_cuts, side="left") / len(sorted_values)
    return float(candidate_cuts[np.argmin(np.abs(share_at_or_above - target_share))])


def opposition_cut(replies):
    """Cut of the opposed-pair share O: cfg.OPP_CUT when set; otherwise matched to the share of groups (with at
    least MIN_MEMBERS members) whose variance is >= VAR_CUT. Falls back to 1.0 if no group has an opposed pair."""
    if cfg.OPP_CUT is not None:
        return float(cfg.OPP_CUT)
    groups = replies[replies["group_size"] >= cfg.MIN_MEMBERS]
    high_variance_share = float((groups["group_variance"] >= cfg.VAR_CUT).mean())
    cut = matched_cut(groups["group_opposed_pair_share"], high_variance_share)
    return 1.0 if np.isinf(cut) else cut
