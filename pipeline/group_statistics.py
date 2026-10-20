"""
Stage 4. Group statistics of every reply, from raw scores (binning into classes happens later, once).

The group of a reply R by author U to parent P is:
    * the parent P, if it was found, is scored, and was not written by U;
    * every sibling: a scored reply to the same P, written STRICTLY before R, by another author.
R itself and every item of U are never members. Only scored members count, so group_size can be smaller than
the real crowd.

Columns added by add_group_statistics:
    group_size                         number of scored members
    group_mean                         mean of the members' raw scores
    group_variance                     sample variance (n / (n - 1)) if cfg.VAR_UNBIASED, else population variance
    group_variance_population          population variance, always
    group_count_class_0 .. _{m-1}      members in each opinion class
    group_opposed_pair_share           O: share of member pairs in opposite extreme classes
    group_mild_pair_share              M: share of member pairs in different, non-opposite classes

Two implementations give the same numbers: a vectorised one (running sums, the default) and an explicit
one (member lists, slow). The explicit one is also the only path for cfg.AUTHORS_ONCE and cfg.M_MAX.
"""
import numpy as np
import pandas as pd

import config as cfg
from categories import class_count_columns, score_to_class
from pipeline.replies import PARENT_KEY

_STATISTIC_COLUMNS = ["group_size", "group_mean", "group_variance"]


def _candidate_members(replies):
    """Scored replies with a known author and a parent reference. They are at the same time the replies whose
    group is computed and the possible siblings inside the group of the other replies to the same parent."""
    is_candidate = (replies["reply_score"].notna() & replies["author"].notna()
                    & replies["parent_item_id"].notna() & replies["parent_kind"].notna())
    columns = PARENT_KEY + ["author", "reply_time", "reply_score",
                            "parent_found", "parent_author", "parent_time", "parent_score"]
    return replies.loc[is_candidate, columns].copy()


def _parent_is_member(candidates):
    """The parent belongs to the group: found, scored, and written by someone other than the replier."""
    return (candidates["parent_found"] & candidates["parent_score"].notna()
            & (candidates["parent_author"] != candidates["author"]))


def _sum_over_strictly_earlier_rows(candidates, group_column, value_column):
    """For each row: the sum of value_column over the rows of the same group_column whose reply_time is
    STRICTLY earlier. `candidates` must be sorted by (parent group, reply_time).

    A plain 'cumulative sum minus own value' would also count rows written at the same second but listed above.
    Every row of a block of equal times therefore takes the value of the block's first row, which is exactly
    the sum before the block started."""
    sum_of_rows_above = candidates.groupby(group_column)[value_column].cumsum() - candidates[value_column]
    return sum_of_rows_above.groupby([candidates[group_column], candidates["reply_time"]]).transform("first")


def group_statistics_vectorised(replies):
    """Group size, mean, population variance and class counts of every candidate reply, from running sums:
        size = sum of 1, mean = sum of x / size, variance = sum of x^2 / size - mean^2.
    Earlier siblings by other authors = (all strictly earlier replies to the parent)
                                      - (strictly earlier replies to the parent by the same author)."""
    candidates = _candidate_members(replies)
    candidates["parent_group"] = candidates.groupby(PARENT_KEY, sort=False).ngroup()
    candidates = candidates.sort_values(["parent_group", "reply_time"], kind="stable")
    candidates["parent_and_author_group"] = candidates.groupby(["parent_group", "author"], sort=False).ngroup()

    reply_classes = score_to_class(candidates["reply_score"], cfg.CUTS)
    summed_quantities = {"member_count": 1.0,
                         "score_sum": candidates["reply_score"],
                         "squared_score_sum": candidates["reply_score"] ** 2}
    for class_index, count_column in enumerate(class_count_columns()):
        summed_quantities[count_column] = (reply_classes == class_index).astype(float)
    for quantity, values in summed_quantities.items():
        candidates[quantity] = values

    sibling_totals = {
        quantity: (_sum_over_strictly_earlier_rows(candidates, "parent_group", quantity)
                   - _sum_over_strictly_earlier_rows(candidates, "parent_and_author_group", quantity)).to_numpy()
        for quantity in summed_quantities}

    parent_is_member = _parent_is_member(candidates).to_numpy()
    parent_score = candidates["parent_score"].fillna(0.0).to_numpy()
    parent_class = score_to_class(candidates["parent_score"], cfg.CUTS)
    group_size = sibling_totals["member_count"] + parent_is_member
    score_sum = sibling_totals["score_sum"] + parent_is_member * parent_score
    squared_score_sum = sibling_totals["squared_score_sum"] + parent_is_member * parent_score ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        group_mean = np.where(group_size > 0, score_sum / group_size, np.nan)
        group_variance = np.where(group_size > 0,
                                  np.maximum(squared_score_sum / group_size - group_mean ** 2, 0.0),  # no rounding below 0
                                  np.nan)

    statistics = {"group_size": group_size, "group_mean": group_mean, "group_variance": group_variance}
    for class_index, count_column in enumerate(class_count_columns()):
        statistics[count_column] = sibling_totals[count_column] + parent_is_member * (parent_class == class_index)
    return pd.DataFrame(statistics, index=candidates.index)


def _keep_latest_item_per_author(members):
    latest_by_author = {}
    for author, member_time, member_score in members:
        author = "__missing__" if pd.isna(author) else author
        if author not in latest_by_author or member_time > latest_by_author[author][1]:
            latest_by_author[author] = (author, member_time, member_score)
    return list(latest_by_author.values())


def group_statistics_explicit(replies):
    """Same output as group_statistics_vectorised, from an explicit member list per reply (quadratic within a
    parent). Applies cfg.AUTHORS_ONCE (each author counts once, with their latest item) and cfg.M_MAX (keep the
    most recent members only)."""
    candidates = _candidate_members(replies)
    statistics_by_reply = {}
    for _, siblings in candidates.groupby(PARENT_KEY, sort=False):
        reply_times = siblings["reply_time"].to_numpy().astype("datetime64[ns]").astype("int64")
        authors = siblings["author"].to_numpy(object)
        reply_scores = siblings["reply_score"].to_numpy(float)
        parent_is_member = _parent_is_member(siblings).to_numpy()
        for position, reply_row in enumerate(siblings.index):
            members = []                                              # (author, time, score)
            if parent_is_member[position]:
                members.append((siblings["parent_author"].iat[position], siblings["parent_time"].iat[position].value,
                                siblings["parent_score"].iat[position]))
            is_earlier_sibling = (reply_times < reply_times[position]) & (authors != authors[position])
            members += list(zip(authors[is_earlier_sibling], reply_times[is_earlier_sibling],
                                reply_scores[is_earlier_sibling]))
            if cfg.AUTHORS_ONCE:
                members = _keep_latest_item_per_author(members)
            if cfg.M_MAX is not None:
                members = sorted(members, key=lambda member: member[1], reverse=True)[:cfg.M_MAX]

            member_scores = np.array([member[2] for member in members], dtype=float)
            if len(member_scores):
                class_counts = np.bincount(score_to_class(member_scores, cfg.CUTS).astype(int), minlength=cfg.N_CLASSES)
                statistics_by_reply[reply_row] = (len(member_scores), member_scores.mean(), member_scores.var(), *class_counts)
            else:
                statistics_by_reply[reply_row] = (0, np.nan, np.nan, *np.zeros(cfg.N_CLASSES, int))
    return pd.DataFrame.from_dict(statistics_by_reply, orient="index",
                                  columns=_STATISTIC_COLUMNS + class_count_columns())


def pair_disagreement_shares(class_counts):
    """Class counts of groups (groups x classes) -> (O, M), both unbiased in the group size n:
        O = share of member pairs that are opposed (one in the first class, one in the last) = c_0 * c_{m-1} / C(n, 2)
        M = share of member pairs that disagree mildly (different classes, not opposed)
    NaN for groups with fewer than 2 members. With 2 classes every disagreement is opposed, so M = 0."""
    class_counts = np.asarray(class_counts, dtype=float)
    group_size = class_counts.sum(axis=1)
    all_pairs = group_size * (group_size - 1) / 2
    same_class_pairs = (class_counts * (class_counts - 1) / 2).sum(axis=1)
    opposed_pairs = class_counts[:, 0] * class_counts[:, -1]
    with np.errstate(invalid="ignore", divide="ignore"):
        opposed_share = np.where(all_pairs > 0, opposed_pairs / all_pairs, np.nan)
        mild_share = np.where(all_pairs > 0, (all_pairs - same_class_pairs - opposed_pairs) / all_pairs, np.nan)
    return opposed_share, mild_share


def add_group_statistics(replies):
    """Attach the group statistics to every reply (rows outside the candidates get size 0 and zero counts)."""
    needs_explicit_members = cfg.AUTHORS_ONCE or cfg.M_MAX is not None
    statistics = (group_statistics_explicit if needs_explicit_members else group_statistics_vectorised)(replies)
    replies = replies.drop(columns=[column for column in statistics.columns if column in replies.columns]).join(statistics)

    replies["group_variance_population"] = replies["group_variance"]
    if cfg.VAR_UNBIASED:                                  # sample variance: times n / (n - 1), undefined for n < 2
        size_with_two_or_more = replies["group_size"].where(replies["group_size"] >= 2)
        replies["group_variance"] = (replies["group_variance_population"] * size_with_two_or_more
                                     / (size_with_two_or_more - 1))
    for count_column in ["group_size"] + class_count_columns():
        replies[count_column] = replies[count_column].fillna(0)
    replies["group_opposed_pair_share"], replies["group_mild_pair_share"] = pair_disagreement_shares(
        replies[class_count_columns()].to_numpy())
    return replies
