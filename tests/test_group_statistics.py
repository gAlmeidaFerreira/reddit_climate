"""Group statistics: vectorised vs explicit, both against brute force from the raw tables."""
import numpy as np
import pandas as pd

import config as cfg
from categories import class_count_columns
from pipeline.group_statistics import (add_group_statistics, group_statistics_explicit, group_statistics_vectorised,
                                       pair_disagreement_shares)
from synthetic_data import brute_force_class, brute_force_pair_shares


def test_vectorised_and_explicit_implementations_agree(replies_before_groups):
    vectorised = group_statistics_vectorised(replies_before_groups)
    explicit = group_statistics_explicit(replies_before_groups)
    assert set(vectorised.index) == set(explicit.index)
    vectorised = vectorised.loc[explicit.index]
    assert (vectorised["group_size"] == explicit["group_size"]).all()
    assert np.allclose(vectorised["group_mean"], explicit["group_mean"], equal_nan=True)
    assert np.allclose(vectorised["group_variance"], explicit["group_variance"], equal_nan=True, atol=1e-9)
    assert (vectorised[class_count_columns()].to_numpy() == explicit[class_count_columns()].to_numpy()).all()


def test_group_members_match_brute_force(replies_before_groups, brute_force):
    statistics = group_statistics_vectorised(replies_before_groups)
    replies = replies_before_groups.loc[statistics.index]
    replies_with_members = 0
    for reply_row, reply in replies.iterrows():
        member_scores = brute_force[(reply["subreddit"], reply["item_id"])]["member_scores"]
        got = statistics.loc[reply_row]
        assert int(got["group_size"]) == len(member_scores), reply["item_id"]
        if member_scores:
            replies_with_members += 1
            assert np.isclose(got["group_mean"], np.mean(member_scores))
            assert np.isclose(got["group_variance"], np.var(member_scores))
        expected_counts = [sum(brute_force_class(score) == class_index for score in member_scores)
                           for class_index in range(cfg.N_CLASSES)]
        assert [int(got[column]) for column in class_count_columns()] == expected_counts, reply["item_id"]
    assert replies_with_members > 100


def test_pair_shares_match_double_loop(replies_before_groups, brute_force):
    replies = add_group_statistics(replies_before_groups)
    scored = replies[replies["reply_score"].notna() & replies["author"].notna()]
    for _, reply in scored.iterrows():
        member_classes = [brute_force_class(score)
                          for score in brute_force[(reply["subreddit"], reply["item_id"])]["member_scores"]]
        opposed_share, mild_share = brute_force_pair_shares(member_classes)
        assert np.isclose(reply["group_opposed_pair_share"], opposed_share, equal_nan=True)
        assert np.isclose(reply["group_mild_pair_share"], mild_share, equal_nan=True)


def test_pair_shares_are_unbiased_in_group_size():
    """For class shares p, two random members are opposed with probability 2 p0 p2 and mildly different with
    probability 2 p1 (p0 + p2), whatever the group size."""
    random_generator = np.random.default_rng(5)
    class_shares = np.array([0.5, 0.27, 0.23])
    for group_size in (2, 3, 5, 12):
        class_counts = random_generator.multinomial(group_size, class_shares, size=200_000)
        opposed_share, mild_share = pair_disagreement_shares(class_counts)
        assert abs(opposed_share.mean() - 2 * class_shares[0] * class_shares[2]) < 0.004, group_size
        assert abs(mild_share.mean() - 2 * class_shares[1] * (class_shares[0] + class_shares[2])) < 0.004, group_size


def test_sample_variance_option(replies_before_groups):
    for unbiased in (True, False):
        cfg.VAR_UNBIASED = unbiased
        replies = add_group_statistics(replies_before_groups)
        two_or_more = replies["group_size"] >= 2
        size = replies.loc[two_or_more, "group_size"]
        factor = size / (size - 1) if unbiased else 1.0
        assert np.allclose(replies.loc[two_or_more, "group_variance_population"] * factor,
                           replies.loc[two_or_more, "group_variance"], atol=1e-12)
        if unbiased:
            assert replies.loc[~two_or_more, "group_variance"].isna().all()
            pairs = replies[replies["group_size"] == 2]["group_variance"]
            assert pairs.between(0, 0.5 + 1e-12).all()                 # (a - b)^2 / 2 with scores in [0, 1]


def test_authors_once_counts_each_author_once(replies_before_groups):
    cfg.AUTHORS_ONCE = True
    statistics = group_statistics_explicit(replies_before_groups)
    cfg.AUTHORS_ONCE = False
    every_member = group_statistics_explicit(replies_before_groups).loc[statistics.index]
    assert (statistics["group_size"] <= every_member["group_size"]).all()
    assert (statistics["group_size"] < every_member["group_size"]).any()


def test_most_recent_members_cap(replies_before_groups):
    cfg.M_MAX = 3
    assert (group_statistics_explicit(replies_before_groups)["group_size"] <= 3).all()


def test_rows_outside_the_candidates_get_empty_groups(replies_before_groups):
    replies = add_group_statistics(replies_before_groups)
    not_candidates = replies["reply_score"].isna()
    assert (replies.loc[not_candidates, "group_size"] == 0).all()
    assert pd.api.types.is_float_dtype(replies["group_size"])
