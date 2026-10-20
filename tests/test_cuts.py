"""The matched opposition cut reaches the closest achievable share."""
import numpy as np

import config as cfg
from pipeline.cuts import matched_cut, opposition_cut
from pipeline.group_statistics import add_group_statistics


def test_matched_cut_on_a_small_example():
    values = [0, 0, 0.2, 0.5, 0.5, 1.0]
    assert matched_cut(values, target_share=0.5) == 0.5            # 3 of 6 values are >= 0.5
    assert np.isinf(matched_cut([0, 0, np.nan], target_share=0.3))


def test_opposition_cut_is_the_closest_achievable_share(replies_before_groups):
    replies = add_group_statistics(replies_before_groups)
    groups = replies[replies["group_size"] >= cfg.MIN_MEMBERS]
    target_share = (groups["group_variance"] >= cfg.VAR_CUT).mean()
    cut = opposition_cut(replies)
    achieved_share = (groups["group_opposed_pair_share"] >= cut).mean()
    shares = groups["group_opposed_pair_share"].dropna()
    best_distance = min(abs((groups["group_opposed_pair_share"] >= candidate).mean() - target_share)
                        for candidate in np.unique(shares[shares > 0]))
    assert abs(abs(achieved_share - target_share) - best_distance) < 1e-12


def test_given_opposition_cut_is_used_as_is(replies_before_groups):
    cfg.OPP_CUT = 0.4
    assert opposition_cut(add_group_statistics(replies_before_groups)) == 0.4
