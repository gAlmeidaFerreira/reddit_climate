"""Own opinion: the replier's last scored item strictly before the parent, checked against brute force."""
import pandas as pd

import config as cfg
from categories import score_to_class


def test_previous_item_is_the_last_scored_item_before_the_parent(replies_before_groups, brute_force):
    checked = 0
    for _, reply in replies_before_groups.iterrows():
        answer = brute_force.get((reply["subreddit"], reply["item_id"]))
        if answer is None or not reply["parent_found"]:
            continue
        expected = answer["previous_item_time"]
        if pd.isna(expected):
            assert pd.isna(reply["previous_item_time"]), f"{reply['item_id']}: own opinion should be missing"
        else:
            assert reply["previous_item_time"] == expected, f"{reply['item_id']}: wrong own-opinion item"
        checked += 1
    assert checked > 100


def test_own_and_new_class_come_from_the_right_scores(replies_before_groups):
    replies = replies_before_groups
    assert replies["own_class"].equals(pd.Series(score_to_class(replies["previous_item_score"], cfg.CUTS),
                                                  index=replies.index))
    assert replies["new_class"].equals(pd.Series(score_to_class(replies["reply_score"], cfg.CUTS), index=replies.index))


def test_age_of_own_opinion_is_measured_to_the_parent(replies_before_groups):
    has_previous = replies_before_groups.dropna(subset=["previous_item_time"])
    assert (has_previous["own_opinion_age_days"] > 0).all()            # strictly earlier than the parent
