"""
Stage 3. Own opinion s of the replier, and the new opinion k expressed by the reply.

Own opinion: the class of the replier's LAST SCORED item (comment or submission, any thread, any subreddit)
written STRICTLY BEFORE the parent's timestamp. Cutting at the parent, not at the reply, keeps anything the
replier wrote after seeing the parent out of the own opinion (leakage protection); an item at exactly the
parent's second is excluded too.

Columns added by add_own_opinion:
    previous_item_score, previous_item_time   the item that gives the own opinion
    own_opinion_age_days                      days between that item and the parent
    own_class                                 class of previous_item_score (missing: no earlier scored item)
Column added by add_new_opinion_class:
    new_class                                 class of the reply's own score
"""
import pandas as pd

import config as cfg
from categories import score_to_class

SECONDS_PER_DAY = 86400


def add_own_opinion(replies, items):
    """Find, for every reply, the replier's last scored item before the parent (merge_asof by author)."""
    scored_history = (items.loc[items["author"].notna() & items["stance_score"].notna(),
                                ["author", "created_at", "stance_score"]]
                      .rename(columns={"created_at": "previous_item_time", "stance_score": "previous_item_score"})
                      .sort_values("previous_item_time"))
    can_look_up = replies["author"].notna() & replies["parent_time"].notna()
    lookups = (replies.loc[can_look_up, ["author", "parent_time"]]
               .assign(reply_row=replies.index[can_look_up])             # merge_asof needs time order; keep the row
               .sort_values("parent_time"))
    previous_items = pd.merge_asof(lookups, scored_history, left_on="parent_time", right_on="previous_item_time",
                                   by="author", allow_exact_matches=False).set_index("reply_row")

    replies = replies.copy()
    replies["previous_item_score"] = previous_items["previous_item_score"]
    replies["previous_item_time"] = previous_items["previous_item_time"]
    replies["own_opinion_age_days"] = ((replies["parent_time"] - replies["previous_item_time"]).dt.total_seconds()
                                       / SECONDS_PER_DAY)
    replies["own_class"] = score_to_class(replies["previous_item_score"], cfg.CUTS)
    return replies


def add_new_opinion_class(replies):
    """new_class = class of the reply's own score (missing for NI replies)."""
    replies = replies.copy()
    replies["new_class"] = score_to_class(replies["reply_score"], cfg.CUTS)
    return replies
