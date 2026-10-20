"""
Stage 2. Replies: every comment with the information of its parent on the same row.

Each comment has exactly one parent (a comment or the submission); a parent can have many replies, so the
join is many-to-one. Nothing is filtered here: a comment whose parent is missing keeps its row with
parent_found = False.

Columns added to the comment's own columns (created_at -> reply_time, stance_score -> reply_score):
    parent_author, parent_time, parent_score, parent_found

The row index (0 .. number of comments - 1) is the reply's identifier in every later stage.
"""
import pandas as pd

PARENT_KEY = ["subreddit", "parent_kind", "parent_item_id"]


def attach_parents(items):
    """Items table -> replies table, one row per comment."""
    possible_parents = (items[["subreddit", "item_id", "item_kind", "author", "created_at", "stance_score"]]
                        .rename(columns={"item_id": "parent_item_id", "item_kind": "parent_kind",
                                         "author": "parent_author", "created_at": "parent_time",
                                         "stance_score": "parent_score"}))
    comments = (items[items["item_kind"] == "comment"]
                .rename(columns={"created_at": "reply_time", "stance_score": "reply_score"}))
    replies = comments.merge(possible_parents, on=PARENT_KEY, how="left", validate="many_to_one",
                             indicator="parent_match")       # validate: a duplicated parent key raises an error
    replies["parent_found"] = replies.pop("parent_match").eq("both").to_numpy()
    replies = replies.reset_index(drop=True)
    assert len(replies) == len(comments), "the parent join changed the number of replies"
    return replies
