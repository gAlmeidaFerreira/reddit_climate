"""
Stage 1. Items: every comment and every submission in one common format.

Columns of the items table:
    subreddit        anonymised subreddit, e.g. "r/2"
    item_id          comment id or submission id
    item_kind        "comment" or "submission"
    thread_id        id of the submission the item belongs to (a submission is its own thread)
    parent_item_id   id of the item a comment replies to (None for submissions)
    parent_kind      "comment" or "submission" (None for submissions, or a parent reference without a known prefix)
    author           author id (missing for deleted accounts)
    created_at       creation time, naive UTC
    stance_score     score in [0, 1]; NaN for NI (not related to climate change)

An item is identified by (subreddit, item_id, item_kind), never by item_id alone.
"""
import re

import numpy as np
import pandas as pd

import config as cfg

ITEM_COLUMNS = ["subreddit", "item_id", "item_kind", "thread_id", "parent_item_id", "parent_kind",
                "author", "created_at", "stance_score"]

_PARENT_PREFIX_PATTERN = re.compile(
    f"^({re.escape(cfg.PARENT_PREFIX_COMMENT)}|{re.escape(cfg.PARENT_PREFIX_SUBMISSION)})")


def _parse_utc_time(raw_times):
    """Text timestamps -> naive UTC timestamps (strings without a time zone are read as UTC; unreadable -> NaT)."""
    return pd.to_datetime(raw_times, utc=True, errors="coerce").dt.tz_convert(None)


def _without_submission_prefix(raw_ids):
    """'t3_sub3423' -> 'sub3423'; ids without the prefix are kept as they are."""
    return raw_ids.astype(object).str.replace(f"^{re.escape(cfg.PARENT_PREFIX_SUBMISSION)}", "", regex=True)


def _parent_kind_from_reference(parent_references):
    """'t1_...' -> 'comment', 't3_...' -> 'submission', anything else -> None."""
    replies_to_comment = parent_references.str.startswith(cfg.PARENT_PREFIX_COMMENT, na=False)
    replies_to_submission = parent_references.str.startswith(cfg.PARENT_PREFIX_SUBMISSION, na=False)
    return np.where(replies_to_comment, "comment", np.where(replies_to_submission, "submission", None))


def _items_table(raw_table, item_id_column, item_kind, parent_item_id, parent_kind):
    """One raw table -> items columns. Built from arrays so pandas never aligns rows on the index."""
    return pd.DataFrame({
        "subreddit": raw_table[cfg.RAW_SUBREDDIT_ID].to_numpy(),
        "item_id": raw_table[item_id_column].to_numpy(),
        "item_kind": item_kind,
        "thread_id": _without_submission_prefix(raw_table[cfg.RAW_SUBMISSION_ID]).to_numpy(),
        "parent_item_id": parent_item_id,
        "parent_kind": parent_kind,
        "author": raw_table[cfg.RAW_AUTHOR_ID].to_numpy(),
        "created_at": _parse_utc_time(raw_table[cfg.RAW_TIME]).to_numpy(),
        "stance_score": pd.to_numeric(raw_table[cfg.RAW_SCORE], errors="coerce").to_numpy()})


def build_items(raw_comments, raw_submissions):
    """Raw comment and submission tables -> items table (submissions first, then comments)."""
    parent_references = raw_comments[cfg.RAW_PARENT_ID].astype(object)
    comments = _items_table(
        raw_comments, item_id_column=cfg.RAW_COMMENT_ID, item_kind="comment",
        parent_item_id=parent_references.str.replace(_PARENT_PREFIX_PATTERN, "", regex=True).to_numpy(),
        parent_kind=_parent_kind_from_reference(parent_references))
    submissions = _items_table(raw_submissions, item_id_column=cfg.RAW_SUBMISSION_ID, item_kind="submission",
                               parent_item_id=None, parent_kind=None)
    return pd.concat([submissions, comments], ignore_index=True)[ITEM_COLUMNS]
