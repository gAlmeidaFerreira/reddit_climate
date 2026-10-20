"""
Stage 5. Filters, funnel and the three events tables.

Every events table has one row per reply (the transition own class s -> new class k under some influence) and
the same columns, cfg.EVENT_COLS, so tensors.py can treat them alike; columns a table does not use are missing.

    dyadic       influencer l = class of the parent's score (scored parent, written by someone else)
    group        influencers g = 2 * lean + variance level and h = 2 * lean + opposition level
                 (at least MIN_MEMBERS scored members); l is also filled where the reply is a dyadic event
    composition  the group events (capped at max_group_size in counts mode) with the composition cell c as well

The tables overlap: the same reply keeps the same index (its row in the replies table) in every table.
Note that the group table also holds self-replies and replies to unscored parents, which are not dyadic events.

The funnel records how many replies each filter keeps: a list of (step name, rows kept).
"""
import numpy as np
import pandas as pd

import config as cfg
from categories import CompositionSettings, assign_composition_cells, class_count_columns, score_to_class
from pipeline.cuts import opposition_cut

_INTEGER_EVENT_COLUMNS = (cfg.COL_S, cfg.COL_L, cfg.COL_G, cfg.COL_C, cfg.COL_H, cfg.COL_LEAN, cfg.COL_K, "n_members")


# =======================================================================================
# Funnel
# =======================================================================================
def record_step(funnel_steps, step_name, kept_replies):
    """Append (step_name, number of rows kept) to the funnel and return the rows unchanged."""
    funnel_steps.append((step_name, len(kept_replies)))
    return kept_replies


def funnel_table(funnel_steps):
    """Funnel steps -> table with rows kept, the share of the previous step and the share of all comments."""
    funnel = pd.DataFrame(funnel_steps, columns=["step", "n"])
    funnel["share_of_previous"] = (funnel["n"] / funnel["n"].shift(1)).round(3)
    funnel["share_of_comments"] = (funnel["n"] / funnel["n"].iloc[0]).round(3)
    return funnel


# =======================================================================================
# Filters every events table shares
# =======================================================================================
def apply_common_filters(replies):
    """Replies usable for any events table, and the funnel steps that led to them."""
    funnel_steps = []
    kept = record_step(funnel_steps, "comments", replies)
    kept = record_step(funnel_steps, "reply is scored (not NI)", kept[kept["reply_score"].notna()])
    kept = record_step(funnel_steps, "author known", kept[kept["author"].notna()])
    kept = record_step(funnel_steps, "parent present in the data", kept[kept["parent_found"]])
    first_day = pd.Timestamp(cfg.FIRST_DATE)
    day_after_last = pd.Timestamp(cfg.LAST_DATE) + pd.Timedelta(days=1)          # the whole last day is included
    in_period = (kept["reply_time"] >= first_day) & (kept["reply_time"] < day_after_last)
    kept = record_step(funnel_steps, f"reply within {cfg.FIRST_DATE} .. {cfg.LAST_DATE}", kept[in_period])
    kept = record_step(funnel_steps, "replier has an earlier scored item (own opinion s)", kept[kept["own_class"].notna()])
    if cfg.MAX_GAP_DAYS is not None:
        kept = record_step(funnel_steps, f"own-opinion item at most {cfg.MAX_GAP_DAYS} days old",
                           kept[kept["own_opinion_age_days"] <= cfg.MAX_GAP_DAYS])
    return kept, funnel_steps


# =======================================================================================
# Events tables
# =======================================================================================
def format_events_table(replies, **event_columns):
    """Final events table: the columns every table shares, plus event_columns (name in cfg.EVENT_COLS -> values),
    in the order of cfg.EVENT_COLS. Category columns are pandas Int64, so they can be missing (<NA>)."""
    events = pd.DataFrame({"thread": replies["thread_id"], "subreddit": replies["subreddit"],
                           "user": replies["author"], "t_reply": replies["reply_time"],
                           "gap": replies["own_opinion_age_days"], "parent_kind": replies["parent_kind"],
                           cfg.COL_S: replies["own_class"], cfg.COL_K: replies["new_class"]})
    for column, values in event_columns.items():
        events[column] = values
    for column in _INTEGER_EVENT_COLUMNS:
        events[column] = (events[column].astype("Int64") if column in events
                          else pd.array([pd.NA] * len(events), dtype="Int64"))
    return events.reindex(columns=list(cfg.EVENT_COLS))


def _parent_class_where_dyadic(replies):
    """Class of the parent's score where the reply is a dyadic event (scored parent by someone else), else NaN."""
    if not {"parent_score", "parent_author", "author"} <= set(replies.columns):    # tables without parents (tests)
        return np.full(len(replies), np.nan)
    is_dyadic = (replies["parent_score"].notna() & (replies["parent_author"] != replies["author"])).to_numpy()
    return np.where(is_dyadic, score_to_class(replies["parent_score"], cfg.CUTS), np.nan)


def group_summary_columns(replies, opposition_cut_value):
    """Event columns shared by the group and composition tables:
        lean = class of the group mean (MEAN_CUTS),  g = 2 * lean + (variance >= VAR_CUT),
        h = 2 * lean + (opposed-pair share >= opposition cut),  l where the reply is also a dyadic event,
        and the raw statistics n_members, mu, var, opp, mild."""
    lean = score_to_class(replies["group_mean"], cfg.MEAN_CUTS)
    is_high_variance = (replies["group_variance"] >= cfg.VAR_CUT).astype(float)
    is_high_opposition = (replies["group_opposed_pair_share"] >= opposition_cut_value).astype(float)
    return {cfg.COL_L: _parent_class_where_dyadic(replies),
            cfg.COL_G: 2 * lean + is_high_variance,
            cfg.COL_H: 2 * lean + is_high_opposition,
            cfg.COL_LEAN: lean,
            "opp": replies["group_opposed_pair_share"], "mild": replies["group_mild_pair_share"],
            "n_members": replies["group_size"], "mu": replies["group_mean"], "var": replies["group_variance"]}


def dyadic_events(common_replies, funnel_steps):
    """Replies to a scored parent written by someone else: s, l (the parent's class), k.
    n_members = 1, mu = the parent's score and var = 0 only give the table the shared columns."""
    kept = record_step(funnel_steps, "parent is scored", common_replies[common_replies["parent_score"].notna()])
    kept = record_step(funnel_steps, "not a self-reply  ->  dyadic events", kept[kept["parent_author"] != kept["author"]])
    return format_events_table(kept, **{cfg.COL_L: score_to_class(kept["parent_score"], cfg.CUTS),
                                         "n_members": 1, "mu": kept["parent_score"], "var": 0.0})


def _replies_with_a_group(common_replies, funnel_steps, step_name):
    has_group = common_replies["group_size"] >= cfg.MIN_MEMBERS
    return record_step(funnel_steps, step_name, common_replies[has_group])


def group_events(common_replies, funnel_steps, opposition_cut_value=None):
    """Replies with at least MIN_MEMBERS scored group members: s, g, h, lean (and l where dyadic), k."""
    kept = _replies_with_a_group(common_replies, funnel_steps,
                                 f"group has >= {cfg.MIN_MEMBERS} scored members  ->  group events")
    if opposition_cut_value is None:
        opposition_cut_value = opposition_cut(kept)
    return format_events_table(kept, **group_summary_columns(kept, opposition_cut_value))


def composition_events(common_replies, funnel_steps, opposition_cut_value=None, settings=None):
    """The group events (in counts mode only groups up to max_group_size) with the composition cell c next to
    g, h and l, so all the summaries can be compared on exactly the same replies."""
    settings = CompositionSettings.from_config() if settings is None else settings
    kept = _replies_with_a_group(common_replies, funnel_steps, f"group has >= {cfg.MIN_MEMBERS} scored members")
    if opposition_cut_value is None:
        opposition_cut_value = opposition_cut(kept)
    if settings.mode == "counts":
        kept = record_step(funnel_steps, f"group has at most {settings.max_group_size} members  ->  composition events",
                           kept[kept["group_size"] <= settings.max_group_size])
    else:
        record_step(funnel_steps, "composition events (shares: no size cap)", kept)
    composition_cells = assign_composition_cells(kept[class_count_columns()].to_numpy(), settings)
    return format_events_table(kept, **{cfg.COL_C: composition_cells,
                                        **group_summary_columns(kept, opposition_cut_value)})
