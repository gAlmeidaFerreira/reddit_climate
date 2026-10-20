"""
Pipeline from the raw Reddit tables to the events tables (Stages 1-3 of the algorithm).

    loading.py           stage 0: raw comment and submission tables
    items.py             stage 1: comments and submissions in one format
    replies.py           stage 2: every comment with its parent
    own_opinion.py       stage 3: own opinion s and new class k
    group_statistics.py  stage 4: group size, mean, variance, class counts, opposed / mild pair shares
    cuts.py              cuts chosen from the data (opposition cut matched to VAR_CUT)
    event_tables.py      stage 5: common filters, funnel, dyadic / group / composition events
    run.py               the whole pipeline and saving

Usage:
    from pipeline import build_events, save_events
    results = build_events()
"""
from pipeline.run import build_events, build_replies_table, prepare_replies, save_events

__all__ = ["build_events", "build_replies_table", "prepare_replies", "save_events"]
