"""
Compatibility layer: the old `events` interface, now implemented by categories.py, pipeline/ and metrics/.

It exists so that tensors.py and notebooks/analysis.ipynb keep working unchanged. Every function here keeps its
old name and returns tables with the OLD column names (x, t, user, g_n, g_mu, R, ...). New code should import
from the new modules, which use descriptive names:

    old (events.*)                 new
    load_raw                       pipeline.loading.load_raw_tables
    to_class, count_cols           categories.score_to_class, categories.class_count_columns
    build_items                    pipeline.items.build_items
    build_replies                  pipeline.replies.attach_parents
    add_own_opinion                pipeline.own_opinion.add_own_opinion + add_new_opinion_class
    add_group_stats, _pair_stats   pipeline.group_statistics.add_group_statistics, pair_disagreement_shares
    composition_cells, assign_cells, influencer_info
                                   categories.composition_registry, assign_composition_cells, influencer_categories
    matched_cut, opp_cut_for       pipeline.cuts.matched_cut, opposition_cut
    funnel, _format, _group_cols   pipeline.event_tables.funnel_table, format_events_table, group_summary_columns
    prepare, build_events, save_events
                                   pipeline.run.prepare_replies, build_events, save_events
    fill_table, fill_summary, compare_fill, fill_by_size
                                   metrics.fill (same names)
    scan_compositions              experiments.composition_scan.scan_composition_settings
    selftest                       tests/ (run with: python -m pytest tests)

The events tables themselves (dyadic, group, comp) have the same columns in both interfaces (cfg.EVENT_COLS).
Remove this file once tensors.py and the notebooks import the new modules.
"""
import subprocess
import sys
from pathlib import Path

import config as cfg
import categories
from experiments import composition_scan
from metrics import fill
from pipeline import cuts, event_tables, group_statistics, items as items_stage, loading, own_opinion, replies as replies_stage
from pipeline import run

_MAX_CLASSES = 3

# new column name -> old column name, for the items and the replies tables
_ITEM_COLUMNS_OLD = {"item_kind": "kind", "thread_id": "thread", "parent_item_id": "parent_item", "author": "user",
                     "created_at": "t", "stance_score": "x"}
_REPLY_COLUMNS_OLD = {
    "item_kind": "kind", "thread_id": "thread", "parent_item_id": "parent_item", "author": "user",
    "reply_time": "t_reply", "reply_score": "x", "parent_author": "parent_user", "parent_time": "t_parent",
    "parent_score": "x_parent", "previous_item_score": "x_prev", "previous_item_time": "t_prev",
    "own_opinion_age_days": "gap", "own_class": cfg.COL_S, "new_class": cfg.COL_K,
    "group_size": "g_n", "group_mean": "g_mu", "group_variance": "g_var", "group_variance_population": "g_var_pop",
    "group_opposed_pair_share": "g_opp", "group_mild_pair_share": "g_mild",
    **{f"group_count_class_{j}": f"g_c{j}" for j in range(_MAX_CLASSES)}}
_REGISTRY_COLUMNS_OLD = {"cell": "c", "size_class": "bin", "min_size": "n_lo", "max_size": "n_hi"}
_FILL_COLUMNS_OLD = {"own_class": "s", "influencer_category": "cell", "own_class_label": "s_label",
                     "influencer_label": "label", "events_in_row": "R", "below_min_row_events": "low",
                     **{f"events_new_class_{j}": f"n_k{j}" for j in range(_MAX_CLASSES)}}
_SUMMARY_COLUMNS_OLD = {"median_row_events": "median_R", "min_row_events_nonempty": "min_R_nonempty",
                        "share_of_empty_cells": "cells_sk_zero", "number_of_classes": "m"}
_BY_SIZE_COLUMNS_OLD = {"size_class": "bin", "events_per_row": "events_per_cell"}


def _inverse(mapping):
    return {old: new for new, old in mapping.items()}


def _items_to_old(items):
    return items.rename(columns=_ITEM_COLUMNS_OLD)


def _items_to_new(items):
    return items.rename(columns=_inverse(_ITEM_COLUMNS_OLD))


def _replies_to_old(replies):
    return replies.rename(columns=_REPLY_COLUMNS_OLD)


def _replies_to_new(replies):
    return replies.rename(columns=_inverse(_REPLY_COLUMNS_OLD))


# ---- stages 0-3 ---------------------------------------------------------------------------
def load_raw(use_cache=True):
    return loading.load_raw_tables(use_cache)


to_class = categories.score_to_class


def count_cols():
    return [f"g_c{j}" for j in range(cfg.N_CLASSES)]


def build_items(threads, submissions):
    return _items_to_old(items_stage.build_items(threads, submissions))


def build_replies(items):
    return _replies_to_old(replies_stage.attach_parents(_items_to_new(items)))


def add_own_opinion(rep, items):
    replies = own_opinion.add_own_opinion(_replies_to_new(rep), _items_to_new(items))
    return _replies_to_old(own_opinion.add_new_opinion_class(replies))


# ---- stage 4 and compositions -------------------------------------------------------------
def add_group_stats(rep):
    return _replies_to_old(group_statistics.add_group_statistics(_replies_to_new(rep)))


_pair_stats = group_statistics.pair_disagreement_shares


def composition_cells():
    return categories.composition_registry().rename(columns=_REGISTRY_COLUMNS_OLD)


def assign_cells(counts, seed=None):
    return categories.assign_composition_cells(counts, seed=seed)


influencer_info = categories.influencer_categories


# ---- stage 5 ------------------------------------------------------------------------------
matched_cut = cuts.matched_cut
funnel = event_tables.funnel_table


def opp_cut_for(d):
    return cuts.opposition_cut(_replies_to_new(d))


def _format(d, **cols):
    return event_tables.format_events_table(_replies_to_new(d), **cols)


def _group_cols(d, opp_cut):
    return event_tables.group_summary_columns(_replies_to_new(d), opp_cut)


def prepare(threads=None, submissions=None):
    items, replies, common_replies, funnel_steps = run.prepare_replies(threads, submissions)
    return _items_to_old(items), _replies_to_old(replies), _replies_to_old(common_replies), funnel_steps


def build_events(threads=None, submissions=None, verbose=True):
    results = run.build_events(threads, submissions, verbose)
    results["items"] = _items_to_old(results["items"])
    results["replies"] = _replies_to_old(results["replies"])
    return results


save_events = run.save_events


# ---- section 6: fill diagnostics ----------------------------------------------------------
def fill_table(ev, col=None):
    return fill.fill_table(ev, col).rename(columns=_FILL_COLUMNS_OLD)


def fill_summary(ev, col=None, thresholds=(10, 30, 100)):
    return fill.fill_summary(ev, col, thresholds).rename(columns=_SUMMARY_COLUMNS_OLD)


def compare_fill(res, thresholds=(10, 30, 100)):
    return fill.compare_fill(res, thresholds).rename(columns=_SUMMARY_COLUMNS_OLD)


def fill_by_size(ev):
    return fill.fill_by_size(ev).rename(columns=_BY_SIZE_COLUMNS_OLD)


def scan_compositions(threads=None, submissions=None, ms=(2, 3), max_sizes=(3, 4, 5, 6), grids=(3, 4, 6), verbose=False):
    scan = composition_scan.scan_composition_settings(threads, submissions, class_numbers=ms,
                                                      max_group_sizes=max_sizes, share_grids=grids, verbose=verbose)
    return scan.rename(columns=_SUMMARY_COLUMNS_OLD)


# ---- tests --------------------------------------------------------------------------------
def selftest(verbose=True):
    """Run the test suite in tests/ (needs pytest)."""
    tests_folder = Path(__file__).resolve().parent / "tests"
    arguments = [sys.executable, "-m", "pytest", str(tests_folder)] + ([] if verbose else ["-q"])
    completed = subprocess.run(arguments)
    if completed.returncode != 0:
        raise AssertionError("tests failed")
    return True


if __name__ == "__main__":
    selftest()
