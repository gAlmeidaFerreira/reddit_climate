"""
The whole pipeline, from the raw tables to the saved events tables.

    raw tables -> items -> replies (+ parent) -> own opinion s, new class k -> group statistics
               -> common filters -> dyadic / group / composition events
"""
import json

import config as cfg
from categories import composition_registry
from pipeline.event_tables import (apply_common_filters, composition_events, dyadic_events, funnel_table,
                                   group_events)
from pipeline.cuts import opposition_cut
from pipeline.group_statistics import add_group_statistics
from pipeline.items import build_items
from pipeline.loading import load_raw_tables
from pipeline.own_opinion import add_new_opinion_class, add_own_opinion
from pipeline.replies import attach_parents

EVENT_TABLE_NAMES = ("dyadic", "group", "comp")


def build_replies_table(items):
    """Items -> replies with parent, own opinion, new class and group statistics (nothing filtered)."""
    replies = attach_parents(items)
    replies = add_own_opinion(replies, items)
    replies = add_new_opinion_class(replies)
    return add_group_statistics(replies)


def prepare_replies(raw_comments=None, raw_submissions=None):
    """Stages 0-4 and the common filters. Returns (items, replies, common_replies, funnel_steps).
    Loads the raw tables when they are not given."""
    if raw_comments is None or raw_submissions is None:
        raw_comments, raw_submissions = load_raw_tables()
    items = build_items(raw_comments, raw_submissions)
    replies = build_replies_table(items)
    common_replies, funnel_steps = apply_common_filters(replies)
    return items, replies, common_replies, funnel_steps


def _print_summary(results):
    group = results["group"]
    cut_source = "given" if cfg.OPP_CUT is not None else "matched to VAR_CUT"
    print(f"OPP_CUT used: {results['opp_cut']:.3f} ({cut_source}) | share of group events with high variance "
          f"{(group['var'] >= cfg.VAR_CUT).mean():.3f}, with high opposition {(group['opp'] >= results['opp_cut']).mean():.3f}")
    for name in EVENT_TABLE_NAMES:
        print(f"\n=== funnel_{name} ===\n{results['funnel_' + name].to_string(index=False)}")


def build_events(raw_comments=None, raw_submissions=None, verbose=True):
    """Whole pipeline. Returns a dict with
        items, replies, opp_cut, dyadic, group, comp, funnel_dyadic, funnel_group, funnel_comp.
    One opposition cut, computed on the common replies, is used by the group and composition tables."""
    items, replies, common_replies, funnel_steps = prepare_replies(raw_comments, raw_submissions)
    shared_opposition_cut = opposition_cut(common_replies)
    steps_dyadic, steps_group, steps_composition = list(funnel_steps), list(funnel_steps), list(funnel_steps)
    results = {"items": items, "replies": replies, "opp_cut": shared_opposition_cut,
               "dyadic": dyadic_events(common_replies, steps_dyadic),
               "group": group_events(common_replies, steps_group, shared_opposition_cut),
               "comp": composition_events(common_replies, steps_composition, shared_opposition_cut)}
    results.update(funnel_dyadic=funnel_table(steps_dyadic), funnel_group=funnel_table(steps_group),
                   funnel_comp=funnel_table(steps_composition))
    if verbose:
        _print_summary(results)
    return results


def save_events(results, folder=None):
    """Write the three events tables, their funnels, the opposition cut, the composition registry and the
    parameters of this run (default folder: cfg.DATA_CACHE)."""
    folder = cfg.DATA_CACHE if folder is None else folder
    folder.mkdir(parents=True, exist_ok=True)
    for name in EVENT_TABLE_NAMES:
        results[name].to_parquet(folder / f"events_{name}.parquet")
        results["funnel_" + name].to_csv(folder / f"funnel_{name}.csv", index=False)
    (folder / "opp_cut.json").write_text(json.dumps({"opp_cut": results.get("opp_cut")}))
    registry = composition_registry()
    registry["parts"] = registry["parts"].astype(str)
    registry.to_csv(folder / "composition_cells.csv", index=False)
    cfg.save(folder / "config_used.json")
