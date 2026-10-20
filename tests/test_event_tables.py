"""The whole pipeline: events tables, their columns, l across tables, composition cells and diagnostics."""
import numpy as np
import pytest

import config as cfg
from categories import CompositionSettings, class_count_columns, composition_registry
from metrics.fill import compare_fill, fill_by_size, fill_summary, fill_table
from pipeline.event_tables import composition_events
from pipeline.run import build_events, prepare_replies

EVENT_TABLES = ("dyadic", "group", "comp")


@pytest.fixture
def results(raw_tables, number_of_classes):
    return build_events(*raw_tables, verbose=False)


def test_every_table_has_the_event_columns(results):
    for name in EVENT_TABLES:
        assert list(results[name].columns) == list(cfg.EVENT_COLS)
    assert results["group"][cfg.COL_G].dropna().between(0, cfg.N_G - 1).all()
    assert results["dyadic"][cfg.COL_L].dropna().between(0, cfg.N_CLASSES - 1).all()


def test_funnels_start_with_all_comments_and_end_with_the_table(results, raw_tables):
    for name in EVENT_TABLES:
        funnel = results["funnel_" + name]
        assert funnel["n"].iloc[0] == len(raw_tables[0])
        assert funnel["n"].iloc[-1] == len(results[name])
        assert funnel["n"].is_monotonic_decreasing


def test_parent_class_is_the_same_in_every_table(results):
    """l in the group and composition tables is the dyadic l, and exists exactly where the dyadic event exists."""
    dyadic = results["dyadic"]
    for name in ("group", "comp"):
        with_parent_class = results[name][cfg.COL_L].dropna()
        assert with_parent_class.index.isin(dyadic.index).all()
        assert (dyadic.loc[with_parent_class.index, cfg.COL_L] == with_parent_class).all()
        shared_replies = results[name].index.intersection(dyadic.index)
        assert results[name].loc[shared_replies, cfg.COL_L].notna().all()


def test_options_that_need_explicit_members_run(raw_tables):
    cfg.AUTHORS_ONCE, cfg.M_MAX = True, 3
    results = build_events(*raw_tables, verbose=False)
    assert (results["group"]["n_members"] <= 3).all()


@pytest.mark.parametrize("settings", [CompositionSettings("counts", 5, 4, (), 2),
                                      CompositionSettings("counts", 3, 4, (), 2),
                                      CompositionSettings("shares", 5, 4, (), 2),
                                      CompositionSettings("shares", 5, 6, (3, 5), 2)],
                         ids=["counts n<=5", "counts n<=3", "shares 1/4", "shares 1/6 with size classes"])
def test_composition_cells_decode_to_the_group(raw_tables, number_of_classes, settings):
    _, replies, common_replies, funnel_steps = prepare_replies(*raw_tables)
    events = composition_events(common_replies, list(funnel_steps), settings=settings)
    assert len(events) > 0 and events[cfg.COL_C].notna().all()
    registry = composition_registry(settings).set_index("cell")
    cells = events[cfg.COL_C].astype(int)
    parts = np.array(registry.loc[cells, "parts"].tolist())
    class_counts = replies.loc[events.index, class_count_columns()].to_numpy().astype(int)
    group_size = events["n_members"].to_numpy()
    if settings.mode == "counts":
        assert (parts == class_counts).all(), "a counts cell must decode to the group's class counts"
        assert (group_size <= settings.max_group_size).all()
    else:
        assert (parts.sum(axis=1) == settings.share_grid).all()
        true_shares = class_counts / class_counts.sum(axis=1, keepdims=True)
        assert np.abs(parts / settings.share_grid - true_shares).max() <= 1 / settings.share_grid + 1e-9
        min_size = registry.loc[cells, "min_size"].to_numpy(dtype=float)
        max_size = registry.loc[cells, "max_size"].astype("Float64").fillna(np.inf).to_numpy(dtype=float)
        assert ((group_size >= min_size) & (group_size <= max_size)).all(), "wrong size class"

    # diagnostics agree with the events
    table = fill_table(events, cfg.COL_C, settings)
    assert table["events_in_row"].sum() == len(events) and len(table) == cfg.N_CLASSES * len(registry)
    assert fill_by_size(events, settings)["events"].sum() == len(events)
    assert int(fill_summary(events, cfg.COL_C, settings=settings)["events"].iloc[0]) == len(events)


def test_compare_fill_runs(results):
    table = compare_fill(results)
    assert len(table) == 6 and (table["events"] > 0).all()
