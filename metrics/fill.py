"""
Cell-fill diagnostics: how many events support each part of a transition tensor p[s, influencer, k].

    row   = one pair (own class s, influencer category): one conditional distribution p(k | s, influencer)
    cell  = one triple (s, influencer category, new class k): one entry of that distribution

These functions only read an events table; they never build data or change the configuration.
"""
import numpy as np
import pandas as pd

import config as cfg
from categories import composition_registry, influencer_categories


def count_events_per_cell(events, influencer_column, settings=None):
    """Array N[s, influencer category, k] with the number of events in each cell (events missing s, the
    influencer or k are left out)."""
    number_of_classes = cfg.N_CLASSES
    number_of_categories = influencer_categories(influencer_column, settings)[0]
    usable = events.dropna(subset=[cfg.COL_S, influencer_column, cfg.COL_K])
    counts = np.zeros((number_of_classes, number_of_categories, number_of_classes), dtype=np.int64)
    np.add.at(counts, (usable[cfg.COL_S].to_numpy(int), usable[influencer_column].to_numpy(int),
                       usable[cfg.COL_K].to_numpy(int)), 1)          # add.at counts repeated indices every time
    return counts


def prior_weight(events_per_row, tau=None):
    """Event-weighted share of the estimates that comes from the shrinkage prior: a row with R events puts
    weight TAU / (R + TAU) on the prior."""
    tau = cfg.TAU if tau is None else tau
    total_events = events_per_row.sum()
    if total_events == 0:
        return np.nan
    return float((events_per_row * tau / (events_per_row + tau + 1e-12)).sum() / total_events)


def plugin_cmi_bias_bits(events_per_row):
    """Approximate upward bias (Miller-Madow) of the plug-in estimate of I(K; influencer | S), in bits:
    (m - 1) * sum over s of (non-empty categories - 1) / (2 N ln 2). It is what noise alone would 'find'."""
    total_events = events_per_row.sum()
    if total_events == 0:
        return np.nan
    nonempty_categories_per_class = (events_per_row > 0).sum(axis=1)
    return float((cfg.N_CLASSES - 1) * np.maximum(nonempty_categories_per_class - 1, 0).sum()
                 / (2 * total_events * np.log(2)))


def fill_table(events, influencer_column=None, settings=None):
    """One line per row (s, influencer category), empty rows included: events in the row, events per new class,
    and a flag for rows below MIN_ROW_EVENTS."""
    influencer_column = influencer_column or cfg.COL_C
    counts = count_events_per_cell(events, influencer_column, settings)
    number_of_classes, number_of_categories = cfg.N_CLASSES, counts.shape[1]
    category_labels = influencer_categories(influencer_column, settings)[1]
    table = pd.DataFrame({"own_class": np.repeat(np.arange(number_of_classes), number_of_categories),
                          "influencer_category": np.tile(np.arange(number_of_categories), number_of_classes)})
    table["own_class_label"] = [cfg.CLASS_LABELS[own_class] for own_class in table["own_class"]]
    table["influencer_label"] = [category_labels[category] for category in table["influencer_category"]]
    table["events_in_row"] = counts.sum(axis=2).reshape(-1)
    for new_class in range(number_of_classes):
        table[f"events_new_class_{new_class}"] = counts[:, :, new_class].reshape(-1)
    table["below_min_row_events"] = table["events_in_row"] < cfg.MIN_ROW_EVENTS
    return table


def fill_summary(events, influencer_column=None, thresholds=(10, 30, 100), settings=None):
    """One-line summary of how well an influencer summary is filled.
    rows_below_t: share of rows with fewer than t events; events_in_rows_below_t: share of the EVENTS sitting in
    such rows; prior_weight and cmi_bias_bits: see the functions of the same name."""
    influencer_column = influencer_column or cfg.COL_C
    counts = count_events_per_cell(events, influencer_column, settings)
    number_of_categories = counts.shape[1]
    events_per_row, total_events = counts.sum(axis=2), int(counts.sum())
    nonempty_rows = events_per_row[events_per_row > 0]
    summary = {"influencer": influencer_column, "categories": number_of_categories,
               "rows": cfg.N_CLASSES * number_of_categories, "rows_nonempty": int(len(nonempty_rows)),
               "events": total_events,
               "median_row_events": float(np.median(nonempty_rows)) if total_events else np.nan,
               "min_row_events_nonempty": int(nonempty_rows.min()) if total_events else 0,
               "share_of_empty_cells": float((counts == 0).mean())}
    for threshold in sorted(set(thresholds) | {cfg.MIN_ROW_EVENTS}):
        is_thin_row = events_per_row < threshold
        summary[f"rows_below_{threshold}"] = float(is_thin_row.mean())
        summary[f"events_in_rows_below_{threshold}"] = (float(events_per_row[is_thin_row].sum() / total_events)
                                                        if total_events else np.nan)
    summary["prior_weight"] = prior_weight(events_per_row)
    summary["cmi_bias_bits"] = plugin_cmi_bias_bits(events_per_row)
    return pd.DataFrame([summary])


def compare_fill(results, thresholds=(10, 30, 100)):
    """fill_summary of every summary side by side. The '(comp events)' rows put g and h on the same replies as c,
    so their difference from c comes only from the number of categories."""
    summaries = [("l  parent class", results["dyadic"], cfg.COL_L),
                 ("g  mean x variance", results["group"], cfg.COL_G),
                 ("h  mean x opposed pairs", results["group"], cfg.COL_H),
                 ("g  (comp events)", results["comp"], cfg.COL_G),
                 ("h  (comp events)", results["comp"], cfg.COL_H),
                 ("c  composition", results["comp"], cfg.COL_C)]
    table = pd.concat([fill_summary(events, column, thresholds).assign(summary=name)
                       for name, events, column in summaries], ignore_index=True)
    return table.set_index("summary")


def _size_range_label(min_size, max_size):
    if pd.isna(max_size):
        return f"{int(min_size)}+"
    if int(max_size) == int(min_size):
        return f"{int(min_size)}"
    return f"{int(min_size)}-{int(max_size)}"


def fill_by_size(composition_events, settings=None):
    """Composition events per group-size class: possible cells, cells observed, events, events per row
    (events / cells / classes) and the share of that class's rows below MIN_ROW_EVENTS."""
    registry = composition_registry(settings)
    events_per_row = count_events_per_cell(composition_events, cfg.COL_C, settings).sum(axis=2)   # (s, cell)
    lines = []
    for size_class, cells_of_class in registry.groupby("size_class"):
        cell_indices = cells_of_class["cell"].to_numpy()
        rows_of_class = events_per_row[:, cell_indices]
        events_of_class = int(rows_of_class.sum())
        lines.append({"size_class": size_class,
                      "sizes": _size_range_label(cells_of_class["min_size"].iloc[0], cells_of_class["max_size"].iloc[0]),
                      "cells": len(cell_indices),
                      "cells_observed": int((rows_of_class.sum(axis=0) > 0).sum()),
                      "events": events_of_class,
                      "events_per_row": events_of_class / len(cell_indices) / cfg.N_CLASSES,
                      "rows_below_min": float((rows_of_class < cfg.MIN_ROW_EVENTS).mean())})
    table = pd.DataFrame(lines)
    table["share_of_events"] = table["events"] / table["events"].sum()
    return table
