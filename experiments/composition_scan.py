"""
Experiment: how well can the data fill the composition summary c under different definitions?

For each number of classes m, the replies are built once; then every composition setting (size caps in 'counts'
mode, grids in 'shares' mode) is applied to the same replies and summarised with metrics.fill.fill_summary.
The composition settings are passed as arguments; only the number of classes is set in config (and restored).
"""
import pandas as pd

import config as cfg
from categories import CompositionSettings
from metrics.fill import fill_summary
from pipeline.event_tables import composition_events
from pipeline.run import prepare_replies

REPORTED_COLUMNS = ["number_of_classes", "mode", "option", "categories", "rows", "events", "median_row_events",
                    "min_row_events_nonempty"]


def _settings_to_scan(max_group_sizes, share_grids):
    default = CompositionSettings.from_config()
    count_settings = [CompositionSettings("counts", max_size, default.share_grid, default.size_class_edges,
                                          default.min_members)
                      for max_size in max_group_sizes if max_size >= default.min_members]
    share_settings = [CompositionSettings("shares", default.max_group_size, grid, default.size_class_edges,
                                          default.min_members)
                      for grid in share_grids]
    return count_settings + share_settings


def _option_label(settings):
    if settings.mode == "counts":
        return f"n<={settings.max_group_size}"
    return f"grid 1/{settings.share_grid}"


def scan_composition_settings(raw_comments=None, raw_submissions=None, class_numbers=(2, 3),
                              max_group_sizes=(3, 4, 5, 6), share_grids=(3, 4, 6), verbose=False):
    """One line per (number of classes, composition setting) with the fill summary of c."""
    original_number_of_classes = cfg.N_CLASSES
    lines = []
    try:
        for number_of_classes in class_numbers:
            cfg.set_classes(number_of_classes)
            _, _, common_replies, funnel_steps = prepare_replies(raw_comments, raw_submissions)
            for settings in _settings_to_scan(max_group_sizes, share_grids):
                events = composition_events(common_replies, list(funnel_steps), settings=settings)
                line = fill_summary(events, cfg.COL_C, settings=settings).iloc[0].to_dict()
                line.update(number_of_classes=number_of_classes, mode=settings.mode, option=_option_label(settings))
                lines.append(line)
                if verbose:
                    print(number_of_classes, settings, line["categories"], line["events"])
    finally:
        cfg.set_classes(original_number_of_classes)
    columns = REPORTED_COLUMNS + [f"rows_below_{cfg.MIN_ROW_EVENTS}", f"events_in_rows_below_{cfg.MIN_ROW_EVENTS}",
                                  "prior_weight", "cmi_bias_bits"]
    return pd.DataFrame(lines)[columns]
