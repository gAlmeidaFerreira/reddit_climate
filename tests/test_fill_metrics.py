"""Fill diagnostics on small hand-made events tables."""
import numpy as np
import pandas as pd

import config as cfg
from categories import CompositionSettings
from metrics.fill import count_events_per_cell, fill_by_size, plugin_cmi_bias_bits, prior_weight


def _events(own_classes, influencer, new_classes, column=cfg.COL_L):
    return pd.DataFrame({cfg.COL_S: pd.array(own_classes, dtype="Int64"), column: pd.array(influencer, dtype="Int64"),
                         cfg.COL_K: pd.array(new_classes, dtype="Int64")})


def test_repeated_events_are_all_counted():
    cfg.set_classes(3)
    counts = count_events_per_cell(_events([0, 0, 0, 2], [1, 1, 1, 0], [2, 2, 2, pd.NA]), cfg.COL_L)
    assert counts[0, 1, 2] == 3 and counts.sum() == 3                  # the event without k is left out


def test_prior_weight_and_bias_on_known_rows():
    cfg.TAU = 50
    events_per_row = np.array([[50, 0], [0, 0]])
    assert np.isclose(prior_weight(events_per_row), 0.5)              # 50 / (50 + 50)
    assert plugin_cmi_bias_bits(events_per_row) == 0                  # one non-empty category per class


def test_fill_by_size_labels_open_and_closed_size_classes():
    cfg.set_classes(3)
    settings = CompositionSettings("shares", 5, 4, (3, 6), 2)
    empty = _events([], [], [], column=cfg.COL_C)
    assert fill_by_size(empty, settings)["sizes"].tolist() == ["2", "3-5", "6+"]
