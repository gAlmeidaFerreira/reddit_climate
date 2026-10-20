"""Opinion classes, the composition registry and the rounding of shares."""
from math import comb

import numpy as np
import pytest

import config as cfg
from categories import (CompositionSettings, assign_composition_cells, composition_registry, influencer_categories,
                        score_to_class)
from synthetic_data import brute_force_class


def test_score_to_class_matches_brute_force(number_of_classes):
    scores = np.concatenate([np.linspace(0, 1, 101), list(cfg.CUTS)])
    assert score_to_class(scores, cfg.CUTS).tolist() == [brute_force_class(score) for score in scores]


def test_score_to_class_keeps_missing_scores_missing():
    assert np.isnan(score_to_class([np.nan], cfg.CUTS)[0])


@pytest.mark.parametrize("max_group_size", [3, 5])
def test_counts_registry_has_one_cell_per_composition(number_of_classes, max_group_size):
    settings = CompositionSettings("counts", max_group_size, 4, (), cfg.MIN_MEMBERS)
    registry = composition_registry(settings)
    expected_cells = sum(comb(size + number_of_classes - 1, number_of_classes - 1)
                         for size in range(cfg.MIN_MEMBERS, max_group_size + 1))
    assert len(registry) == expected_cells
    assert registry["label"].is_unique
    assert (registry["cell"] == np.arange(len(registry))).all()
    assert all(sum(parts) == size for parts, size in zip(registry["parts"], registry["min_size"]))


def test_shares_registry_repeats_the_grid_for_every_size_class():
    settings = CompositionSettings("shares", 5, 4, (3, 5), 2)
    registry = composition_registry(settings)
    assert registry["size_class"].nunique() == 3                       # sizes 2 | 3-4 | 5+
    assert len(registry) == 3 * comb(4 + 2, 2)
    assert registry["label"].iloc[-1].startswith("n=5+")


def test_shares_rounding_known_cases_and_reproducible():
    cfg.set_classes(3)
    settings = CompositionSettings("shares", 5, 4, (), 2)
    class_counts = np.array([[1, 1, 1], [2, 1, 0], [3, 0, 0], [1, 1, 0], [2, 2, 2]])
    first, second = (assign_composition_cells(class_counts, settings, seed=1) for _ in range(2))
    assert (first == second).all(), "seeded rounding must be reproducible"
    parts = composition_registry(settings).set_index("cell")["parts"]
    rounded = [parts[int(cell)] for cell in first]
    assert rounded[1] == (3, 1, 0) and rounded[2] == (4, 0, 0)
    assert all(sum(group_parts) == 4 for group_parts in rounded)


def test_counts_mode_leaves_small_and_large_groups_without_a_cell():
    cfg.set_classes(3)
    settings = CompositionSettings("counts", 3, 4, (), 2)
    cells = assign_composition_cells(np.array([[1, 0, 0], [1, 1, 0], [2, 1, 1]]), settings)
    assert np.isnan(cells[0]) and not np.isnan(cells[1]) and np.isnan(cells[2])


def test_influencer_categories_rejects_unknown_columns():
    with pytest.raises(ValueError):
        influencer_categories("unknown")
