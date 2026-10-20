"""
Opinion classes and influencer categories, shared by the pipeline, the metrics and tensors.py.

    score_to_class               raw stance score in [0, 1] -> opinion class (0 against, 1 neutral, 2 favour)
    class_count_columns          names of the per-class member counts of a group
    CompositionSettings          how the composition c of a group is turned into a cell
    composition_registry         every possible composition cell c, with its label
    assign_composition_cells     class counts of groups -> composition cell c
    influencer_categories        number of categories and labels of an influencer column (l, g, h, lean, c)

Nothing here reads replies or threads: everything works on scores, class counts and config.py.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as cfg


# =======================================================================================
# Opinion classes
# =======================================================================================
def score_to_class(scores, cuts):
    """Raw scores -> class 0 .. len(cuts); missing scores stay NaN, so the result is a float array.

    With cuts (c0, c1):  score < c0 -> 0 ;  c0 <= score <= c1 -> 1 ;  score > c1 -> 2.
    Both cut values belong to the middle class. With one cut: score < c0 -> 0, otherwise 1.
    """
    scores = np.asarray(scores, dtype=float)
    classes = (scores >= cuts[0]).astype(float)
    for upper_cut in cuts[1:]:
        classes = classes + (scores > upper_cut)
    return np.where(np.isnan(scores), np.nan, classes)


def class_count_columns():
    """Names of the per-class member counts of a group, one per opinion class (read at call time, so they
    follow cfg.set_classes)."""
    return [f"group_count_class_{class_index}" for class_index in range(cfg.N_CLASSES)]


# =======================================================================================
# Composition cells
# =======================================================================================
@dataclass(frozen=True)
class CompositionSettings:
    """How the composition of a group (its class counts) becomes one cell c.

    mode              "counts": one cell per exact composition, for groups of min_members .. max_group_size
                      "shares": class shares rounded to multiples of 1 / share_grid, no size cap
    max_group_size    counts mode: larger groups get no cell
    share_grid        shares mode: number of units the shares are rounded to
    size_class_edges  shares mode: group-size classes, e.g. (3, 6, 11) -> sizes 2 | 3-5 | 6-10 | 11+
    min_members       smallest group that gets a cell
    """
    mode: str
    max_group_size: int
    share_grid: int
    size_class_edges: tuple
    min_members: int

    @classmethod
    def from_config(cls):
        return cls(mode=cfg.COMP_MODE, max_group_size=cfg.COMP_MAX_SIZE, share_grid=cfg.COMP_GRID,
                   size_class_edges=tuple(cfg.COMP_SIZE_EDGES), min_members=cfg.MIN_MEMBERS)

    def __post_init__(self):
        if self.mode not in ("counts", "shares"):
            raise ValueError(f"mode must be 'counts' or 'shares', got {self.mode!r}")


def _settings_or_config(settings):
    return CompositionSettings.from_config() if settings is None else settings


def _all_compositions(total, number_of_classes):
    """Every tuple of `number_of_classes` non-negative integers that adds up to `total`, e.g.
    total 2, 3 classes: (0,0,2) (0,1,1) (0,2,0) (1,0,1) (1,1,0) (2,0,0)."""
    if number_of_classes == 1:
        yield (total,)
        return
    for first_part in range(total + 1):
        for remaining_parts in _all_compositions(total - first_part, number_of_classes - 1):
            yield (first_part,) + remaining_parts


def _group_size_classes(settings):
    """Shares mode: [(min_size, max_size), ...]; max_size is None for the last, open class."""
    edges = [settings.min_members] + list(settings.size_class_edges)
    closed_classes = [(lower, upper - 1) for lower, upper in zip(edges, edges[1:])]
    return closed_classes + [(edges[-1], None)]


def _size_tag(min_size, max_size):
    if max_size is None:
        return f"n={min_size}+"
    if max_size == min_size:
        return f"n={min_size}"
    return f"n={min_size}-{max_size}"


def composition_registry(settings=None):
    """One row per possible composition cell.

    Columns: cell (the integer index used in the tensors), size_class, min_size, max_size (missing for an open
    size class), parts (tuple of units per class, in class order) and label, e.g. "n=3: 1-0-2".
    'counts': one cell per composition of n members, n = min_members .. max_group_size (size_class = n - min_members).
    'shares': one cell per way to split share_grid units among the classes, repeated for every group-size class.
    """
    settings = _settings_or_config(settings)
    number_of_classes = cfg.N_CLASSES
    rows = []
    if settings.mode == "counts":
        for group_size in range(settings.min_members, settings.max_group_size + 1):
            for parts in _all_compositions(group_size, number_of_classes):
                label = f"n={group_size}: " + "-".join(map(str, parts))
                rows.append((group_size - settings.min_members, group_size, group_size, parts, label))
    else:
        for size_class, (min_size, max_size) in enumerate(_group_size_classes(settings)):
            tag = _size_tag(min_size, max_size)
            for parts in _all_compositions(settings.share_grid, number_of_classes):
                label = f"{tag}: " + "-".join(map(str, parts)) + f" (/{settings.share_grid})"
                rows.append((size_class, min_size, max_size, parts, label))
    registry = pd.DataFrame(rows, columns=["size_class", "min_size", "max_size", "parts", "label"])
    registry["min_size"] = registry["min_size"].astype("Int64")
    registry["max_size"] = pd.array(registry["max_size"].tolist(), dtype="Int64")
    registry.insert(0, "cell", np.arange(len(registry)))
    return registry


def _round_shares_to_grid(class_counts, share_grid, seed):
    """Largest-remainder rounding of the class shares to multiples of 1 / share_grid; the parts always add up to
    share_grid. Exact ties between remainders are broken at random (seeded, so a run is reproducible)."""
    group_sizes = class_counts.sum(axis=1)[:, None]
    scaled_counts = class_counts * share_grid
    floor_parts, remainders = scaled_counts // group_sizes, scaled_counts % group_sizes     # exact integer arithmetic
    missing_units = share_grid - floor_parts.sum(axis=1)
    random_generator = np.random.default_rng(cfg.SEED if seed is None else seed)
    tie_breaker = random_generator.random(remainders.shape)
    remainder_rank = np.argsort(np.argsort(-(remainders + tie_breaker), axis=1), axis=1)    # 0 = largest remainder
    return floor_parts + (remainder_rank < missing_units[:, None])


def assign_composition_cells(class_counts, settings=None, seed=None):
    """Class counts of groups (array: groups x classes) -> cell index (float; NaN for a group without a cell).

    'counts': the cell is the exact composition; groups smaller than min_members or larger than max_group_size get NaN.
    'shares': the cell is the rounded shares plus the group-size class; groups smaller than min_members get NaN.
    """
    settings = _settings_or_config(settings)
    class_counts = np.asarray(class_counts, dtype=np.int64)
    group_sizes = class_counts.sum(axis=1)
    if settings.mode == "counts":
        has_cell = (group_sizes >= settings.min_members) & (group_sizes <= settings.max_group_size)
        parts = class_counts[has_cell]
        size_classes = group_sizes[has_cell] - settings.min_members
    else:
        has_cell = group_sizes >= settings.min_members
        parts = _round_shares_to_grid(class_counts[has_cell], settings.share_grid, seed)
        size_edges = np.array(settings.size_class_edges, dtype=np.int64)
        size_classes = np.searchsorted(size_edges, group_sizes[has_cell], side="right")

    registry = composition_registry(settings)
    cell_by_parts_and_size = {(tuple(cell_parts), int(size_class)): int(cell)
                              for cell_parts, size_class, cell in zip(registry["parts"], registry["size_class"], registry["cell"])}
    found_cells = np.array([cell_by_parts_and_size.get((tuple(group_parts), int(size_class)), -1)
                            for group_parts, size_class in zip(parts.tolist(), size_classes.tolist())], dtype=np.int64)
    assert (found_cells >= 0).all(), "a group fell outside the composition registry"

    cells = np.full(len(group_sizes), np.nan)
    cells[has_cell] = found_cells
    return cells


# =======================================================================================
# Influencer categories
# =======================================================================================
def influencer_categories(influencer_column, settings=None):
    """(number of categories, labels) of an influencer column of the events tables."""
    if influencer_column == cfg.COL_L:
        return cfg.N_CLASSES, list(cfg.CLASS_LABELS)
    if influencer_column == cfg.COL_G:
        return cfg.N_G, list(cfg.G_LABELS)
    if influencer_column == cfg.COL_H:
        return cfg.N_H, list(cfg.H_LABELS)
    if influencer_column == cfg.COL_LEAN:
        return cfg.N_CLASSES, list(cfg.MEAN_LABELS)
    if influencer_column == cfg.COL_C:
        registry = composition_registry(settings)
        return len(registry), registry["label"].tolist()
    raise ValueError(f"influencer column must be one of l, g, h, lean, c; got {influencer_column!r}")
