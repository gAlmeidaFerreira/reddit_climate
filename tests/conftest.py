"""
Shared fixtures. Run the tests from the project folder with:  python -m pytest tests
"""
import sys
import warnings
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")                    # placeholder-cut warnings of config.py
    import config as cfg

from pipeline.items import build_items
from pipeline.own_opinion import add_new_opinion_class, add_own_opinion
from pipeline.replies import attach_parents
from synthetic_data import brute_force_answers, make_synthetic_raw_tables


@pytest.fixture(autouse=True)
def restore_config():
    """Every test may change config values; they are put back afterwards."""
    saved = {name: value for name, value in vars(cfg).items() if name.isupper()}
    yield
    for name, value in saved.items():
        setattr(cfg, name, value)
    cfg.set_classes(saved["N_CLASSES"])


@pytest.fixture(params=[3, 2], ids=["3 classes", "2 classes"])
def number_of_classes(request):
    cfg.set_classes(request.param)
    return request.param


@pytest.fixture(scope="session")
def raw_tables():
    return make_synthetic_raw_tables()


@pytest.fixture(scope="session")
def brute_force(raw_tables):
    return brute_force_answers(*raw_tables)


@pytest.fixture
def items(raw_tables):
    return build_items(*raw_tables)


@pytest.fixture
def replies_before_groups(items, number_of_classes):
    """Replies with parent, own opinion and new class, for the current number of classes."""
    return add_new_opinion_class(add_own_opinion(attach_parents(items), items))
