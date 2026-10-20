"""
Stage 0. Load the raw comment and submission tables of the released dataset.
"""
import pandas as pd

import config as cfg

CACHED_COMMENTS_FILE = "threads.parquet"
CACHED_SUBMISSIONS_FILE = "submissions.parquet"


def _read_all_csv_files(folder):
    """Every CSV under `folder` (recursively), concatenated; all columns read as text so nothing is guessed."""
    csv_files = sorted(folder.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no CSV files under {folder}; extract the zip into data/raw/ "
                                f"or fix the paths in config.py")
    return pd.concat((pd.read_csv(csv_file, dtype=str) for csv_file in csv_files), ignore_index=True)


def load_raw_tables(use_cache=True):
    """Return (raw_comments, raw_submissions), one row per comment and per submission, with the raw column names.

    Uses data/cache/{threads,submissions}.parquet when both exist; otherwise reads every CSV, converts the score
    to numbers ("NI" -> NaN) and writes the cache. The cache is never checked against the CSVs: after changing
    the raw data, call with use_cache=False (or delete the cache).
    """
    cached_comments = cfg.DATA_CACHE / CACHED_COMMENTS_FILE
    cached_submissions = cfg.DATA_CACHE / CACHED_SUBMISSIONS_FILE
    if use_cache and cached_comments.exists() and cached_submissions.exists():
        return pd.read_parquet(cached_comments), pd.read_parquet(cached_submissions)

    raw_comments = _read_all_csv_files(cfg.THREADS_DIR)
    raw_submissions = _read_all_csv_files(cfg.SUBMISSIONS_DIR)
    cfg.DATA_CACHE.mkdir(parents=True, exist_ok=True)
    for raw_table, cache_path in ((raw_comments, cached_comments), (raw_submissions, cached_submissions)):
        raw_table[cfg.RAW_SCORE] = pd.to_numeric(raw_table[cfg.RAW_SCORE], errors="coerce")   # "NI" -> NaN
        raw_table.to_parquet(cache_path)
    return raw_comments, raw_submissions
