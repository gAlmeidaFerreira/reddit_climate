"""
Loading-completeness check for filtered_subs_and_threads_anonymized.zip

Answers three separate questions:
  A. Is the zip intact, and what exactly is inside it?
  B. Was everything extracted, and did every extracted CSV reach the DataFrames
     (raw line counts vs loaded rows, per folder)?
  C. If loading is complete, how much was already missing from the release itself
     (dangling parents, orphan comments, gaps in the anonymised ID numbering)?

Assumes your loader added two columns, as in your notebook:
  subreddit  = folder prefix, e.g. "subreddit2"
  thread_id  = file name without "_scored"

Usage:
    from check_loading import check_loading
    res = check_loading("filtered_subs_and_threads_anonymized.zip",
                        "data/filtered_subs_and_threads_anonymized", threads, submissions)
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _count_data_rows(path):
    with open(path, "rb") as f:
        n = sum(1 for _ in f)
    return max(n - 1, 0)  # minus header


def check_loading(zip_path, root, threads, submissions, verbose=True):
    zip_path, root = Path(zip_path), Path(root)
    base = root.parent                      # directory the zip was extracted into
    out = {}

    # ---- A. zip integrity and inventory ------------------------------------
    with zipfile.ZipFile(zip_path) as z:
        out["first_corrupt_member"] = z.testzip()          # None means all CRCs are fine
        infos = [i for i in z.infolist() if not i.is_dir()]
    inv = pd.DataFrame({"name": [i.filename for i in infos], "size": [i.file_size for i in infos]})
    inv["is_junk"] = inv["name"].str.contains(r"(?:^|/)(?:__MACOSX|\.DS_Store|\._)", regex=True)
    inv["ext"] = inv["name"].str.extract(r"(\.[A-Za-z0-9]+(?:\.gz)?)$")[0].str.lower().fillna("(none)")
    inv["kind"] = np.where(inv["name"].str.contains("/threads/"), "threads",
                           np.where(inv["name"].str.contains("/submissions/"), "submissions", "other"))
    good = inv[~inv["is_junk"]]
    out["zip_inventory"] = good.groupby(["kind", "ext"]).size().rename("n_files")
    out["zip_junk_files"] = int(inv["is_junk"].sum())
    out["zip_zero_byte_files"] = int((good["size"] == 0).sum())

    # ---- B1. extracted on disk? -------------------------------------------
    disk = {p.relative_to(base).as_posix(): p.stat().st_size for p in root.rglob("*") if p.is_file()}
    zip_sizes = dict(zip(good["name"], good["size"]))
    out["zip_files_missing_on_disk"] = sorted(set(zip_sizes) - set(disk))[:20]
    out["n_zip_files_missing_on_disk"] = len(set(zip_sizes) - set(disk))
    out["n_size_mismatch"] = sum(1 for k, v in zip_sizes.items() if k in disk and disk[k] != v)

    # ---- B2. raw rows vs loaded rows, per kind and folder ------------------
    csvs = sorted(root.rglob("*.csv"))
    raw = pd.DataFrame(
        [(p.relative_to(root).parts[0], p.parent.name.split("_")[0], p.stem, _count_data_rows(p))
         for p in csvs],
        columns=["kind", "subreddit", "file", "raw_rows"])
    out["n_csv_found_by_rglob"] = len(csvs)
    out["n_csv_in_zip"] = int((good["ext"] == ".csv").sum())

    raw_t = raw[raw["kind"] == "threads"].groupby("subreddit").agg(
        thread_files=("file", "size"), raw_rows=("raw_rows", "sum"),
        empty_files=("raw_rows", lambda s: int((s == 0).sum())))
    loaded_t = threads.groupby("subreddit").agg(
        loaded_rows=("comm_id", "size"), loaded_threads=("thread_id", "nunique"))
    cmp_t = raw_t.join(loaded_t, how="outer")
    cmp_t["rows_lost"] = cmp_t["raw_rows"] - cmp_t["loaded_rows"]
    cmp_t["files_lost"] = cmp_t["thread_files"] - cmp_t["loaded_threads"]
    out["threads_raw_vs_loaded"] = cmp_t

    raw_s = raw[raw["kind"] == "submissions"].groupby("subreddit").agg(
        sub_files=("file", "size"), raw_rows=("raw_rows", "sum"))
    loaded_s = submissions.groupby("subreddit").size().rename("loaded_rows")
    cmp_s = raw_s.join(loaded_s, how="outer")
    cmp_s["rows_lost"] = cmp_s["raw_rows"] - cmp_s["loaded_rows"]
    out["submissions_raw_vs_loaded"] = cmp_s

    # ---- C. what is missing from the release itself ------------------------
    sid = threads["sub_id"].str.replace("t3_", "", regex=False)
    tkey = threads["subreddit_id"] + "|" + sid
    skey = submissions["subreddit_id"] + "|" + submissions["sub_id"]
    out["comments_without_submission"] = int((~tkey.isin(set(skey))).sum())
    nothr = ~skey.isin(set(tkey))
    out["submissions_without_thread_file"] = (
        submissions.loc[nothr].groupby("subreddit_id").size().rename("n").to_frame()
        .join(submissions.groupby("subreddit_id").size().rename("of_total")))
    out["files_with_more_than_one_sub_id"] = int(
        (threads.groupby(["subreddit", "thread_id"])["sub_id"].nunique() > 1).sum())
    out["duplicate_comm_ids"] = int(threads.duplicated(["subreddit_id", "comm_id"]).sum())

    pid = threads["parent_id"]
    is_c = pid.str.startswith("t1_")
    miss = is_c & ~pid.str[3:].isin(set(threads["comm_id"]))
    out["dangling_parent_share"] = round(float(miss.sum() / is_c.sum()), 3)
    out["dangling_parents_distinct"] = (
        threads.loc[miss].assign(p=pid[miss].str[3:]).groupby("subreddit_id")["p"].nunique()
        .rename("distinct_missing_parent_comments"))

    num = pd.to_numeric(threads["comm_id"].str.extract(r"(\d+)$")[0], errors="coerce")
    gap = threads.assign(n=num).groupby("subreddit_id")["n"].agg(["min", "max", "count"])
    gap["id_span"] = gap["max"] - gap["min"] + 1
    gap["unseen_ids_in_span"] = gap["id_span"] - gap["count"]
    out["comm_id_numbering"] = gap

    # ---- verdict -----------------------------------------------------------
    lost = int(cmp_t["rows_lost"].fillna(0).abs().sum() + cmp_s["rows_lost"].fillna(0).abs().sum())
    ok_zip = out["first_corrupt_member"] is None and out["n_zip_files_missing_on_disk"] == 0
    out["verdict"] = (
        f"Zip intact and fully extracted: {ok_zip}. "
        f"CSV files in zip vs found/read: {out['n_csv_in_zip']} vs {out['n_csv_found_by_rglob']}. "
        f"Total rows lost between raw files and DataFrames: {lost}. "
        + ("Loading is complete: any gap with Table 1 comes from the release, not from your code."
           if ok_zip and lost == 0 and out["n_csv_in_zip"] == out["n_csv_found_by_rglob"]
           else "Loading is NOT complete: fix the loader before analysing."))

    if verbose:
        for k, v in out.items():
            print(f"\n=== {k} ===")
            print(v.to_string() if hasattr(v, "to_string") else v)
    return out