"""
Synthetic raw tables in the same format as the released CSVs, and brute-force answers computed straight from
them. The brute force shares NO code with the pipeline: it reads the raw columns, never items or replies.
"""
import numpy as np
import pandas as pd

import config as cfg

BASE_TIME = pd.Timestamp("2021-03-01")


def make_synthetic_raw_tables(seed=1, number_of_threads=12, comments_per_thread=40, number_of_users=7):
    """Raw (comments, submissions) designed to stress the pipeline:
    few users (many self-replies and repeated authors), replies 0-5 seconds after their parent (many ties,
    including a reply at the same second as its parent), NI scores (20% of posts, 25% of comments),
    35% of comments reply to an earlier comment, the rest to the post, threads in two subreddits."""
    random_generator = np.random.default_rng(seed)
    comment_rows, submission_rows = [], []
    for thread_index in range(number_of_threads):
        subreddit, submission_id = f"r/{thread_index % 2}", f"s{thread_index}"
        submission_second = int(random_generator.integers(0, 20))
        submission_rows.append({
            cfg.RAW_SUBMISSION_ID: submission_id, cfg.RAW_SUBREDDIT_ID: subreddit,
            cfg.RAW_AUTHOR_ID: f"u{random_generator.integers(number_of_users)}",
            cfg.RAW_TIME: BASE_TIME + pd.Timedelta(seconds=submission_second),
            cfg.RAW_SCORE: random_generator.random() if random_generator.random() > .2 else "NI"})
        comments_of_thread = []                                        # (comment id, second)
        for comment_index in range(comments_per_thread):
            if comments_of_thread and random_generator.random() < .35:
                parent_comment_id, parent_second = comments_of_thread[random_generator.integers(len(comments_of_thread))]
                parent_reference = cfg.PARENT_PREFIX_COMMENT + parent_comment_id
            else:
                parent_second, parent_reference = submission_second, cfg.PARENT_PREFIX_SUBMISSION + submission_id
            comment_second = max(parent_second, submission_second) + int(random_generator.integers(0, 6))
            comment_id = f"c{thread_index}_{comment_index}"
            comments_of_thread.append((comment_id, comment_second))
            comment_rows.append({
                cfg.RAW_COMMENT_ID: comment_id, cfg.RAW_SUBMISSION_ID: cfg.PARENT_PREFIX_SUBMISSION + submission_id,
                cfg.RAW_PARENT_ID: parent_reference, cfg.RAW_SUBREDDIT_ID: subreddit,
                cfg.RAW_AUTHOR_ID: f"u{random_generator.integers(number_of_users)}",
                cfg.RAW_TIME: BASE_TIME + pd.Timedelta(seconds=comment_second),
                cfg.RAW_SCORE: random_generator.random() if random_generator.random() > .25 else "NI"})
    return pd.DataFrame(comment_rows), pd.DataFrame(submission_rows)


def brute_force_class(score):
    """Class of one raw score, written without categories.score_to_class."""
    cuts = cfg.CUTS
    if score < cuts[0]:
        return 0
    return 1 if (len(cuts) == 1 or score <= cuts[1]) else 2


def _with_time_and_score(raw_table):
    return raw_table.assign(time=pd.to_datetime(raw_table[cfg.RAW_TIME]),
                            score=pd.to_numeric(raw_table[cfg.RAW_SCORE], errors="coerce"))


def brute_force_answers(raw_comments, raw_submissions):
    """For every scored comment with an author, keyed by (subreddit, comment id):
        member_scores        scores of its group (parent if scored and by another author + earlier scored siblings
                             by other authors)
        previous_item_time   time of the author's last scored item strictly before the parent (NaT if none)"""
    comments, submissions = _with_time_and_score(raw_comments), _with_time_and_score(raw_submissions)
    answers = {}
    for _, comment in comments[comments["score"].notna() & comments[cfg.RAW_AUTHOR_ID].notna()].iterrows():
        subreddit, author = comment[cfg.RAW_SUBREDDIT_ID], comment[cfg.RAW_AUTHOR_ID]
        same_subreddit = comments[cfg.RAW_SUBREDDIT_ID] == subreddit
        siblings = comments[same_subreddit & (comments[cfg.RAW_PARENT_ID] == comment[cfg.RAW_PARENT_ID])
                            & (comments["time"] < comment["time"]) & (comments[cfg.RAW_AUTHOR_ID] != author)
                            & comments["score"].notna()]
        member_scores = list(siblings["score"])

        parent_reference = comment[cfg.RAW_PARENT_ID]
        parent_id = parent_reference[len(cfg.PARENT_PREFIX_SUBMISSION):]
        if parent_reference.startswith(cfg.PARENT_PREFIX_SUBMISSION):
            parent = submissions[(submissions[cfg.RAW_SUBMISSION_ID] == parent_id)
                                 & (submissions[cfg.RAW_SUBREDDIT_ID] == subreddit)]
        else:
            parent = comments[(comments[cfg.RAW_COMMENT_ID] == parent_id) & same_subreddit]
        if len(parent) and pd.notna(parent["score"].iloc[0]) and parent[cfg.RAW_AUTHOR_ID].iloc[0] != author:
            member_scores.append(parent["score"].iloc[0])

        previous_item_time = pd.NaT
        if len(parent):
            own_items = pd.concat([comments.loc[comments[cfg.RAW_AUTHOR_ID] == author, ["time", "score"]],
                                   submissions.loc[submissions[cfg.RAW_AUTHOR_ID] == author, ["time", "score"]]])
            earlier_scored = own_items[own_items["score"].notna() & (own_items["time"] < parent["time"].iloc[0])]
            if len(earlier_scored):
                previous_item_time = earlier_scored["time"].max()
        answers[(subreddit, comment[cfg.RAW_COMMENT_ID])] = {"member_scores": member_scores,
                                                             "previous_item_time": previous_item_time}
    return answers


def brute_force_pair_shares(member_classes):
    """(opposed share, mild share) of a group by looping over every pair of members."""
    pairs = [(a, b) for i, a in enumerate(member_classes) for b in member_classes[i + 1:]]
    if not pairs:
        return np.nan, np.nan
    opposed = sum(1 for a, b in pairs if abs(a - b) == cfg.N_CLASSES - 1)
    mild = sum(1 for a, b in pairs if 0 < abs(a - b) < cfg.N_CLASSES - 1)
    return opposed / len(pairs), mild / len(pairs)
