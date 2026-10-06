"""
Scorecard for the nfl_predictor.py pipeline.

    .venv/bin/python evaluate.py

Prints diagnostics, then a final line `SCORE: <log loss>` (lower is better).
Needs data/ -- run `.venv/bin/python fetch_data.py` once first.

What it does, in order:
  1. Loads seasons 2010-2024 from data/. The holdout season (2025) is not in data/ at
     all, and any rows from it are dropped again on load as a second guard.
  2. Leakage test: for a handful of checkpoint weeks, rebuilds that week's features
     with every game from that week onward deleted (the week's fixtures are kept --
     matchups are known in advance -- but their scores are scrambled). Features must be
     identical to the full-data build. Any difference means a feature peeks at the
     future: exits 1 without printing a SCORE.
  3. Walk-forward: from the second season on, retrains nfl_predictor.train_and_predict
     on strictly earlier games every RETRAIN_EVERY weeks and predicts the weeks until
     the next retrain. The model never receives the label columns of the games it
     predicts.
  4. Scores all walk-forward predictions with log loss.

Model and feature changes belong in nfl_predictor.py; this file is the referee and
should not change between experiments, or scores stop being comparable.
"""
import contextlib
import io
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import brier_score_loss, log_loss

from nfl_predictor import NFLGamePredictor, train_and_predict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SCHEDULE_FILE = os.path.join(DATA_DIR, 'schedule_2010_2024.csv')
WEEKLY_FILE = os.path.join(DATA_DIR, 'weekly_2010_2024.csv')

HOLDOUT_SEASON = 2025
RETRAIN_EVERY = 4  # weeks; retraining restarts at the first eval week of each season

LABEL_COLS = ['home_win', 'home_score', 'away_score', 'game_id']
# Schedule columns that are only known after kickoff. Scrambled in the leakage test.
OUTCOME_COLS = ['home_score', 'away_score', 'result', 'total', 'overtime']

# (season, week) checkpoints for the leakage test: early, mid, late regular season,
# Wild Card round, Super Bowl, spread across the data range.
LEAKAGE_CHECKPOINTS = [(2011, 2), (2014, 9), (2017, 17), (2019, 18), (2021, 18), (2023, 19), (2024, 22)]


def quiet(fn, *args, **kwargs):
    """Run fn with the pipeline's chatty prints suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def week_key(df):
    """Sortable integer per (season, week) so 'earlier than' is a single comparison."""
    return df['season'] * 100 + df['week']


def load_data():
    schedule = pd.read_csv(SCHEDULE_FILE, low_memory=False)
    weekly = pd.read_csv(WEEKLY_FILE, low_memory=False)

    schedule = schedule[schedule['season'] < HOLDOUT_SEASON].reset_index(drop=True)
    weekly = weekly[weekly['season'] < HOLDOUT_SEASON].reset_index(drop=True)
    assert schedule['season'].max() < HOLDOUT_SEASON and weekly['season'].max() < HOLDOUT_SEASON
    return schedule, weekly


def build_features(schedule, weekly):
    return quiet(NFLGamePredictor().build_dataset, None, weekly, schedule, save_csv=False)


def truncate_at(schedule, weekly, season, week):
    """History as it looked before (season, week) kicked off: every game from that week
    onward deleted, except the week's own fixtures, which keep their matchup info but
    get scrambled outcomes."""
    cutoff = season * 100 + week
    sched_key = week_key(schedule)

    past = schedule[sched_key < cutoff]
    target = schedule[sched_key == cutoff].copy()
    # Flip every winner so any feature that reads the result changes.
    home, away = target['home_score'].copy(), target['away_score'].copy()
    target['home_score'] = away
    target['away_score'] = home + 1
    target['result'] = target['home_score'] - target['away_score']
    target['total'] = target['home_score'] + target['away_score']
    target['overtime'] = 1 - target['overtime'].fillna(0)
    assert set(OUTCOME_COLS) <= set(target.columns)

    return pd.concat([past, target], ignore_index=True), weekly[week_key(weekly) < cutoff]


def check_week(schedule, weekly, expected, feature_cols, season, week):
    """Return None if the week's features survive truncation unchanged, else a reason."""
    if expected.empty:
        return f"{season} wk{week}: no games in full build -- bad checkpoint"

    rebuilt = build_features(*truncate_at(schedule, weekly, season, week))
    rebuilt = rebuilt[(rebuilt['season'] == season) & (rebuilt['week'] == week)]

    exp = expected.set_index('game_id')[feature_cols].sort_index()
    got = rebuilt.set_index('game_id')[feature_cols].sort_index() if not rebuilt.empty else exp.iloc[0:0]
    if not exp.index.equals(got.index):
        return f"{season} wk{week}: games differ (full={len(exp)}, truncated={len(got)})"

    diff = ~np.isclose(exp.values.astype(float), got.values.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True)
    if diff.any():
        bad = [feature_cols[i] for i in np.where(diff.any(axis=0))[0]]
        return f"{season} wk{week}: features changed when future was deleted: {bad}"
    return None


def leakage_test(schedule, weekly, full_features):
    feature_cols = [c for c in full_features.columns if c not in LABEL_COLS]
    results = Parallel(n_jobs=-1)(
        delayed(check_week)(
            schedule, weekly,
            full_features[(full_features['season'] == s) & (full_features['week'] == w)],
            feature_cols, s, w)
        for s, w in LEAKAGE_CHECKPOINTS)

    failures = [r for r in results if r is not None]
    for (s, w), r in zip(LEAKAGE_CHECKPOINTS, results):
        if r is None:
            print(f"  ok  {s} wk{w}")

    if failures:
        print("\n" + "!" * 70)
        print("LEAKAGE TEST FAILED -- a feature is peeking at the future. No score.")
        for f in failures:
            print(f"  FAIL {f}")
        print("!" * 70)
        sys.exit(1)


def retrain_blocks(features):
    """Lists of week keys predicted by one model each: RETRAIN_EVERY consecutive eval
    weeks, never spanning two seasons."""
    keys = features.loc[features['season'] > features['season'].min()].assign(_key=week_key)
    blocks = []
    for _, season_keys in keys.groupby('season')['_key']:
        weeks = sorted(season_keys.unique())
        blocks += [weeks[i:i + RETRAIN_EVERY] for i in range(0, len(weeks), RETRAIN_EVERY)]
    return blocks


def predict_block(features, block):
    key = week_key(features)
    train = features[key < block[0]]
    test = features[key.isin(block)]
    assert week_key(train).max() < week_key(test).min()

    probs = np.asarray(quiet(train_and_predict, train, test.drop(columns=LABEL_COLS)), dtype=float)
    if probs.shape != (len(test),) or not np.all((probs >= 0) & (probs <= 1)):
        raise ValueError(f"train_and_predict returned bad probabilities for block starting {block[0]}: "
                         f"shape {probs.shape}, expected ({len(test)},), values must be in [0, 1]")
    return test[['game_id', 'season', 'week', 'home_score', 'away_score', 'home_win']].assign(
        prob=probs, base_rate=train['home_win'].mean())


def walk_forward(features):
    blocks = retrain_blocks(features)
    n_weeks = sum(len(b) for b in blocks)
    print(f"Walk-forward: {n_weeks} weeks in {len(blocks)} retrains (every {RETRAIN_EVERY} weeks), "
          f"{blocks[0][0] // 100} wk{blocks[0][0] % 100} -> {blocks[-1][-1] // 100} wk{blocks[-1][-1] % 100}")
    results = Parallel(n_jobs=-1)(delayed(predict_block)(features, b) for b in blocks)
    return pd.concat(results, ignore_index=True)


def main():
    start = time.time()
    schedule, weekly = load_data()
    print(f"Holdout season {HOLDOUT_SEASON}: not in data/, never used")
    print(f"Using seasons {schedule['season'].min()}-{schedule['season'].max()}: "
          f"{len(schedule)} games, {len(weekly)} player-week rows")

    features = build_features(schedule, weekly)
    print(f"Built features for {len(features)} games\n")

    print("Leakage test:")
    leakage_test(schedule, weekly, features)
    print()

    preds = walk_forward(features)
    scored = preds[preds['home_score'] != preds['away_score']]  # ties have no winner
    y, p = scored['home_win'].values, scored['prob'].values

    print(f"\nScored games:   {len(scored)} ({len(preds) - len(scored)} ties excluded)")
    for season, g in scored.groupby('season'):
        print(f"  {season}: log loss {log_loss(g['home_win'], g['prob'], labels=[0, 1]):.4f}  "
              f"acc {((g['prob'] > 0.5) == g['home_win']).mean():.3f}  (n={len(g)})")
    print(f"Accuracy:       {((p > 0.5) == y).mean():.3f}  (always pick home: {y.mean():.3f})")
    print(f"Brier:          {brier_score_loss(y, p):.4f}")
    print(f"Baseline:       {log_loss(y, scored['base_rate'].values, labels=[0, 1]):.4f} "
          f"(predict historical home-win rate; coin flip = 0.6931)")
    print(f"Runtime:        {time.time() - start:.0f}s")
    print(f"SCORE: {log_loss(y, p, labels=[0, 1]):.4f}")


if __name__ == '__main__':
    main()
