"""
Scorecard for the nfl_predictor.py pipeline.

    .venv/bin/python evaluate.py

Prints diagnostics, then a final line `SCORE: <log loss>` (lower is better).

What it does, in order:
  1. Loads the cached schedule + weekly stats and immediately drops the most recent
     season (the holdout). Nothing below ever sees it.
  2. Leakage test: for a handful of checkpoint weeks, rebuilds that week's features
     with every game from that week onward deleted (the week's fixtures are kept --
     matchups are known in advance -- but their scores are scrambled). Features must be
     identical to the full-data build. Any difference means a feature peeks at the
     future: exits 1 without printing a SCORE.
  3. Walk-forward: for every week after the first season, retrains the full pipeline
     (feature selection + ensemble) on strictly earlier games and predicts that week.
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

from nfl_predictor import NFLGamePredictor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Week22', 'nfl_data')
SCHEDULE_FILE = os.path.join(DATA_DIR, 'schedule_data_2020_2025.csv')
WEEKLY_FILE = os.path.join(DATA_DIR, 'weekly_data_2020_2025.csv')

LABEL_COLS = ['home_win', 'home_score', 'away_score', 'game_id']
# Schedule columns that are only known after kickoff. Scrambled in the leakage test.
OUTCOME_COLS = ['home_score', 'away_score', 'result', 'total', 'overtime']

# (season, week) checkpoints for the leakage test: early, mid, late regular season,
# Wild Card round, Super Bowl.
LEAKAGE_CHECKPOINTS = [(2021, 2), (2022, 9), (2023, 18), (2024, 19), (2024, 22)]


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

    holdout = int(schedule['season'].max())
    schedule = schedule[schedule['season'] < holdout].reset_index(drop=True)
    weekly = weekly[weekly['season'] < holdout].reset_index(drop=True)
    assert schedule['season'].max() < holdout and weekly['season'].max() < holdout
    return schedule, weekly, holdout


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


def leakage_test(schedule, weekly, full_features):
    feature_cols = [c for c in full_features.columns if c not in LABEL_COLS]
    failures = []

    for season, week in LEAKAGE_CHECKPOINTS:
        expected = full_features[(full_features['season'] == season) & (full_features['week'] == week)]
        if expected.empty:
            failures.append(f"{season} wk{week}: no games in full build -- bad checkpoint")
            continue

        t_sched, t_weekly = truncate_at(schedule, weekly, season, week)
        rebuilt = build_features(t_sched, t_weekly)
        rebuilt = rebuilt[(rebuilt['season'] == season) & (rebuilt['week'] == week)]

        exp = expected.set_index('game_id')[feature_cols].sort_index()
        got = rebuilt.set_index('game_id')[feature_cols].sort_index() if not rebuilt.empty else exp.iloc[0:0]
        if not exp.index.equals(got.index):
            failures.append(f"{season} wk{week}: games differ "
                            f"(full={len(exp)}, truncated={len(got)})")
            continue

        diff = ~np.isclose(exp.values.astype(float), got.values.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True)
        if diff.any():
            bad = [feature_cols[i] for i in np.where(diff.any(axis=0))[0]]
            failures.append(f"{season} wk{week}: features changed when future was deleted: {bad}")
        else:
            print(f"  ok  {season} wk{week:<2}  ({len(exp)} games, {len(feature_cols)} features)")

    if failures:
        print("\n" + "!" * 70)
        print("LEAKAGE TEST FAILED -- a feature is peeking at the future. No score.")
        for f in failures:
            print(f"  FAIL {f}")
        print("!" * 70)
        sys.exit(1)


def fit_predict(train, test):
    """Train the full pipeline on `train` only, return P(home win) for `test`."""
    predictor = NFLGamePredictor()
    quiet(predictor.select_features, train)
    quiet(predictor.create_ensemble_model, train)
    cols = predictor.best_features
    # Fill gaps with training means, never test-batch statistics.
    X_test = test[cols].fillna(train[cols].mean())
    return predictor.final_model.predict_proba(X_test)[:, 1]


def walk_forward(features):
    features = features.assign(_key=week_key(features))
    first_season = features['season'].min()
    eval_keys = sorted(features.loc[features['season'] > first_season, '_key'].unique())

    def one_week(key):
        train = features[features['_key'] < key]
        test = features[features['_key'] == key]
        assert train['_key'].max() < key
        return test[['game_id', 'season', 'week', 'home_score', 'away_score', 'home_win']].assign(
            prob=fit_predict(train.drop(columns='_key'), test.drop(columns='_key')),
            base_rate=train['home_win'].mean(),
        )

    print(f"Walk-forward: {len(eval_keys)} weeks, "
          f"{eval_keys[0] // 100} wk{eval_keys[0] % 100} -> {eval_keys[-1] // 100} wk{eval_keys[-1] % 100}")
    results = Parallel(n_jobs=-1)(delayed(one_week)(k) for k in eval_keys)
    return pd.concat(results, ignore_index=True)


def main():
    start = time.time()
    schedule, weekly, holdout = load_data()
    print(f"Holdout season {holdout}: dropped at load, never used")
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
    print(f"Accuracy:       {((p > 0.5) == y).mean():.3f}")
    print(f"Brier:          {brier_score_loss(y, p):.4f}")
    print(f"Baseline:       {log_loss(y, scored['base_rate'].values, labels=[0, 1]):.4f} "
          f"(predict historical home-win rate; coin flip = 0.6931)")
    print(f"Runtime:        {time.time() - start:.0f}s")
    print(f"SCORE: {log_loss(y, p, labels=[0, 1]):.4f}")


if __name__ == '__main__':
    main()
