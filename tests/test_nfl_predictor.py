"""
Tests for nfl_predictor.py.

Scope is deliberately limited to pure/deterministic logic that doesn't need live
nfl_data_py network access or the ~900MB cached play-by-play data: feature
construction, the validation gate, and the team-name normalization table. These
are also exactly the pieces where real bugs were previously found (see
CLAUDE.md "Known Issues") -- test_build_dataset_is_playoff_and_is_neutral below
is a regression test for the two fixed bugs and would have caught both.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pytest

from nfl_predictor import (
    TEAM_MAPPING,
    NFLGamePredictor,
    NFLSpreadPredictor,
    run_validation_gate,
)


# ---------------------------------------------------------------------------
# TEAM_MAPPING
# ---------------------------------------------------------------------------

def test_team_mapping_covers_all_32_teams():
    assert len(set(TEAM_MAPPING.values())) == 32


def test_team_mapping_normalizes_rams_la_and_lar():
    # CLAUDE.md Conventions: "the Rams are LA in predictions but LAR in
    # nfl_data_py schedules -- normalize when joining, or the game silently drops."
    assert TEAM_MAPPING['LA'] == 'LA'
    assert TEAM_MAPPING['LAR'] == 'LA'
    assert TEAM_MAPPING['Los Angeles Rams'] == 'LA'


# ---------------------------------------------------------------------------
# NFLGamePredictor.create_game_features
# ---------------------------------------------------------------------------

def _team_stats(**overrides):
    base = {
        'passing_yards_pg': 250.0,
        'rushing_yards_pg': 100.0,
        'total_yards_pg': 350.0,
        'points_pg': 24.0,
        'passing_tds_pg': 2.0,
        'interceptions_thrown_pg': 1.0,
        'injury_percentage': 15.0,
    }
    base.update(overrides)
    return base


def test_create_game_features_returns_none_for_unknown_team():
    predictor = NFLGamePredictor()
    team_features = {'AAA': _team_stats()}
    result = predictor.create_game_features('AAA', 'ZZZ', team_features, 2025, 3)
    assert result is None


def test_create_game_features_neutral_site_zeroes_home_field_advantage():
    predictor = NFLGamePredictor()
    team_features = {'AAA': _team_stats(), 'BBB': _team_stats()}

    home_result = predictor.create_game_features('AAA', 'BBB', team_features, 2025, 3, is_neutral=False)
    neutral_result = predictor.create_game_features('AAA', 'BBB', team_features, 2025, 3, is_neutral=True)

    assert home_result['home_field_advantage'] == 2.5
    assert home_result['is_neutral'] == 0
    assert neutral_result['home_field_advantage'] == 0
    assert neutral_result['is_neutral'] == 1


def test_create_game_features_is_playoff_flag_passthrough():
    predictor = NFLGamePredictor()
    team_features = {'AAA': _team_stats(), 'BBB': _team_stats()}

    reg = predictor.create_game_features('AAA', 'BBB', team_features, 2025, 3, is_playoff=False)
    playoff = predictor.create_game_features('AAA', 'BBB', team_features, 2025, 3, is_playoff=True)

    assert reg['is_playoff'] == 0
    assert playoff['is_playoff'] == 1


def test_create_game_features_scoring_advantage_is_home_minus_away():
    predictor = NFLGamePredictor()
    team_features = {
        'AAA': _team_stats(points_pg=30.0),
        'BBB': _team_stats(points_pg=20.0),
    }
    result = predictor.create_game_features('AAA', 'BBB', team_features, 2025, 3)
    assert result['scoring_advantage'] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# NFLGamePredictor.build_dataset -- regression test for the is_playoff /
# is_neutral bugs described in CLAUDE.md "Known Issues"
# ---------------------------------------------------------------------------

def _synthetic_weekly_data():
    rows = []
    for team in ('AAA', 'BBB'):
        for week in (1, 2):
            rows.append({
                'season': 2025, 'week': week, 'recent_team': team,
                'passing_yards': 220, 'rushing_yards': 90, 'completions': 20,
                'passing_tds': 2, 'interceptions': 1, 'rushing_tds': 1,
                'fantasy_points': 20.0,
            })
    return pd.DataFrame(rows)


def _synthetic_schedule_data():
    # All four rows share the same two teams/weekly history; only game_type and
    # location vary, isolating exactly the fields the two fixed bugs depended on.
    return pd.DataFrame([
        {'season': 2025, 'week': 3, 'home_team': 'AAA', 'away_team': 'BBB',
         'game_type': 'REG', 'location': 'Home', 'home_score': 24, 'away_score': 17},
        {'season': 2025, 'week': 4, 'home_team': 'AAA', 'away_team': 'BBB',
         'game_type': 'WC', 'location': 'Home', 'home_score': 24, 'away_score': 17},
        {'season': 2025, 'week': 5, 'home_team': 'AAA', 'away_team': 'BBB',
         'game_type': 'REG', 'location': 'Neutral', 'home_score': 24, 'away_score': 17},
        {'season': 2025, 'week': 6, 'home_team': 'AAA', 'away_team': 'BBB',
         'game_type': 'SB', 'location': 'Neutral', 'home_score': 24, 'away_score': 17},
    ])


def test_build_dataset_is_playoff_and_is_neutral():
    predictor = NFLGamePredictor()
    df = predictor.build_dataset(
        pbp_data=None,
        weekly_data=_synthetic_weekly_data(),
        schedule_data=_synthetic_schedule_data(),
        save_csv=False,
    )
    by_week = df.set_index('week')

    # Week 3: regular season, home site -> not playoff, not neutral
    assert by_week.loc[3, 'is_playoff'] == 0
    assert by_week.loc[3, 'is_neutral'] == 0
    assert by_week.loc[3, 'home_field_advantage'] == 2.5

    # Week 4: Wild Card round, home site -> playoff, not neutral
    assert by_week.loc[4, 'is_playoff'] == 1
    assert by_week.loc[4, 'is_neutral'] == 0

    # Week 5: regular season, neutral (e.g. international) site -> not playoff, neutral
    assert by_week.loc[5, 'is_playoff'] == 0
    assert by_week.loc[5, 'is_neutral'] == 1
    assert by_week.loc[5, 'home_field_advantage'] == 0

    # Week 6: Super Bowl -> playoff AND neutral
    assert by_week.loc[6, 'is_playoff'] == 1
    assert by_week.loc[6, 'is_neutral'] == 1
    assert by_week.loc[6, 'home_field_advantage'] == 0


# ---------------------------------------------------------------------------
# NFLGamePredictor._calculate_injury_percentage
# ---------------------------------------------------------------------------

def test_injury_percentage_defaults_to_league_average_without_signal():
    predictor = NFLGamePredictor()
    team_data = pd.DataFrame({'some_other_column': [1, 2, 3]})
    assert predictor._calculate_injury_percentage(team_data) == 15.0


def test_injury_percentage_uses_real_injury_column_when_present():
    predictor = NFLGamePredictor()
    team_data = pd.DataFrame({'injuries': [10.0, 20.0, 30.0]})
    assert predictor._calculate_injury_percentage(team_data) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# NFLSpreadPredictor.prepare_spread_data
# ---------------------------------------------------------------------------

def test_prepare_spread_data_computes_point_spread():
    spread_predictor = NFLSpreadPredictor()
    df = pd.DataFrame({'home_score': [24, 10], 'away_score': [17, 20]})
    result = spread_predictor.prepare_spread_data(df)
    assert list(result['point_spread']) == [7, -10]


# ---------------------------------------------------------------------------
# run_validation_gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "accuracy, brier, expected",
    [
        (0.60, 0.20, True),    # comfortably clears both bars
        (0.55, 0.25, True),    # exactly at the boundary (>=, <=) -> still PASS
        (0.50, 0.20, False),   # accuracy below floor
        (0.60, 0.30, False),   # brier above ceiling
        (0.50, 0.30, False),   # fails both
    ],
)
def test_run_validation_gate(accuracy, brier, expected):
    metrics = {'accuracy': accuracy, 'brier_score': brier, 'log_loss': 0.69, 'accuracy_std': 0.01}
    assert run_validation_gate(metrics) is expected
