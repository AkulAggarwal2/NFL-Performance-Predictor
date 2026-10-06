"""
Download the data evaluate.py scores against into data/ (gitignored).

    .venv/bin/python fetch_data.py

Seasons 2010-2024 only. The 2025 holdout season is deliberately never downloaded here.
"""
import os

import nfl_data_py as nfl

SEASONS = list(range(2010, 2025))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Schedules use the team code of the day; weekly stats use today's code for every year.
# Without this, every game involving a relocated team silently drops out of the features.
RELOCATED = {'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    schedule = nfl.import_schedules(SEASONS)
    schedule = schedule[schedule['season'].isin(SEASONS)]
    for col in ('home_team', 'away_team'):
        schedule[col] = schedule[col].replace(RELOCATED)

    weekly = nfl.import_weekly_data(SEASONS)

    for season in SEASONS:
        sched_teams = set(schedule.loc[schedule['season'] == season, ['home_team', 'away_team']].values.ravel())
        weekly_teams = set(weekly.loc[weekly['season'] == season, 'recent_team'].dropna())
        assert sched_teams == weekly_teams, f"{season}: team codes differ: {sched_teams ^ weekly_teams}"

    schedule.to_csv(os.path.join(DATA_DIR, 'schedule_2010_2024.csv'), index=False)
    weekly.to_csv(os.path.join(DATA_DIR, 'weekly_2010_2024.csv'), index=False)
    print(f"Saved {len(schedule)} games and {len(weekly)} player-week rows to {DATA_DIR}")


if __name__ == '__main__':
    main()
