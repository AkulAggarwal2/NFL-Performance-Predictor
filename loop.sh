#!/usr/bin/env bash
# Autonomous model-improvement loop.
#
#   ./loop.sh <focus> <iterations>
#   ./loop.sh "feature engineering: rest days and division games" 5
#
# Each iteration asks a headless Claude to make one change aimed at <focus>, then
# re-scores it here. The change is committed only if pytest passes and SCORE beats the
# best so far by at least MIN_GAIN; anything else is reverted. Every iteration is logged
# to loop_log.tsv; each agent transcript is saved under loop_runs/ (both gitignored).
set -uo pipefail

cd "$(dirname "$0")"

PY=.venv/bin/python
MIN_GAIN=0.002
PROTECTED=(evaluate.py fetch_data.py tests CLAUDE.md loop.sh .gitignore)
LOG=loop_log.tsv
RUNS=loop_runs

die() { echo "loop.sh: $*" >&2; exit 1; }

[ $# -eq 2 ] || die "usage: ./loop.sh <focus> <iterations>"
FOCUS=$1
ITERATIONS=$2
[[ $ITERATIONS =~ ^[1-9][0-9]*$ ]] || die "iterations must be a positive integer, got '$ITERATIONS'"
[ -x "$PY" ] || die "$PY not found (see CLAUDE.md: Environment & Commands)"
command -v claude >/dev/null || die "claude CLI not on PATH"
# Reverts use reset --hard + clean -fd, which would destroy uncommitted work.
[ -z "$(git status --porcelain)" ] || die "working tree not clean; commit or stash first"

# data/ is gitignored, so git can't revert edits to it. Fingerprint it and stop if it changes.
data_hash() { shasum data/*.csv | shasum | cut -d' ' -f1; }
DATA_HASH=$(data_hash)

# Prints the SCORE value, or nothing if evaluate.py failed (e.g. the leakage test).
score() { "$PY" evaluate.py 2>&1 | tee "$1" | awk '/^SCORE:/ {print $2}' | tail -1; }

# True if $1 beats $2 by at least MIN_GAIN. Compared in integer 1e-4 units so 4-decimal
# scores don't hit float rounding at the boundary.
beats() {
  awk -v n="$1" -v b="$2" -v g="$MIN_GAIN" \
    'BEGIN { exit !(int(n * 10000 + 0.5) <= int(b * 10000 + 0.5) - int(g * 10000 + 0.5)) }'
}

revert() { git reset --hard -q "$START_HEAD" && git clean -fdq; }

log() {  # score pytest outcome commit
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$ITER" "$FOCUS" "$1" "$BEST" "$2" "$3" "$4" >> "$LOG"
  echo "  -> $3 (score $1, best $BEST, pytest $2)"
}

mkdir -p "$RUNS"
[ -f "$LOG" ] || printf 'time\titer\tfocus\tscore\tbest_before\tpytest\toutcome\tcommit\n' > "$LOG"
STAMP=$(date '+%Y%m%d-%H%M%S')

echo "Scoring the starting point..."
BEST=$(score "$RUNS/$STAMP-baseline-evaluate.txt")
[ -n "$BEST" ] || die "evaluate.py printed no SCORE at baseline; see $RUNS/$STAMP-baseline-evaluate.txt"
echo "Starting SCORE: $BEST"

for ((ITER = 1; ITER <= ITERATIONS; ITER++)); do
  echo "=== Iteration $ITER/$ITERATIONS (best $BEST) ==="
  START_HEAD=$(git rev-parse HEAD)
  RUN="$RUNS/$STAMP-iter$ITER"

  PROMPT="Read CLAUDE.md and follow its Experiment rules exactly.
Focus for this attempt: $FOCUS
The current best SCORE is $BEST (log loss, lower is better).
Make one change in nfl_predictor.py that is likely to lower SCORE. Then run
'.venv/bin/python evaluate.py' and '.venv/bin/python -m pytest tests/ -q', and finish by
reporting SCORE, accuracy, Brier, min/max probability, and a one-line description of the
change. Do not commit; git is not available to you, and the calling script decides
whether to keep the change."

  # --allowedTools takes every following argument, so it must stay last.
  claude -p "$PROMPT" \
    --max-turns 30 \
    --permission-mode acceptEdits \
    --allowedTools "Read" "Edit" "Write" "Bash(.venv/bin/python evaluate.py)" "Bash(.venv/bin/python -m pytest:*)" \
    > "$RUN-agent.txt" 2>&1
  echo "  agent exited $? (transcript: $RUN-agent.txt)"

  if [ "$(data_hash)" != "$DATA_HASH" ]; then
    revert
    log "-" "-" "ABORT_data_modified" "-"
    die "data/ changed during iteration $ITER; git can't restore it. Rerun: $PY fetch_data.py"
  fi

  if [ "$(git rev-parse HEAD)" != "$START_HEAD" ]; then
    revert; log "-" "-" "reverted_agent_moved_HEAD" "-"; continue
  fi

  if [ -n "$(git status --porcelain -- "${PROTECTED[@]}")" ]; then
    revert; log "-" "-" "reverted_protected_file_changed" "-"; continue
  fi

  if [ -z "$(git status --porcelain)" ]; then
    log "-" "-" "no_change" "-"; continue
  fi

  NEW=$(score "$RUN-evaluate.txt")
  if "$PY" -m pytest tests/ -q > "$RUN-pytest.txt" 2>&1; then TESTS=pass; else TESTS=fail; fi

  if [ -z "$NEW" ]; then
    revert; log "none" "$TESTS" "reverted_no_score" "-"
  elif [ "$TESTS" != pass ]; then
    revert; log "$NEW" "$TESTS" "reverted_tests_failed" "-"
  elif beats "$NEW" "$BEST"; then
    git add -A
    git commit -q -m "loop: $FOCUS (SCORE $BEST -> $NEW)" \
      -m "Iteration $ITER. Agent transcript: $RUN-agent.txt" \
      -m "Co-Authored-By: Claude <noreply@anthropic.com>"
    log "$NEW" "$TESTS" "kept" "$(git rev-parse --short HEAD)"
    BEST=$NEW
  else
    revert; log "$NEW" "$TESTS" "reverted_no_gain" "-"
  fi
done

echo "Done. Best SCORE: $BEST. Log: $LOG"
