#!/usr/bin/env bash
# Verifier entry point. Runs as root.
#
# The submitted store is executed only inside unprivileged 'runner' subprocesses
# spawned by the grader. The process that decides pass or fail (pytest) never
# imports it, the reference and the expected answers live in a root-only
# directory the runner cannot read, and the reward file is root owned and set to
# 0 before any submitted code can run.
set -uo pipefail

LOG=/logs/verifier
mkdir -p "$LOG"
chown root:root "$LOG"
chmod 700 "$LOG"
echo 0 > /logs/verifier/reward.txt
chmod 600 /logs/verifier/reward.txt

# The only directory on the submitted code's import path: the store itself plus
# the neutral driver and runner. It holds no answers.
SUT=/run/sut
rm -rf "$SUT"; mkdir -p "$SUT"
cp /tests/sut_runner.py /tests/driver.py "$SUT/"
if [ ! -f /app/src/store.py ]; then
    echo "no /app/src/store.py found; reward stays 0"
    exit 0
fi
cp /app/src/store.py "$SUT/store.py"
chown -R root:root "$SUT"
chmod -R 755 "$SUT"

# Grader tree: reference, generator, driver, tests. Root only.
GRADE=/run/grader
rm -rf "$GRADE"; mkdir -p "$GRADE"
cp /tests/reference.py /tests/gen.py /tests/driver.py /tests/conftest.py \
   /tests/test_outputs.py "$GRADE/"
chown -R root:root "$GRADE"
chmod -R 700 "$GRADE"

# Scratch space the unprivileged runner may read and write.
WORK=/tmp/grader
rm -rf "$WORK"; mkdir -p "$WORK"
chown runner:runner "$WORK"
chmod 755 "$WORK"

cd "$GRADE"
GRADER_SUT_DIR="$SUT" GRADER_WORK="$WORK" GRADER_RUNNER_USER=runner \
PYTHONDONTWRITEBYTECODE=1 \
python -m pytest "$GRADE/test_outputs.py" \
    -p no:cacheprovider -q -rA \
    -o cache_dir="$WORK/pcache" \
    --ctrf /logs/verifier/ctrf.json
STATUS=$?

if [ "$STATUS" -eq 0 ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
chmod 600 /logs/verifier/reward.txt

echo "status=$STATUS reward=$(cat /logs/verifier/reward.txt)"
exit 0
