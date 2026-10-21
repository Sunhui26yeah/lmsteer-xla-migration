#!/bin/bash
# Sample tpu-info every 5 minutes while the full run is alive (stops when END appears).
LOG=~/lmsteer/tpu_info_full.log
while ! grep -q "^END" ~/lmsteer/a$1_driver.log 2>/dev/null; do
  echo "=== $(date -u +%FT%TZ) stage=$(tail -1 ~/lmsteer/a$1_driver.log)" >> $LOG
  ~/venv_lmsteer/bin/tpu-info >> $LOG 2>&1
  sleep 300
done
