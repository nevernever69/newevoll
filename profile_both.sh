#!/bin/bash
set -e
cd /home/ubuntu/pureevol

echo "=== Go1PushRecovery ==="
python profile_go1.py 2>&1 | tee /tmp/go1_profile.log

echo ""
echo "=== PandaPickAndTrack ==="
python profile_panda.py 2>&1 | tee /tmp/panda_profile.log

echo ""
echo "=== DONE ==="
