#!/usr/bin/env bash
set -euo pipefail

# Paths you specified
SDS_ROOT="/data/remastered/SDS_KSC"
BUD_ROOT="/data/remastered/BUD_KSC"

# Options
OVERWRITE="--overwrite"     # remove/replace existing symlinks; set to "" to skip replacing
LOC_BLANK_AS=".."           # how to render blank/-- location codes in BUD filenames

# 1) Dry run (prints planned links)
echo "=== Dry run (no changes) ==="
python3 sds_to_bud_symlinks.py "$SDS_ROOT" "$BUD_ROOT" --dry-run --loc-blank-as "$LOC_BLANK_AS"

# 2) Real run
echo ""
echo "=== Creating symlinks ==="
python3 sds_to_bud_symlinks.py "$SDS_ROOT" "$BUD_ROOT" ${OVERWRITE} --loc-blank-as "$LOC_BLANK_AS"

echo "Done."