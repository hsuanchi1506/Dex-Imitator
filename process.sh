
ROOT="data/OakInk-v2"

mkdir -p "$ROOT/coacd_object_preview/align_ds"


find "$ROOT/object_preview/align_ds" -type f \( -iname '*.obj' -o -iname '*.ply' \) -print0 |
while IFS= read -r -d '' SRC; do
  REL="${SRC#"$ROOT/object_preview/"}"         # e.g. align_ds/xx/xx.obj
  DST="$ROOT/coacd_object_preview/$REL"        # e.g. coacd_object_preview/align_ds/xx/xx.obj

  mkdir -p "$(dirname "$DST")"
  echo "[COACD] $SRC -> $DST"
  python maniptrans_envs/lib/utils/coacd_process.py \
    -i "$SRC" -o "$DST" \
    --max-convex-hull 32 --seed 1 -mi 2000 -md 5 -t 0.07
done
