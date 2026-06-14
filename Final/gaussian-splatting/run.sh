# Quick reference — copy-paste or use batch scripts below.
#
# IMPORTANT (2024-09 upstream bugs fixed in train.py + dataset_readers.py):
#   NeRF Synthetic must be RETRAINED after the fix (issue #1038 / #1124).
#   GT white background + full-pixel supervision now work correctly.
#
# === Recommended: one command for everything (use tmux) ===
#   cd ~/gaussian-splatting
#   conda activate gaussian_splatting
#   ./run_all.sh
#
# === Or run separately ===
#   ./run_nerf_synthetic.sh    # 8 scenes (chair … ship)
#   ./run_tandt_db.sh          # 4 scenes (truck, train, drjohnson, playroom)
#
# === Single scene ===
#   ./run_high_quality.sh  <nerf_synthetic/scene>  <output/scene>
#   ./run_high_quality_colmap.sh  <tandt_db/.../scene>  <output/scene>
#
# See run_all.sh for ONLY_NERF=1 / ONLY_TANDT=1 splits.
