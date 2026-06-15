# geometry/ - Part 3 geometry analysis package.
#
# This package takes the spike-train snapshots saved during training and computes
# the three population-level activity measures we care about: participation ratio
# (how many dimensions the activity uses), jPCA (whether the activity rotates in a
# structured way), and prep/exec subspace orthogonality (whether planning and
# movement activity live in separate directions).
# See README.md, "Part 3: Geometry analysis pipeline", for the full design.
