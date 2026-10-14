"""Local (open-weight) model methods. Importing this package does not import torch."""

# Bump when local scoring numerics change; folded into local runs' config hash.
# v2: output head applied in float32 (bf16 logits are quantized in ~0.125 steps).
SCORING_VERSION = 2
