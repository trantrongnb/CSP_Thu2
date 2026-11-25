# Temporal encoder model experiment

This change adds an optional, lightweight dilated temporal convolution to MFPA-FSLR. The existing model encodes each flow/landmark transition independently and then applies attention pooling; this can weight transitions but has limited ability to represent the order and local evolution of a gesture. The new encoder processes each region's sequence before temporal attention pooling.

- Two residual depthwise temporal convolution blocks with dilations 1 and 2.
- Shared across left hand, right hand, and mouth to limit additional parameters.
- Preserves sequence length and explicitly masks missing/padded transitions.
- No pretrained weights or additional preprocessing required.
- `model.temporal_encoder: none` restores the prior framewise model; `dilated_conv` enables the new path.

## Fair comparison

The original configs remain unchanged and use the framewise path. New `*_temporal.yaml` configs enable the temporal encoder and use separate output directories. Compare the original and temporal configurations for the same shot setting using the same training seed and fixed validation/test episode files. Do not overwrite prior checkpoints or use test accuracy to select the model.

This is a hypothesis-driven architecture change, not a measured accuracy claim. Validate with multiple training seeds and fixed episodes. The unit tests check shapes, masking, gradients, and order sensitivity; they do not establish a recognition-accuracy gain.
