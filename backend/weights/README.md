# Model weights

Runtime location for checkpoints. Files here are git-ignored and must never be
committed.

## Present

`zero_dce_plus_plus.pth` — the published Zero-DCE++ checkpoint used by NoirNet:

| Field | Value |
| --- | --- |
| Source | `Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth` in `Li-Chongyi/Zero-DCE_extension` |
| Size | 52,395 bytes |
| Git blob SHA | `3d1a7b8f5aee769d8ae55531cd43c4cf8d45efa8` |
| SHA-256 | `ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84` |
| Parameters | 10,561 (28 tensors, float32) |

Download URL:

```
https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/main/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth
```

The wrapper validates the checkpoint strictly (parameter count, tensor count,
exact key set) before loading. A file that does not match the Zero-DCE++
architecture is rejected with `checkpoint_invalid`, including the original
Zero-DCE checkpoint (~79K parameters), which is a different network.

## Deployment

```bash
cp /path/to/Epoch99.pth backend/weights/zero_dce_plus_plus.pth
```

Or point elsewhere:

```bash
LUMENOIR_MODEL_ID=zero_dce_plus_plus
LUMENOIR_MODEL_WEIGHTS=/absolute/path/to/checkpoint.pth
```

Trained/fine-tuned checkpoints keep the same key names, so the inference wrapper
loads them without code changes. The licence of the pretrained checkpoint is
CC BY-NC 4.0 (academic research use). See `docs/zero_dce.md`.
