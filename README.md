# Vision Transformer (ViT) from Scratch — CIFAR-10

A Vision Transformer implemented **from scratch in PyTorch** — no `timm`, no pretrained
weights — and trained on CIFAR-10. Every component is built from first principles to
understand the architecture end to end.

## Results

| Metric | Value |
|---|---|
| Parameters | 683K |
| Test accuracy (CIFAR-10) | **75%** |
| Epochs | 20 |
| Optimizer | AdamW |
| Train–test accuracy gap | < 3% |

The model was still improving at epoch 20, indicating headroom with longer training
schedules.

## Architecture

Built component by component:

- **Patch embedding** — a strided convolution splits the 32×32 image into 4×4 patches
  and projects each to a 96-dim embedding (64 patches → tokens).
- **[CLS] token + positional embeddings** — a learnable classification token is prepended
  and learnable position vectors are added (65 tokens total).
- **Multi-head self-attention** — scaled dot-product attention across 4 heads, with a
  fused QKV projection.
- **Transformer encoder blocks** — pre-norm blocks with residual connections and a
  GELU MLP, stacked 6 deep.
- **Classification head** — the final `[CLS]` token is read out through a linear layer to
  produce 10 class logits.

## Key design notes

- Scaling attention scores by `√head_dim` keeps score magnitudes stable across the
  per-head dimension.
- Training augmentation (random crop + horizontal flip) held the train–test gap under
  3%, controlling overfitting on a data-hungry architecture.
- A small embedding dimension (96) and depth (6) were chosen deliberately for a small
  dataset trained from scratch on a single GPU.

## Running it

The notebook (`vit_from_scratch.ipynb`) runs top to bottom.

- **On Kaggle:** attach the *CIFAR-10 Python* dataset and set `DATA_ROOT` to its input
  path (`download=False`).
- **Locally:** set `DATA_ROOT = './data'` and `DOWNLOAD = True` to fetch CIFAR-10
  automatically.

```bash
pip install torch torchvision matplotlib
jupyter notebook vit_from_scratch.ipynb
```

## Roadmap

- [ ] CNN (ResNet) baseline for a head-to-head accuracy–data comparison
- [ ] Ablation studies: patch size, model depth, augmentation strength
- [ ] Attention-map visualization for interpretability
