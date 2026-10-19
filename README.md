# Vision Transformer (ViT) from Scratch — CIFAR-10

A Vision Transformer implemented **from scratch in PyTorch** — no `timm`, no pretrained
weights — trained on CIFAR-10, benchmarked against a CNN baseline, and studied through
ablations and attention-map visualization. Every component is built from first principles.

---

## Headline results

| | |
|---|---|
| **Best test accuracy (CIFAR-10)** | **79.9%** (depth-12, 40 epochs, cosine LR schedule) |
| Base model | 75.7% (depth-6, 683K params, 20 epochs) |
| Train–test gap | ~1% (minimal overfitting) |
| CNN (ResNet-18) baseline | 89.3% |

![Attention map](attention_map.png)

*The CLS token's attention concentrates on the object (a frog) rather than the background —
evidence the model learned to focus on relevant regions.*

---

## Architecture

Built component by component in PyTorch:

- **Patch embedding** — a strided convolution splits each 32×32 image into 4×4 patches
  (64 patches) and projects them to 96-dim embeddings.
- **[CLS] token + positional embeddings** — a learnable classification token is prepended
  and learnable position vectors added (65 tokens total).
- **Multi-head self-attention** — scaled dot-product attention across 4 heads with a fused
  QKV projection: `Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V`.
- **Transformer encoder blocks** — pre-norm blocks with residual connections and a GELU
  MLP, stacked 6 (base) to 12 (best) deep.
- **Classification head** — the final CLS token is read out through a linear layer → 10
  class logits.

The base model has **683K parameters** — deliberately small for a dataset trained from
scratch on a single GPU (the original ViT was 86M, for 224×224 ImageNet).

---

## Training

![Training curves](training_curves.png)

Trained with CrossEntropyLoss and AdamW (lr 3e-4, weight decay 0.05). The training set uses
RandomCrop + horizontal-flip augmentation; the test set is kept clean. Train and test curves
track closely — augmentation held the gap to ~1%. The best model (depth 12) used a cosine
learning-rate schedule over 40 epochs to reach 79.9%.

---

## Experiments

### CNN baseline & inductive bias

A ResNet-18 CNN, adapted for 32×32 inputs, trained under identical settings:

| Model | Parameters | Test accuracy |
|---|---|---|
| ViT (from scratch) | 683K | 75.7% |
| ResNet-18 CNN | 11.2M | **89.3%** |

The CNN wins despite parameter count being irrelevant to the outcome. CNNs have strong
**inductive biases** — locality and translation invariance — built into the architecture and
perfectly suited to images, making them data-efficient. ViTs have weak inductive biases; they
must learn spatial structure from data, so they need much more of it. On a small dataset like
CIFAR-10 without pretraining, the CNN's built-in assumptions win — which is why the original
ViT paper required pretraining on ~300M images to surpass CNNs.

### Ablation studies

All ablations use a consistent 10 epochs for fair comparison.

**Patch size:**

| Patch size | Patches | Test acc |
|---|---|---|
| 2×2 | 256 | 0.683 |
| 4×4 | 64 | **0.692** |
| 8×8 | 16 | 0.599 |

4×4 is the sweet spot — it matches the finer 2×2's accuracy while being ~4× cheaper
(attention is O(N²) in sequence length). 8×8 loses detail. Smaller isn't automatically better:
a small model can't fully exploit a 257-token sequence in limited training.

**Model depth:**

| Depth | Parameters | Test acc |
|---|---|---|
| 2 | 236K | 0.628 |
| 6 | 683K | 0.689 |
| 12 | 1.35M | **0.698** |

Accuracy rises monotonically with depth, with no overfitting even at 1.35M params — but with
diminishing returns.

**Augmentation:**

| Setting | Train acc | Test acc | Gap |
|---|---|---|---|
| With augmentation | 0.678 | 0.689 | **−0.011** |
| Without augmentation | 0.762 | 0.662 | **+0.101** |

A controlled test of overfitting: without augmentation the train–test gap blows up to 10
points (the model memorizes the training set). With augmentation the gap essentially vanishes.
The no-aug model has *higher* train accuracy — but that's memorization, not generalization.

---

## Real-world images & distribution shift

Tested on downloaded real-world photos (not the CIFAR test set). The model classifies clean,
centered, in-distribution images well but degrades on blurry, oddly-angled, or out-of-class
images — a textbook **distribution-shift** limitation: a model only performs where test data
resembles training data. It is also a closed-set classifier (10 classes, no "unknown" option).
Center-crop preprocessing (`Resize(36)` → `CenterCrop(32)`) noticeably improved custom-image
accuracy over a direct squash-resize. The production fix would be transfer learning from a
pretrained backbone — deliberately out of scope here, since the goal was to build and
understand the architecture from scratch.

---

## Running it

The notebook (`vit_from_scratch.ipynb`) runs top to bottom.

- **On Colab / locally:** set `DATA_ROOT = './data'` and `DOWNLOAD = True` — CIFAR-10
  downloads automatically. Pick a GPU runtime.
- **On Kaggle:** attach the *CIFAR-10 Python* dataset and set `DATA_ROOT` to its input path
  (`download=False`).

```bash
pip install torch torchvision matplotlib
jupyter notebook vit_from_scratch.ipynb
```

---

## Key takeaways

- Built and understood every part of a transformer — attention, positional encoding, the CLS
  token, residual/norm structure — from scratch.
- Demonstrated the CNN-vs-ViT inductive-bias tradeoff empirically on identical settings.
- Quantified design choices (patch size, depth, augmentation) through controlled ablations.
- Visualized attention to interpret what the model attends to, and analyzed distribution shift
  on real-world images.
