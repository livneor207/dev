# Ellipse Detection Assignment

This project detects whether a 50×50 image contains an ellipse and, when it does, predicts its center, semi-axes, and orientation.

The network returns seven values for each image:

```text
[ellipse_logit, cx, cy, a, b, u, v]
```

- `ellipse_logit` is a raw classification logit.
- `cx`, `cy` are the ellipse center coordinates, normalized by image width and height.
- `a`, `b` are the semi-axis radii, normalized by the 50-pixel image size.
- `(u, v)` represents orientation as `(cos(2θ), sin(2θ))`.

The prediction is converted back to pixels and degrees at inference time. If the ellipse probability is below the selected threshold, its geometric output is zeroed.

## Model outputs: activation per head

Each of the seven numbers comes from a separate linear head on top of the CNN features. The table below is the full path from network to loss.

| Index | Name | Head output | Activation after head | Value range | Used in loss as |
|------:|------|-------------|------------------------|-------------|-----------------|
| 0 | `ellipse_logit` | 1 linear value | **none** (raw logit) | (−∞, +∞) | `BCEWithLogitsLoss(logit, y)`; at inference use `sigmoid(logit)` for probability |
| 1 | `cx` | 1 linear value | **sigmoid** | [0, 1] | MSE vs `center_x / 50` |
| 2 | `cy` | 1 linear value | **sigmoid** | [0, 1] | MSE vs `center_y / 50` |
| 3 | `a` | 1 linear value | **sigmoid** | [0, 1] | MSE vs major semi-axis `/ 50` |
| 4 | `b` | 1 linear value | **sigmoid** | [0, 1] | MSE vs minor semi-axis `/ 50` |
| 5 | `u` | 1 linear value | **L2 normalize** with `v` | unit vector | MSE vs `cos(2θ)` (same optimum as cosine loss) |
| 6 | `v` | 1 linear value | **L2 normalize** with `u` | unit vector | MSE vs `sin(2θ)` |

In code (`ellipse_cv/model.py`):

```python
logit = cls_head(features)                    # no activation
center = sigmoid(center_head(features))       # cx, cy
radii  = sigmoid(radii_head(features))        # a, b
angle  = normalize(angle_head(features))      # u, v on unit circle
```

**Why this design**

- **Logit without sigmoid:** `BCEWithLogitsLoss` expects a raw score; applying sigmoid inside the model would be wrong for training.
- **Sigmoid on center and radii:** keeps predictions in normalized image coordinates without clipping in the loss.
- **L2 on angle (not sigmoid, not tanh):** the network outputs two free numbers, then they are projected onto the unit circle so `(u, v)` always has length 1. Orientation is stored as `(cos 2θ, sin 2θ)`, not as a single angle in degrees.

## Canonical axes and angle (label preprocessing)

An ellipse can be described in two equivalent ways: which semi-axis is “first”, and an angle that differs by 90° when you swap them. Without fixing this, the same ellipse could have two valid labels and the model would be punished for predicting either one.

**Rule used in training:** always store the **major** semi-axis as `a` (axis_1) and the **minor** as `b` (axis_2), with `a ≥ b`. If the CSV has `axis_1 < axis_2`, we swap the axes and add 90° to the angle (mod 180°).

**Example**

CSV row (before canonicalization):

```text
center = (25, 25), angle = 30°, axis_1 = 10, axis_2 = 20
```

Here the longer radius is 20, but it was stored in `axis_2`. After canonicalization:

```text
a = 20, b = 10, angle = (30 + 90) % 180 = 120°
```

The ellipse shape is unchanged; only the parameterization is unique. This removed a large part of the early angle error (roughly 45° MAE down to about 6°) because the target angle no longer jumps by 90° for the same visual ellipse.

**At inference:** the model predicts normalized `a`, `b` and `(u, v)`. We decode angle with:

```text
θ_deg = (0.5 * atan2(v, u)) in degrees, wrapped to [0, 180)
```

then convert `cx, cy, a, b` back to pixels by multiplying by 50.

## Setup

This repository uses a Python virtual environment and `requirements.txt`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run commands from the repository root with `.venv/bin/python`, or activate the environment first:

```bash
source .venv/bin/activate
```

## Data and preprocessing

`train_data.csv` contains the image path, whether an ellipse is present, center, angle, and two radii.

- The data is split once into 80% training and 20% validation data.
- The split is stratified by ellipse presence and fixed by `--seed 42`.
- RGB images are converted from `uint8` to `[0, 1]`, then standardized using per-channel mean and standard deviation computed on the training split only.
- The current RGB statistics are mean `[0.60428, 0.60480, 0.60422]` and standard deviation `[0.04637, 0.04709, 0.04551]`.
- Labels are canonicalized as described above (`a ≥ b`, angle adjusted when swapping). See **Canonical axes and angle**.

The image geometry is never changed by augmentation. Photometric augmentation, when requested, may only change brightness, contrast, saturation, color/grayscale appearance, or blur.

## Model

The preferred architecture is a four-stage RGB residual CNN:

```text
RGB image (3×50×50)
  → residual block (3×3, same padding) → max pool
  → residual block (3×3, same padding) → max pool
  → residual block (3×3, same padding) → max pool
  → residual block (3×3, same padding) → max pool
  → four task heads
  → [logit, cx, cy, a, b, cos(2θ), sin(2θ)]
```

Residual connections preserve information across convolution blocks. A 1×1 convolution is used only in a skip path when channel counts differ; it is not used as the spatial feature extractor. The 3×3 convolutions use padding 1, so each residual block preserves its spatial size before pooling.

See **Model outputs: activation per head** for sigmoid vs linear vs L2-normalize on each output.

## Loss function

Total loss (only for images with an ellipse, `y = 1`, do we add the geometry terms):

```text
L = λ1 * L_cls
  + y * (λ2 * L_center + λ3 * L_radii + λ4 * L_angle)
```

Default weights: λ1 = λ2 = λ3 = λ4 = 1.

| Term | Formula (conceptually) | When applied |
|------|------------------------|--------------|
| `L_cls` | BCEWithLogits on `ellipse_logit` vs `y` | every sample |
| `L_center` | MSE on `(cx, cy)` vs normalized center | only if `y = 1` |
| `L_radii` | MSE on `(a, b)` vs normalized axes | only if `y = 1` |
| `L_angle` | MSE on `(u, v)` vs `(cos 2θ, sin 2θ)` | only if `y = 1` |

**Angle loss and cosine loss**

Target vector: `v = (cos 2θ, sin 2θ)`. Predicted vector after L2 norm: `v_hat = (u, v)`.

Cosine-style loss:

```text
L_angle_cos = 1 - dot(v_hat, v)
```

We train with MSE on the two components instead:

```text
L_angle_mse = mean( (u - cos 2θ)^2 + (v - sin 2θ)^2 )
```

Because both vectors have length 1, these are equivalent up to a constant factor:

```text
||v_hat - v||^2 = 2 * (1 - dot(v_hat, v))
```

So MSE on `(u, v)` pushes the prediction the same way as cosine loss; only the scale in the total loss differs.

## Angle conversion and remaining errors

**Training target:** angle θ in degrees from the canonical label → `(cos 2θ, sin 2θ)`.

**Inference:** from model outputs `u, v`:

```text
θ_deg = degrees( 0.5 * atan2(v, u) ) mod 180
```

The `0.5` undoes the `2θ` encoding. Using `2θ` means 0° and 180° are the same orientation for an ellipse (you cannot tell “which end” of the major axis is which from the image alone).

**How angle error is reported:** circular distance in degrees:

```text
err = min( |θ_pred - θ_true|, 180 - |θ_pred - θ_true| )
```

**Why some images still have ~90° errors:** often near-circular ellipses (`a ≈ b`) or ambiguous labels. When the ellipse is almost a circle, rotation barely changes the pixels, so any angle is almost equally valid. Canonicalization fixes the swap-by-90° ambiguity in the **labels**; it does not remove physical ambiguity when the shape is nearly round.

## Train the preferred configuration

The current champion is `runs/rgb_photometric_aug_dropout0p1/` (epoch 44).
It uses an RGB fine-grained residual CNN with photometric-only augmentation.
Run the documented wrapper to reproduce the configuration; it always writes to
a new output directory:

```bash
.venv/bin/python scripts/train_best_rgb.py
```

The exact command is:

```bash
.venv/bin/python scripts/train.py \
  --csv train_data.csv \
  --base-dir . \
  --rgb \
  --color-space rgb \
  --depth 4 \
  --base-filters 32 \
  --kernel-size 3 \
  --fine-grained \
  --head-hidden-dim 128 \
  --dropout 0.1 \
  --batch-size 128 \
  --optimizer adamw \
  --lr 0.001 \
  --weight-decay 0.01 \
  --cls-loss bce \
  --reg-loss mse \
  --scheduler plateau \
  --scheduler-patience 5 \
  --scheduler-factor 0.5 \
  --min-lr 1e-6 \
  --brightness-jitter 0.1 \
  --contrast-jitter 0.1 \
  --saturation-jitter 0.1 \
  --grayscale-aug-prob 0.1 \
  --blur-prob 0.15 \
  --blur-radius 0.75 \
  --epochs 50 \
  --seed 42 \
  --output-dir runs/<new_experiment_name>
```

Never train into `runs/best_baseline/` and never reuse an existing run directory. Each run saves `config.json`, `best.pt`, `last.pt`, `metrics.csv`, `best_threshold.json`, prediction examples, curves, and highest-error analysis.

The optimizer is AdamW with initial learning rate `0.001`, weight decay `0.01`, and betas `(0.9, 0.99)`. `ReduceLROnPlateau` halves the learning rate after five epochs without validation-loss improvement, down to `1e-6`. Early stopping is enabled.

## Run inference

Use the checkpoint and threshold from the same run directory:

```bash
.venv/bin/python scripts/infer.py \
  --ckpt runs/rgb_photometric_aug_dropout0p1/best.pt \
  --data images/test \
  --out predictions.csv
```

For evaluation on a labeled CSV, provide the optional `--csv` argument. The script loads the checkpoint configuration and its training normalization statistics automatically.

## How predictions are matched and evaluated

Classification predictions are matched to the binary `is_ellipse` label using `sigmoid(ellipse_logit)` and the run's F2-optimized threshold.

For images with a ground-truth ellipse:

- Center and axis error are reported as normalized MAE/MSE; multiply by 50 for pixels.
- Angle is decoded to degrees and evaluated using circular MAE.
- Classification reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and F2.

Do not compare weighted total loss between experiments that use different loss types or weights. Compare raw validation metrics on the same split: F2, angle MAE in degrees, center error in pixels, and axis error in pixels.

## Results

Best fixed-weight RGB configuration (`rgb_photometric_aug_dropout0p1`, epoch 44):

- Validation loss: `0.0768`
- F1: `0.9989`
- Angle MAE: `5.55°`
- Center MAE: `0.00958` normalized, about `0.48 px`
- Axes MAE: `0.00954` normalized, about `0.48 px`

Experiments with 48 base filters, automatic loss weighting, and combined photometric augmentation did not improve the overall result. The wider model modestly improved axes but worsened center and angle; the combined photometric experiment slightly improved angle/axes but worsened center and classification.

## Latest validation predictions

The latest experiment used color jitter, occasional RGB-to-grayscale conversion, and Gaussian blur. It is shown here for qualitative inspection; it is not the preferred checkpoint.

![Latest validation predictions](runs/rgb_photometric_aug_dropout0p1/examples_val.png)

For the champion model's examples and error grid, see:

- `runs/rgb_photometric_aug_dropout0p1/examples_val.png`
- `runs/rgb_photometric_aug_dropout0p1/highest_errors.png`
- `runs/rgb_photometric_aug_dropout0p1/highest_errors.json`
