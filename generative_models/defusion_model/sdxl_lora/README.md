# SDXL LoRA fine-tuning

Train a LoRA adapter on SDXL over your own images, then run inference with it.
Only the LoRA matrices injected into the UNet's attention projections are
trained -- roughly 0.5% of the UNet -- so runs fit on a single consumer GPU.

## Files

| file | role |
|---|---|
| `dataset.py` | image + caption dataset, with SDXL size/crop micro-conditioning |
| `train_lora.py` | training loop: accelerate, mixed precision, PEFT LoRA, TensorBoard |
| `inference.py` | base model + trained LoRA, with an optional base-model comparison |
| `../requirements.txt` | dependencies |

## Data layout

A flat folder of images, each with an optional caption of the same stem:

```
data/my_dog/
  img_001.jpg
  img_001.txt      -> "a photo of a sks dog sitting on grass"
  img_002.png
  img_002.txt
```

Images with no `.txt` fall back to `--instance-prompt`, which is what you want
when every image is the same subject.

Converting a class-folder dataset such as Kaggle Dogs vs. Cats:

```bash
python - <<'PY'
from pathlib import Path
src, dst = Path('data/dogs_vs_cats/train'), Path('data/lora_dogs')
dst.mkdir(parents=True, exist_ok=True)
for i, p in enumerate(sorted(src.glob('dog.*.jpg'))[:200]):
    out = dst / f'dog_{i:04d}.jpg'
    out.write_bytes(p.read_bytes())
    out.with_suffix('.txt').write_text('a photo of a dog')
PY
```

## Train

```bash
accelerate config default          # once per machine

accelerate launch train_lora.py \
  --data-dir ../data/lora_dogs \
  --output-dir ../outputs/lora_dogs \
  --instance-prompt "a photo of a sks dog" \
  --resolution 1024 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --num-epochs 40 \
  --learning-rate 1e-4 \
  --rank 16 --lora-alpha 16 \
  --mixed-precision bf16 \
  --gradient-checkpointing \
  --checkpointing-steps 500
```

From a local single-file checkpoint instead of the hub:

```bash
accelerate launch train_lora.py --data-dir ../data/lora_dogs \
  --base-model /models/RealVisXL_V4.0.safetensors
```

Watch it: `tensorboard --logdir ../outputs/lora_dogs/logs`

## Infer

```bash
python inference.py \
  --lora-path ../outputs/lora_dogs \
  --prompt "a photo of a sks dog running on a beach|a sks dog wearing sunglasses" \
  --num-images 2 --lora-scale 0.8 --compare-base
```

`--compare-base` renders each prompt twice at the same seed, once with the
adapter muted, so you can see exactly what the LoRA changed.

## VRAM

At 1024px, batch 1, rank 16, bf16:

| configuration | approx. VRAM |
|---|---|
| baseline | ~19 GB |
| `--gradient-checkpointing` | ~12 GB |
| + `--enable-xformers` (CUDA) | ~10 GB |
| + 8-bit Adam | ~9 GB |

Under 12 GB, drop to `--resolution 768` and raise
`--gradient-accumulation-steps` to hold the effective batch size.

## Notes

- **fp16 and the VAE.** The stock SDXL VAE overflows in fp16 and yields black
  images or NaN loss. Under `--mixed-precision fp16` the script substitutes
  `madebyollin/sdxl-vae-fp16-fix` automatically. Prefer `bf16` where the
  hardware supports it.
- **LoRA params stay fp32.** Even under mixed precision, fp16 master weights
  underflow at 1e-4 and the adapter quietly stops learning.
- **Micro-conditioning is real.** SDXL consumes original size and crop offset as
  inputs. `dataset.py` reports the true values per image (mirroring the crop
  offset when it flips); feeding constants there produces soft, off-centre output.
- **`--snr-gamma 5.0`** enables Min-SNR loss weighting, which usually converges
  faster than uniform weighting.
- **`--caption-dropout`** trains a fraction of steps with an empty caption,
  preserving the base model's response to classifier-free guidance.
