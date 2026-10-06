"""Score LoRA checkpoints so the best one can be chosen from numbers, not vibes.

Diffusion training loss does not rank image quality -- it is dominated by which
random timestep each step happened to draw. This evaluates checkpoints the way
the outcome is actually judged: generate a fixed prompt/seed grid from each, then
measure four things with CLIP.

    breed accuracy   top-1 over the dataset's breed names   -> should rise
    clip_i           similarity to real photos of the breed -> should rise
    clip_t           image/prompt agreement                 -> should stay flat
    diversity        spread across seeds for one prompt     -> should stay flat

Breed accuracy and clip_i keep climbing as an adapter overfits, so they cannot be
maximised on their own. The checkpoint to ship is the last one before clip_t or
diversity starts to fall.

Example:
    python eval_lora.py --checkpoints base,outputs/lora_pets37/checkpoint-1000 \\
        --breeds Egyptian_Mau,Birman,Bombay,Siamese,beagle --seeds 2
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path('data/lora_pets37')
CLIP_MODEL = 'openai/clip-vit-base-patch32'


def parse_args():
    parser = argparse.ArgumentParser(description='Score LoRA checkpoints')
    parser.add_argument('--checkpoints', required=True,
                        help='Comma separated adapter paths. The literal "base" means no adapter')
    parser.add_argument('--breeds', required=True, help='Comma separated breed folder names')
    parser.add_argument('--base-model', default='SG161222/RealVisXL_V4.0')
    parser.add_argument('--data-dir', default=str(DATA_DIR))
    parser.add_argument('--output-dir', default='./outputs/eval')
    parser.add_argument('--seeds', type=int, default=2, help='Samples per breed')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance-scale', type=float, default=7.0)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--lora-scale', type=float, default=1.0)
    parser.add_argument('--real-refs', type=int, default=16,
                        help='Real images per breed used as the clip_i reference set')
    parser.add_argument('--device', default=None)
    parser.add_argument('--skip-existing', action='store_true',
                        help='Reuse images already on disk instead of regenerating')
    return parser.parse_args()


def resolve_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def breed_label(breed):
    species = 'cat' if breed[:1].isupper() else 'dog'
    return f'{breed.replace("_", " ").lower()} {species}'


def prompt_for(breed):
    return f'a photo of a {breed_label(breed)}'


class ClipScorer:
    """CLIP embeddings for the four metrics, all cosine similarities."""

    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    @staticmethod
    def _as_tensor(output):
        """get_*_features returns a tensor on some transformers versions and a
        ModelOutput on others; normalise to the embedding tensor."""
        if torch.is_tensor(output):
            return output
        for attribute in ('text_embeds', 'image_embeds', 'pooler_output'):
            value = getattr(output, attribute, None)
            if torch.is_tensor(value):
                return value
        raise TypeError(f'Cannot read embeddings from {type(output).__name__}')

    def _normalise(self, output):
        feats = self._as_tensor(output).float()
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def image_embeds(self, images):
        inputs = self.processor(images=images, return_tensors='pt').to(self.device)
        return self._normalise(self.model.get_image_features(**inputs))

    @torch.no_grad()
    def text_embeds(self, texts):
        inputs = self.processor(text=texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        return self._normalise(self.model.get_text_features(**inputs))


def load_real_references(data_dir, breeds, scorer, limit):
    """Mean CLIP embedding of real photos per breed, the clip_i target."""
    # One mean embedding per breed. Note this metric rewards matching the training
    # distribution, including whatever is wrong with it, so it describes how far the
    # model moved rather than whether it improved -- do not select checkpoints on it.
    refs = {}
    for breed in breeds:
        paths = sorted(Path(data_dir).glob(f'{breed}_*.jpg'))[:limit]
        if not paths:
            logger.warning('no real images for %s; clip_i will be nan', breed)
            continue
        images = [Image.open(p).convert('RGB') for p in paths]
        refs[breed] = scorer.image_embeds(images).mean(dim=0, keepdim=True)
        refs[breed] = refs[breed] / refs[breed].norm(dim=-1, keepdim=True)
    return refs


def build_pipeline(base_model, device):
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
    dtype = torch.float32 if device.type == 'cpu' else torch.float16
    if base_model.endswith('.safetensors') and Path(base_model).exists():
        pipeline = StableDiffusionXLPipeline.from_single_file(base_model, torch_dtype=dtype)
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            base_model, torch_dtype=dtype, use_safetensors=True,
            variant='fp16' if dtype == torch.float16 else None)
    if dtype == torch.float16:
        pipeline.vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=dtype)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config, use_karras_sigmas=True)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline.to(device)


def attach_adapter(pipeline, checkpoint):
    """Swap the adapter in place; 'base' leaves the pipeline untouched."""
    pipeline.unload_lora_weights()
    if checkpoint == 'base':
        return
    path = Path(checkpoint)
    if path.is_dir():
        path = path / 'pytorch_lora_weights.safetensors'
    if not path.exists():
        raise SystemExit(f'No adapter at {checkpoint}')
    pipeline.load_lora_weights(str(path.parent), weight_name=path.name)


def main():
    args = parse_args()
    checkpoints = [c.strip() for c in args.checkpoints.split(',') if c.strip()]
    breeds = [b.strip() for b in args.breeds.split(',') if b.strip()]
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)

    scorer = ClipScorer(device)
    refs = load_real_references(args.data_dir, breeds, scorer, args.real_refs)

    # Zero-shot label set: every breed present in the dataset, so a wrong guess is
    # a real confusion with a plausible alternative rather than a trivial one.
    all_breeds = sorted({p.stem.rsplit('_', 1)[0] for p in Path(args.data_dir).glob('*.jpg')})
    label_texts = [prompt_for(b) for b in all_breeds]
    label_embeds = scorer.text_embeds(label_texts)
    logger.info('CLIP label set: %s breeds', len(all_breeds))

    pipeline = build_pipeline(args.base_model, device)
    rows = []

    for checkpoint in checkpoints:
        tag = 'base' if checkpoint == 'base' else Path(checkpoint).name
        attach_adapter(pipeline, checkpoint)
        logger.info('=== %s ===', tag)
        image_dir = output_dir / tag
        image_dir.mkdir(parents=True, exist_ok=True)

        correct = total = 0
        clip_t_all, clip_i_all, diversity_all = [], [], []

        for breed in breeds:
            prompt = prompt_for(breed)
            images = []
            for seed in range(args.seeds):
                path = image_dir / f'{breed}_{seed:02d}.png'
                if args.skip_existing and path.exists():
                    images.append(Image.open(path).convert('RGB'))
                    continue
                generator = torch.Generator(device='cpu').manual_seed(seed)
                kwargs = {} if checkpoint == 'base' else {
                    'cross_attention_kwargs': {'scale': args.lora_scale}}
                image = pipeline(
                    prompt=prompt, num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale, height=args.size, width=args.size,
                    generator=generator, **kwargs,
                ).images[0]
                image.save(path)
                images.append(image)

            embeds = scorer.image_embeds(images)

            # breed accuracy: nearest breed text among all 37
            sims = embeds @ label_embeds.T
            predictions = [all_breeds[i] for i in sims.argmax(dim=-1).tolist()]
            correct += sum(p == breed for p in predictions)
            total += len(predictions)

            clip_t = float((embeds @ scorer.text_embeds([prompt]).T).mean())
            clip_i = float((embeds @ refs[breed].T).mean()) if breed in refs else float('nan')
            # spread across seeds: 1 - mean pairwise cosine similarity
            if len(images) > 1:
                pair = embeds @ embeds.T
                off = pair[~torch.eye(len(images), dtype=torch.bool, device=pair.device)]
                diversity = float(1 - off.mean())
            else:
                diversity = float('nan')

            clip_t_all.append(clip_t); clip_i_all.append(clip_i); diversity_all.append(diversity)
            logger.info('  %-22s acc %s/%s  clip_t %.4f  clip_i %.4f  div %.4f',
                        breed, sum(p == breed for p in predictions), len(predictions),
                        clip_t, clip_i, diversity)

        mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')
        rows.append({
            'checkpoint': tag,
            'breed_accuracy': correct / total if total else float('nan'),
            'clip_i': mean(clip_i_all),
            'clip_t': mean(clip_t_all),
            'diversity': mean(diversity_all),
            'n_images': total,
        })
        logger.info('%s -> acc %.3f  clip_i %.4f  clip_t %.4f  div %.4f',
                    tag, rows[-1]['breed_accuracy'], rows[-1]['clip_i'],
                    rows[-1]['clip_t'], rows[-1]['diversity'])

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'scores.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n{"checkpoint":<22}{"breed_acc":>11}{"clip_i":>9}{"clip_t":>9}{"diversity":>11}')
    for row in rows:
        print(f'{row["checkpoint"]:<22}{row["breed_accuracy"]:>11.3f}{row["clip_i"]:>9.4f}'
              f'{row["clip_t"]:>9.4f}{row["diversity"]:>11.4f}')
    print(f'\nRule: take the last checkpoint before clip_t or diversity falls.')
    print(f'wrote {csv_path}')


if __name__ == '__main__':
    main()
