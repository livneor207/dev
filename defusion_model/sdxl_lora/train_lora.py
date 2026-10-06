"""Train a LoRA adapter on SDXL over a folder of images and captions.

Only the LoRA matrices injected into the UNet's attention projections are
trained. The VAE, both text encoders and all original UNet weights stay frozen,
so a run touches a few tens of millions of parameters instead of 2.6 billion.

Example:
    accelerate launch sdxl_lora/train_lora.py \
        --data-dir ./data/my_dog \
        --output-dir ./outputs/lora_my_dog \
        --instance-prompt "a photo of a sks dog" \
        --resolution 1024 --batch-size 1 --gradient-accumulation-steps 4 \
        --num-epochs 40 --learning-rate 1e-4 --mixed-precision bf16 \
        --gradient-checkpointing

    # from a local single-file checkpoint instead of the hub
    accelerate launch sdxl_lora/train_lora.py --data-dir ./data/my_dog \
        --base-model /models/RealVisXL_V4.0.safetensors

Single-GPU runs work with plain `python` too; `accelerate launch` is only
required for multi-GPU or when you want accelerate's config to apply.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoTokenizer, PretrainedConfig

# Works both when run as a script (sys.path[0] is this folder) and when
# imported as a package by the poetry console script.
try:
    from defusion_model.sdxl_lora.dataset import ImageCaptionDataset, collate_fn
except ImportError:  # pragma: no cover - direct script invocation
    from dataset import ImageCaptionDataset, collate_fn

logger = get_logger(__name__)

DEFAULT_BASE_MODEL = 'stabilityai/stable-diffusion-xl-base-1.0'
# The stock SDXL VAE overflows in fp16 and silently produces black images or NaN
# loss. This community re-scaling of the same weights is numerically safe there.
FP16_SAFE_VAE = 'madebyollin/sdxl-vae-fp16-fix'
# SDXL cross-attention projections. Training these four is the standard LoRA
# surface: it covers both self- and cross-attention without touching the
# convolutional trunk, which is where most of the base model's knowledge lives.
LORA_TARGET_MODULES = ['to_q', 'to_k', 'to_v', 'to_out.0']


def parse_args():
    parser = argparse.ArgumentParser(description='LoRA fine-tuning for SDXL')

    paths = parser.add_argument_group('paths')
    paths.add_argument('--data-dir', required=True, help='Folder of images with optional .txt captions')
    paths.add_argument('--output-dir', default='./outputs/sdxl_lora')
    paths.add_argument('--base-model', default=DEFAULT_BASE_MODEL,
                       help='Hub id of an SDXL model, or a path to a single .safetensors checkpoint')
    paths.add_argument('--vae-path', default=None,
                       help=f'Override the VAE. Defaults to {FP16_SAFE_VAE} under fp16')
    paths.add_argument('--revision', default=None, help='Hub revision of the base model')
    paths.add_argument('--variant', default=None,
                       help="Weight variant to load, e.g. fp16. Many hub models ship only an "
                            "fp16 variant; without this they fall back to the fp32 files")

    data = parser.add_argument_group('data')
    data.add_argument('--resolution', type=int, default=1024)
    data.add_argument('--instance-prompt', default=None, help='Caption for images with no .txt sidecar')
    data.add_argument('--caption-dropout', type=float, default=0.05,
                      help='Fraction of steps trained with an empty caption, preserving CFG behaviour')
    data.add_argument('--center-crop', action='store_true')
    data.add_argument('--no-random-flip', dest='random_flip', action='store_false')
    data.add_argument('--repeats', type=int, default=1, help='Dataset passes per epoch')
    data.add_argument('--dataloader-num-workers', type=int, default=2)

    lora = parser.add_argument_group('lora')
    lora.add_argument('--rank', type=int, default=16, help='LoRA rank r')
    lora.add_argument('--lora-alpha', type=int, default=16)
    lora.add_argument('--lora-dropout', type=float, default=0.0)

    optim = parser.add_argument_group('optimisation')
    optim.add_argument('--batch-size', type=int, default=1)
    optim.add_argument('--num-epochs', type=int, default=40)
    optim.add_argument('--max-train-steps', type=int, default=None,
                       help='Overrides --num-epochs when set')
    optim.add_argument('--learning-rate', type=float, default=1e-4)
    optim.add_argument('--gradient-accumulation-steps', type=int, default=4)
    optim.add_argument('--gradient-checkpointing', action='store_true',
                       help='Trade ~30%% speed for a large VRAM saving')
    optim.add_argument('--lr-scheduler', default='cosine',
                       choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial',
                                'constant', 'constant_with_warmup'])
    optim.add_argument('--lr-warmup-steps', type=int, default=100)
    optim.add_argument('--adam-beta1', type=float, default=0.9)
    optim.add_argument('--adam-beta2', type=float, default=0.999)
    optim.add_argument('--adam-weight-decay', type=float, default=1e-2)
    optim.add_argument('--adam-epsilon', type=float, default=1e-8)
    optim.add_argument('--max-grad-norm', type=float, default=1.0)
    optim.add_argument('--snr-gamma', type=float, default=None,
                       help='Min-SNR loss weighting (paper suggests 5.0). Off by default')
    optim.add_argument('--mixed-precision', default='bf16', choices=['no', 'fp16', 'bf16'])
    optim.add_argument('--enable-xformers', action='store_true',
                       help='Use xformers attention (CUDA only, must be installed)')
    optim.add_argument('--allow-tf32', action='store_true', help='Faster matmuls on Ampere+')

    run = parser.add_argument_group('run')
    run.add_argument('--seed', type=int, default=42)
    run.add_argument('--checkpointing-steps', type=int, default=500)
    run.add_argument('--checkpoints-total-limit', type=int, default=3)
    run.add_argument('--resume-from-checkpoint', default=None,
                     help='Path to a checkpoint dir, or "latest". Requires --save-full-state '
                          'to have been used, since it needs the optimiser state')
    run.add_argument('--init-lora-from', default=None,
                     help='Start from an existing adapter (a pytorch_lora_weights.safetensors or '
                          'its directory). Weights only: the optimiser and LR schedule restart')
    run.add_argument('--save-full-state', action='store_true',
                     help='Also write accelerate resume state. This serialises the whole frozen '
                          'UNet (~10GB per checkpoint for SDXL) to preserve a ~90MB adapter, so '
                          'it is off by default; enable only if you need exact mid-run resume')
    run.add_argument('--logging-dir', default='logs')
    run.add_argument('--report-to', default='tensorboard', choices=['tensorboard', 'wandb', 'all', 'none'])

    evaluation = parser.add_argument_group('evaluation / early stopping')
    evaluation.add_argument('--eval-every', type=int, default=0,
                            help='Evaluate every N optimisation steps with the breed probe. '
                                 '0 disables evaluation and early stopping')
    evaluation.add_argument('--probe-path', default='outputs/breed_probe.npz',
                            help='Classifier from breed_probe.py, used as the judge')
    evaluation.add_argument('--eval-breeds',
                            default='beagle,pug,samoyed,keeshond,Bombay,Birman,Siamese,Maine_Coon',
                            help='Breeds to generate at each evaluation. Pick ones the probe is '
                                 'accurate on, or the score is mostly judge error')
    evaluation.add_argument('--eval-seeds', type=int, default=2, help='Images per breed per eval')
    evaluation.add_argument('--eval-steps', type=int, default=20, help='Denoising steps when evaluating')
    evaluation.add_argument('--eval-size', type=int, default=512, help='Resolution when evaluating')
    evaluation.add_argument('--eval-guidance', type=float, default=7.0)
    evaluation.add_argument('--early-stop-patience', type=int, default=5,
                            help='Stop after this many consecutive evaluations without improvement. '
                                 '0 disables stopping but keeps evaluating')
    evaluation.add_argument('--early-stop-min-delta', type=float, default=0.002,
                            help='Improvement smaller than this does not count')
    return parser.parse_args()


def import_text_encoder_class(base_model, revision, subfolder):
    """SDXL pairs CLIPTextModel with CLIPTextModelWithProjection; read which from the config."""
    config = PretrainedConfig.from_pretrained(base_model, subfolder=subfolder, revision=revision)
    architecture = config.architectures[0]
    if architecture == 'CLIPTextModel':
        from transformers import CLIPTextModel
        return CLIPTextModel
    if architecture == 'CLIPTextModelWithProjection':
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    raise ValueError(f'Unsupported text encoder architecture: {architecture}')


def load_base_components(args, weight_dtype):
    """Load tokenizers, text encoders, VAE, UNet and the noise schedule.

    Handles both hub-style folder layouts and single-file ``.safetensors``
    checkpoints (RealVisXL, Juggernaut and most civitai downloads).
    """
    single_file = args.base_model.endswith('.safetensors') and Path(args.base_model).exists()

    if single_file:
        logger.info(f'Loading single-file checkpoint {args.base_model}')
        pipeline = StableDiffusionXLPipeline.from_single_file(args.base_model, torch_dtype=torch.float32)
        components = dict(
            tokenizer_one=pipeline.tokenizer,
            tokenizer_two=pipeline.tokenizer_2,
            text_encoder_one=pipeline.text_encoder,
            text_encoder_two=pipeline.text_encoder_2,
            vae=pipeline.vae,
            unet=pipeline.unet,
            noise_scheduler=DDPMScheduler.from_config(pipeline.scheduler.config),
        )
        del pipeline
    else:
        logger.info(f'Loading {args.base_model}')
        text_encoder_cls_one = import_text_encoder_class(args.base_model, args.revision, 'text_encoder')
        text_encoder_cls_two = import_text_encoder_class(args.base_model, args.revision, 'text_encoder_2')
        components = dict(
            tokenizer_one=AutoTokenizer.from_pretrained(
                args.base_model, subfolder='tokenizer', revision=args.revision, use_fast=False),
            tokenizer_two=AutoTokenizer.from_pretrained(
                args.base_model, subfolder='tokenizer_2', revision=args.revision, use_fast=False),
            text_encoder_one=text_encoder_cls_one.from_pretrained(
                args.base_model, subfolder='text_encoder', revision=args.revision,
                variant=args.variant),
            text_encoder_two=text_encoder_cls_two.from_pretrained(
                args.base_model, subfolder='text_encoder_2', revision=args.revision,
                variant=args.variant),
            vae=AutoencoderKL.from_pretrained(args.base_model, subfolder='vae',
                                              revision=args.revision, variant=args.variant),
            unet=UNet2DConditionModel.from_pretrained(
                args.base_model, subfolder='unet', revision=args.revision, variant=args.variant),
            noise_scheduler=DDPMScheduler.from_pretrained(args.base_model, subfolder='scheduler'),
        )

    # Swap in an fp16-safe VAE when running fp16, otherwise the latents go to NaN.
    vae_path = args.vae_path
    if vae_path is None and weight_dtype == torch.float16:
        vae_path = FP16_SAFE_VAE
        logger.info(f'fp16 requested: substituting {FP16_SAFE_VAE} to avoid VAE overflow')
    if vae_path is not None:
        components['vae'] = AutoencoderKL.from_pretrained(vae_path)

    return components


def encode_prompts(text_encoders, tokenizers, captions, device):
    """Build SDXL's two conditioning tensors from a batch of captions.

    SDXL concatenates the penultimate hidden states of both text encoders along
    the feature axis (768 + 1280 = 2048) and separately takes the pooled output
    of the second encoder. Returning the penultimate rather than final layer
    matches how SDXL was trained.
    """
    prompt_embeds_list = []
    pooled_prompt_embeds = None

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            captions,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        outputs = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
        # text_embeds is only present on the projection model (encoder two).
        pooled_prompt_embeds = outputs[0]
        prompt_embeds_list.append(outputs.hidden_states[-2])

    return torch.concat(prompt_embeds_list, dim=-1), pooled_prompt_embeds


def compute_time_ids(original_sizes, crop_top_lefts, resolution, device, dtype):
    """SDXL's micro-conditioning vector: (orig_h, orig_w, crop_top, crop_left, target_h, target_w)."""
    target_size = torch.tensor([resolution, resolution], device=device).repeat(len(original_sizes), 1)
    add_time_ids = torch.cat([
        original_sizes.to(device), crop_top_lefts.to(device), target_size,
    ], dim=1)
    return add_time_ids.to(dtype=dtype)


def compute_snr_weights(noise_scheduler, timesteps, snr_gamma):
    """Min-SNR-gamma loss weighting (arXiv:2303.09556).

    Balances the wildly different loss magnitudes across timesteps, which
    usually converges faster than uniform weighting.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus = (1.0 - alphas_cumprod[timesteps]) ** 0.5
    snr = (sqrt_alphas_cumprod / sqrt_one_minus) ** 2
    weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
    if noise_scheduler.config.prediction_type == 'v_prediction':
        return weights / (snr + 1)
    return weights / snr


def load_lora_into_unet(unet, lora_path):
    """Seed the freshly-added adapter from a previously saved one.

    Saved adapters use the diffusers key layout (``unet.<module>.lora.down``);
    PEFT expects its own (``<module>.lora_A``). Convert, then load.
    """
    path = Path(lora_path)
    if path.is_dir():
        path = path / 'pytorch_lora_weights.safetensors'
    if not path.exists():
        raise FileNotFoundError(f'No adapter at {lora_path}')

    state_dict = StableDiffusionXLPipeline.lora_state_dict(str(path))
    if isinstance(state_dict, tuple):          # older diffusers also returns network_alphas
        state_dict = state_dict[0]
    unet_state = {k.removeprefix('unet.'): v for k, v in state_dict.items() if k.startswith('unet.')}
    if not unet_state:
        raise ValueError(f'{path} contains no unet.* LoRA keys')

    incompatible = set_peft_model_state_dict(unet, convert_unet_state_dict_to_peft(unet_state))
    missing = getattr(incompatible, 'unexpected_keys', None)
    if missing:
        logger.warning(f'unexpected keys while loading adapter: {list(missing)[:5]}')
    logger.info(f'Initialised adapter from {path} ({len(unet_state)} tensors)')
    return unet


class BreedProbeEvaluator:
    """Generates a fixed prompt/seed grid and scores it with the supervised probe.

    Training loss on a diffusion model is dominated by the random timestep each
    step draws, so it cannot say whether the adapter is getting better. This asks
    the question that actually matters -- does the model produce the breed it was
    asked for -- using a classifier whose held-out accuracy is known.

    ``mean p(expected)`` is the tracked metric rather than top-1 accuracy: on a
    handful of images accuracy moves in coarse jumps, while the probability is
    continuous and reacts to partial progress.
    """

    def __init__(self, args, accelerator, pipeline_parts):
        import numpy as np

        blob = np.load(args.probe_path, allow_pickle=False)
        self.classes = [str(c) for c in blob['classes']]
        self.weight = torch.tensor(blob['weight'], device=accelerator.device)
        self.bias = torch.tensor(blob['bias'], device=accelerator.device)
        self.probe_accuracy = float(blob['test_accuracy'])

        from transformers import CLIPModel, CLIPProcessor
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(accelerator.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

        self.args = args
        self.accelerator = accelerator
        self.parts = pipeline_parts
        self.breeds = [b.strip() for b in args.eval_breeds.split(',') if b.strip()]
        unknown = [b for b in self.breeds if b not in self.classes]
        if unknown:
            raise ValueError(f'probe does not know these eval breeds: {unknown}')
        logger.info('Evaluator: %s breeds x %s seeds, judge held-out accuracy %.3f',
                    len(self.breeds), args.eval_seeds, self.probe_accuracy)

    @staticmethod
    def _prompt(breed):
        species = 'cat' if breed[:1].isupper() else 'dog'
        return f'a photo of a {breed.replace("_", " ").lower()} {species}'

    def _build_pipeline(self, unet):
        """Wrap the live components; nothing is re-loaded from disk."""
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

        pipeline = StableDiffusionXLPipeline(
            vae=self.parts['vae'],
            text_encoder=self.parts['text_encoder_one'],
            text_encoder_2=self.parts['text_encoder_two'],
            tokenizer=self.parts['tokenizer_one'],
            tokenizer_2=self.parts['tokenizer_two'],
            unet=unet,
            scheduler=DPMSolverMultistepScheduler.from_config(self.parts['scheduler_config']),
        )
        pipeline.set_progress_bar_config(disable=True)
        return pipeline

    @torch.no_grad()
    def _score(self, images, breeds):
        inputs = self.clip_processor(images=images, return_tensors='pt').to(self.accelerator.device)
        feats = self.clip.get_image_features(**inputs)
        if not torch.is_tensor(feats):
            feats = getattr(feats, 'image_embeds', None) or feats.pooler_output
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        probabilities = (feats @ self.weight.T + self.bias).softmax(-1)

        hits, target_probs = 0, []
        for row, breed in zip(probabilities, breeds):
            index = self.classes.index(breed)
            target_probs.append(float(row[index]))
            hits += int(int(row.argmax()) == index)
        return hits / len(breeds), sum(target_probs) / len(target_probs)

    @torch.no_grad()
    def evaluate(self, unet, step, save_dir=None):
        was_training = unet.training
        unet.eval()
        pipeline = self._build_pipeline(unet)
        images, breeds = [], []
        for breed in self.breeds:
            for seed in range(self.args.eval_seeds):
                generator = torch.Generator(device='cpu').manual_seed(1000 + seed)
                image = pipeline(
                    prompt=self._prompt(breed),
                    num_inference_steps=self.args.eval_steps,
                    guidance_scale=self.args.eval_guidance,
                    height=self.args.eval_size,
                    width=self.args.eval_size,
                    generator=generator,
                ).images[0]
                images.append(image)
                breeds.append(breed)
                if save_dir is not None:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    image.save(Path(save_dir) / f'{breed}_{seed:02d}.png')
        accuracy, mean_probability = self._score(images, breeds)
        unet.train(was_training)
        logger.info('eval @ step %s: breed_acc %.3f  mean p(expected) %.4f  (n=%s, judge %.3f)',
                    step, accuracy, mean_probability, len(images), self.probe_accuracy)
        return {'eval/breed_accuracy': accuracy, 'eval/breed_probability': mean_probability}


def save_lora_weights(unet, output_path):
    """Write the adapter alone, in the key layout diffusers' loaders expect."""
    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(output_path),
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )
    return Path(output_path) / 'pytorch_lora_weights.safetensors'


def prune_old_checkpoints(output_dir, limit):
    if not limit:
        return
    checkpoints = sorted(
        (p for p in Path(output_dir).glob('checkpoint-*') if p.is_dir()),
        key=lambda p: int(p.name.split('-')[1]),
    )
    for stale in checkpoints[:max(0, len(checkpoints) - limit)]:
        shutil.rmtree(stale, ignore_errors=True)
        logger.info(f'Removed old checkpoint {stale}')


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    logging_dir = output_dir / args.logging_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == 'none' else args.report_to,
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(logging_dir)),
    )

    logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s', level=logging.INFO)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(args.seed)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(accelerator.mixed_precision, torch.float32)

    components = load_base_components(args, weight_dtype)
    scheduler_config = dict(components['noise_scheduler'].config)
    tokenizer_one, tokenizer_two = components['tokenizer_one'], components['tokenizer_two']
    text_encoder_one, text_encoder_two = components['text_encoder_one'], components['text_encoder_two']
    vae, unet = components['vae'], components['unet']
    noise_scheduler = components['noise_scheduler']

    # ---- freeze everything; LoRA supplies the only trainable tensors ----
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # Under fp16 we only trust fp16 for the VAE when we substituted the fp16-safe
    # weights ourselves. A VAE the user pinned might be the stock one, which
    # overflows, so that case falls back to fp32. bf16 has the range to be fine.
    vae_dtype = torch.float32 if (weight_dtype == torch.float16 and args.vae_path) else weight_dtype
    vae.to(accelerator.device, dtype=vae_dtype)
    logger.info(f'dtypes: unet/text {weight_dtype}, vae {vae_dtype}')
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    unet.add_adapter(LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights='gaussian',
        target_modules=LORA_TARGET_MODULES,
    ))
    if args.init_lora_from:
        load_lora_into_unet(unet, args.init_lora_from)

    # LoRA params must be fp32 even under mixed precision; fp16 master weights
    # underflow at these learning rates and the adapter silently stops learning.
    lora_parameters = [p for p in unet.parameters() if p.requires_grad]
    for param in lora_parameters:
        param.data = param.data.to(torch.float32)

    trainable = sum(p.numel() for p in lora_parameters)
    total = sum(p.numel() for p in unet.parameters())
    logger.info(f'LoRA rank {args.rank}: training {trainable:,} of {total:,} UNet params '
                f'({100 * trainable / total:.2f}%)')

    if args.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info('xformers attention enabled')
        except (ModuleNotFoundError, ValueError, AttributeError) as exc:
            logger.warning(f'xformers unavailable ({exc}); using default attention')
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # ---- data ----
    dataset = ImageCaptionDataset(
        data_dir=args.data_dir,
        resolution=args.resolution,
        instance_prompt=args.instance_prompt,
        caption_dropout=args.caption_dropout,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        repeats=args.repeats,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        persistent_workers=args.dataloader_num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or args.num_epochs * steps_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler)

    evaluator = None
    if args.eval_every and accelerator.is_main_process:
        evaluator = BreedProbeEvaluator(args, accelerator, {
            'vae': vae,
            'text_encoder_one': text_encoder_one,
            'text_encoder_two': text_encoder_two,
            'tokenizer_one': tokenizer_one,
            'tokenizer_two': tokenizer_two,
            'scheduler_config': scheduler_config,
        })

    if accelerator.is_main_process and args.report_to != 'none':
        accelerator.init_trackers('sdxl_lora', config=vars(args))

    total_batch = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info('***** Training *****')
    logger.info(f'  examples = {len(dataset)} | epochs = {args.num_epochs}')
    logger.info(f'  batch/device = {args.batch_size} | accum = {args.gradient_accumulation_steps} '
                f'| effective batch = {total_batch}')
    logger.info(f'  optimisation steps = {max_train_steps} | precision = {accelerator.mixed_precision}')

    global_step = 0
    first_epoch = 0
    best_metric = float('-inf')
    best_step = None
    evals_without_improvement = 0
    stop_training = False
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if path == 'latest':
            found = sorted(output_dir.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1]))
            path = str(found[-1]) if found else None
        if path and Path(path).exists():
            accelerator.load_state(path)
            global_step = int(Path(path).name.split('-')[1])
            first_epoch = global_step // steps_per_epoch
            logger.info(f'Resumed from {path} at step {global_step}')
        else:
            logger.warning(f'No checkpoint at {args.resume_from_checkpoint}; starting fresh')

    progress = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process,
                    desc='steps')
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]

    for epoch in range(first_epoch, args.num_epochs):
        unet.train()
        running_loss = 0.0
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # 1. images -> latents (frozen VAE, no grad)
                with torch.no_grad():
                    latents = vae.encode(
                        batch['pixel_values'].to(accelerator.device, dtype=vae_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)

                # 2. sample noise and a timestep per example
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                    device=latents.device, dtype=torch.long)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. conditioning: captions plus SDXL's size/crop micro-conditioning
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = encode_prompts(
                        text_encoders, tokenizers, batch['captions'], accelerator.device)
                add_time_ids = compute_time_ids(
                    batch['original_sizes'], batch['crop_top_lefts'],
                    args.resolution, accelerator.device, weight_dtype)

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype),
                    added_cond_kwargs={
                        'text_embeds': pooled_prompt_embeds.to(dtype=weight_dtype),
                        'time_ids': add_time_ids,
                    },
                ).sample

                # 4. target depends on the schedule's parameterisation
                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f'Unsupported prediction type {noise_scheduler.config.prediction_type}')

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
                else:
                    weights = compute_snr_weights(noise_scheduler, timesteps, args.snr_gamma)
                    per_example = F.mse_loss(model_pred.float(), target.float(), reduction='none')
                    per_example = per_example.mean(dim=list(range(1, len(per_example.shape))))
                    loss = (per_example * weights).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_parameters, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += accelerator.gather(loss.repeat(args.batch_size)).mean().item()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                mean_loss = running_loss / args.gradient_accumulation_steps
                progress.set_postfix(loss=f'{mean_loss:.4f}', lr=f'{lr_scheduler.get_last_lr()[0]:.2e}')
                if args.report_to != 'none':
                    accelerator.log({'train/loss': mean_loss,
                                     'train/lr': lr_scheduler.get_last_lr()[0],
                                     'train/epoch': epoch}, step=global_step)
                running_loss = 0.0

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    checkpoint_dir = output_dir / f'checkpoint-{global_step}'
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    # Adapter first: it is the artefact that matters, and writing it
                    # before the (huge, optional) resume state means a full disk
                    # cannot cost us the trained weights.
                    saved = save_lora_weights(accelerator.unwrap_model(unet), checkpoint_dir)
                    logger.info(f'Saved {saved}')
                    if args.save_full_state:
                        accelerator.save_state(str(checkpoint_dir))
                    prune_old_checkpoints(output_dir, args.checkpoints_total_limit)

                if evaluator is not None and global_step % args.eval_every == 0:
                    metrics = evaluator.evaluate(
                        accelerator.unwrap_model(unet), global_step,
                        save_dir=output_dir / 'eval' / f'step-{global_step}')
                    if args.report_to != 'none':
                        accelerator.log(metrics, step=global_step)
                    current = metrics['eval/breed_probability']
                    if current > best_metric + args.early_stop_min_delta:
                        best_metric, best_step = current, global_step
                        evals_without_improvement = 0
                        best_dir = output_dir / 'best'
                        best_dir.mkdir(parents=True, exist_ok=True)
                        save_lora_weights(accelerator.unwrap_model(unet), best_dir)
                        (best_dir / 'best.json').write_text(json.dumps(
                            {'step': global_step, **metrics}, indent=2))
                        logger.info('new best at step %s (%.4f) -> %s', global_step, current, best_dir)
                    else:
                        evals_without_improvement += 1
                        logger.info('no improvement (%.4f vs best %.4f at step %s) - %s/%s',
                                    current, best_metric, best_step,
                                    evals_without_improvement, args.early_stop_patience)
                        if args.early_stop_patience and evals_without_improvement >= args.early_stop_patience:
                            logger.info('early stopping: %s evaluations without improvement. '
                                        'best step %s (%.4f)',
                                        evals_without_improvement, best_step, best_metric)
                            stop_training = True

                if global_step >= max_train_steps or stop_training:
                    break
        if global_step >= max_train_steps or stop_training:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final = save_lora_weights(accelerator.unwrap_model(unet), output_dir)
        logger.info(f'Training complete. Final LoRA weights: {final}')
        if best_step is not None:
            logger.info('Best by breed probability: step %s (%.4f) at %s',
                        best_step, best_metric, output_dir / 'best')
            logger.info('The "best" adapter is usually the one to ship, not the final one.')
    accelerator.end_training()


if __name__ == '__main__':
    main()
