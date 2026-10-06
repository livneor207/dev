import math

import torch
from torch import nn
from torch.nn import functional as F

from defusion_model.diffusion.unet import NULL_CLASS


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def extract(schedule, timestep, x_shape):
    gathered = schedule.gather(-1, timestep)
    return gathered.reshape(timestep.shape[0], *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=1000, schedule='cosine', beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps
        self.schedule_name = schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        if schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif schedule == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
        else:
            raise ValueError(f'Unsupported schedule: {schedule}')
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start, timestep, noise):
        return (
            extract(self.sqrt_alphas_cumprod, timestep, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timestep, x_start.shape) * noise
        )

    def predict_x0_from_eps(self, x_t, timestep, eps):
        return (
            extract(1.0 / self.sqrt_alphas_cumprod, timestep, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod, timestep, x_t.shape) * eps
        )

    def p_losses(self, model, x_start, timestep, noise, class_label=None, noise_loss=None,
                 label_dropout=0.0):
        """Noise-prediction loss; ``label_dropout`` replaces labels with the
        unconditional token so the same weights learn both scores for CFG."""
        if noise_loss is None:
            noise_loss = nn.MSELoss()
        if class_label is not None and label_dropout > 0.0:
            drop = torch.rand(class_label.shape, device=class_label.device) < label_dropout
            class_label = torch.where(drop, torch.full_like(class_label, NULL_CLASS), class_label)
        x_t = self.q_sample(x_start, timestep, noise)
        predicted_noise = model(noisy_image=x_t, timestep=timestep, class_label=class_label)
        return noise_loss(predicted_noise, noise)

    def predict_noise(self, model, x_t, timestep, class_label=None, cfg_scale=1.0):
        """Noise prediction, optionally sharpened by classifier-free guidance.

        ``cfg_scale <= 1`` is a plain conditional pass; above that the conditional
        and unconditional predictions are extrapolated apart.
        """
        if class_label is None or cfg_scale <= 1.0:
            return model(noisy_image=x_t, timestep=timestep, class_label=class_label)
        null_label = torch.full_like(class_label, NULL_CLASS)
        batch = torch.cat([x_t, x_t], dim=0)
        batch_timestep = torch.cat([timestep, timestep], dim=0)
        batch_label = torch.cat([class_label, null_label], dim=0)
        predicted = model(noisy_image=batch, timestep=batch_timestep, class_label=batch_label)
        conditional, unconditional = predicted.chunk(2, dim=0)
        return unconditional + cfg_scale * (conditional - unconditional)

    @torch.no_grad()
    def p_sample(self, model, x_t, timestep, class_label=None, cfg_scale=1.0, clip_denoised=True):
        predicted_noise = self.predict_noise(model, x_t, timestep, class_label=class_label, cfg_scale=cfg_scale)
        x_start = self.predict_x0_from_eps(x_t, timestep, predicted_noise)
        if clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)
        mean = (
            extract(self.posterior_mean_coef1, timestep, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, timestep, x_t.shape) * x_t
        )
        if timestep[0] == 0:
            return mean
        noise = torch.randn_like(x_t)
        log_variance = extract(self.posterior_log_variance_clipped, timestep, x_t.shape)
        return mean + torch.exp(0.5 * log_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, class_label=None, device=None, cfg_scale=1.0,
                      clip_denoised=True):
        device = device or next(model.parameters()).device
        x = torch.randn(shape, device=device)
        for t in range(self.timesteps - 1, -1, -1):
            timestep = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, timestep, class_label=class_label, cfg_scale=cfg_scale,
                              clip_denoised=clip_denoised)
        return x

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, class_label=None, device=None, sample_steps=50, eta=0.0,
                         cfg_scale=1.0, clip_denoised=True):
        """Deterministic DDIM sampling.

        ``clip_denoised`` is not cosmetic: sampling starts where
        ``sqrt(alphas_cumprod)`` is ~0.03, so it divides the noise error by ~32
        and an unclipped x0 estimate leaves [-1, 1] by an order of magnitude and
        never comes back, which renders as saturated speckle.
        """
        device = device or next(model.parameters()).device
        step_size = max(self.timesteps // sample_steps, 1)
        times = list(range(0, self.timesteps, step_size))
        times_next = [-1] + times[:-1]
        x = torch.randn(shape, device=device)
        for t, t_next in zip(reversed(times), reversed(times_next)):
            timestep = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = self.predict_noise(model, x, timestep, class_label=class_label,
                                                 cfg_scale=cfg_scale)
            alpha = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
            x0 = (x - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
            if clip_denoised:
                x0 = x0.clamp(-1.0, 1.0)
            if t_next < 0:
                x = x0
                continue
            sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next))
            dir_xt = torch.sqrt(1 - alpha_next - sigma ** 2) * predicted_noise
            x = torch.sqrt(alpha_next) * x0 + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x

    def config_dict(self):
        return {
            'timesteps': self.timesteps,
            'schedule': self.schedule_name,
            'beta_start': self.beta_start,
            'beta_end': self.beta_end,
        }
