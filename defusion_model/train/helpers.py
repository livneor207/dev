import logging

import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model):
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def state_dict(self):
        return {name: value.detach().cpu() for name, value in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {name: value.clone() for name, value in state.items()}


def forward_all_dataset(model, data_loader, diffusion, noise_loss=None, device='cpu',
                        optimizer=None, ema=None, max_grad_norm=1.0, label_dropout=0.0):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_samples = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for samples_dict, class_label in data_loader:
            images = samples_dict['image'].to(device, dtype=torch.float32)
            class_label = class_label.to(device)
            batch_size = images.shape[0]
            timestep = torch.randint(0, diffusion.timesteps, (batch_size,), device=device, dtype=torch.long)
            noise = torch.randn_like(images)
            label_arg = class_label if model.num_classes > 0 else None
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            loss = diffusion.p_losses(
                model, images, timestep, noise, class_label=label_arg, noise_loss=noise_loss,
                label_dropout=label_dropout if is_train else 0.0,
            )
            if is_train:
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
            total_loss += float(loss.item()) * batch_size
            n_samples += batch_size
    mean_loss = total_loss / max(n_samples, 1)
    return 0.0, -mean_loss, mean_loss, torch.empty(0), torch.empty(0)


def initial_best_result(max_opt):
    return float('-inf') if max_opt else float('inf')


def update_epoch_result(val_loss, val_score, max_opt=True):
    return val_score if max_opt else val_loss


def print_epoch_results(epoch, train_acc, train_loss, train_score, val_acc, val_loss, val_score,
                        single_label=True):
    logging.info(
        'epoch %s train_loss=%.5f val_loss=%.5f train_score=%.5f val_score=%.5f',
        epoch, train_loss, val_loss, train_score, val_score,
    )


def add_epoch_results(results_list, epoch, train_acc, train_score, train_loss, val_acc, val_score, val_loss):
    results_list.append({
        'epoch': epoch,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_score': train_score,
        'val_score': val_score,
        'train_loss': train_loss,
        'val_loss': val_loss,
    })


def write_training_report(results_list, training_summary_df_path='./model.csv'):
    train_results_df = pd.DataFrame(results_list)
    train_results_df.to_csv(training_summary_df_path, index=False)
    return train_results_df


def scheduler_step(scheduler, val_loss):
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


def save_checkpoint(model, model_path, extra_state=None):
    extra_state = extra_state or {}
    payload = {
        'state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
        'model_config': extra_state.get('model_config'),
        'diffusion_config': extra_state.get('diffusion_config'),
        'class_names': extra_state.get('class_names'),
        'epoch': extra_state.get('epoch'),
        'val_loss': extra_state.get('val_loss'),
    }
    ema = extra_state.get('ema')
    if ema is not None:
        payload['ema_state_dict'] = ema.state_dict()
    torch.save(payload, model_path)
    return model_path


def check_if_model_improved(best_model_score, current_val, max_opt, model, model_path, patience,
                           extra_state=None):
    improved = current_val > best_model_score if max_opt else current_val < best_model_score
    if not improved:
        return best_model_score, patience + 1
    save_checkpoint(model, model_path, extra_state)
    logging.info('saved improved checkpoint to %s (monitored=%.5f)', model_path, current_val)
    return current_val, 0


def declare_early_stopping_condition(max_patience):
    logging.info('early stopping triggered after %s epochs without improvement', max_patience)


def select_sample_classes(class_names, num_classes, sample_classes=None):
    """Pick which classes an epoch preview renders.

    With 37 breeds a full sweep would dominate epoch time, so previews cover an
    evenly spaced subset unless specific class names are requested.
    """
    if num_classes == 0:
        return [None]
    if sample_classes:
        wanted = [name.lower() for name in sample_classes]
        lookup = {name.lower(): index for index, name in enumerate(class_names or [])}
        missing = [name for name in wanted if name not in lookup]
        if missing:
            raise ValueError(f'Unknown sample classes {missing}; available {sorted(lookup)}')
        return [lookup[name] for name in wanted]
    if num_classes <= 4:
        return list(range(num_classes))
    step = num_classes / 4.0
    return sorted({int(index * step) for index in range(4)})


def maybe_save_epoch_samples(model, diffusion, epoch, sample_dir, device, image_size, num_classes,
                             ema=None, n_per_class=4, sample_steps=50, class_names=None,
                             sample_classes=None, cfg_scale=1.0):
    sample_dir.mkdir(parents=True, exist_ok=True)
    backup = None
    if ema is not None:
        backup = {name: param.detach().clone() for name, param in model.named_parameters()}
        ema.copy_to(model)
    was_training = model.training
    model.eval()
    images = []
    labels = []
    class_ids = select_sample_classes(class_names, num_classes, sample_classes)
    with torch.no_grad():
        for class_id in class_ids:
            if class_id is None:
                class_label = None
            else:
                class_label = torch.full((n_per_class,), class_id, device=device, dtype=torch.long)
            batch = diffusion.ddim_sample_loop(
                model,
                shape=(n_per_class, 3, image_size, image_size),
                class_label=class_label,
                device=device,
                sample_steps=sample_steps,
                cfg_scale=cfg_scale,
            )
            images.append(batch)
            if class_id is not None:
                name = class_names[class_id] if class_names else str(class_id)
                labels.append(name)
    grid = torch.cat(images, dim=0)
    grid = ((grid.clamp(-1, 1) + 1.0) * 0.5)
    save_path = sample_dir / f'epoch_{epoch:04d}.png'
    save_image(grid, save_path, nrow=n_per_class)
    logging.info('wrote samples %s (rows: %s)', save_path, ', '.join(labels) if labels else 'unconditional')
    if backup is not None:
        for name, param in model.named_parameters():
            param.data.copy_(backup[name])
    model.train(was_training)
