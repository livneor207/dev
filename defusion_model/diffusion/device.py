import logging

import torch


def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f'Checkpoint at {path} must be a dict')
    if 'state_dict' not in payload:
        raise ValueError(f'Checkpoint at {path} is missing state_dict')
    return payload


def load_compatible_weights(model, checkpoint_path, use_ema=True, device='cpu'):
    """Initialise ``model`` from a checkpoint, keeping only tensors that still fit.

    Lets a run start from earlier weights even when the architecture moved on -- most
    importantly when the class count changed, since ``class_embed`` is sized by it.
    Mismatched tensors keep their fresh initialisation instead of failing the load.
    """
    payload = load_checkpoint(checkpoint_path, device=device)
    key = 'ema_state_dict' if use_ema and payload.get('ema_state_dict') else 'state_dict'
    source = payload[key]
    own = model.state_dict()
    loaded, skipped = [], []
    for name, tensor in source.items():
        if name in own and own[name].shape == tensor.shape:
            own[name] = tensor
            loaded.append(name)
        else:
            shape = tuple(tensor.shape)
            target = tuple(own[name].shape) if name in own else None
            skipped.append((name, shape, target))
    model.load_state_dict(own)
    n_loaded = sum(own[name].numel() for name in loaded)
    logging.info('initialised %s/%s tensors (%s params) from %s [%s]',
                 len(loaded), len(source), f'{n_loaded:,}', checkpoint_path, key)
    for name, shape, target in skipped:
        logging.info('  kept fresh init for %s: checkpoint %s vs model %s', name, shape, target)
    return model


def load_model(model, model_path, device='cpu', use_ema=True):
    payload = load_checkpoint(model_path, device=device)
    key = 'ema_state_dict' if use_ema and payload.get('ema_state_dict') else 'state_dict'
    model.load_state_dict(payload[key])
    logging.info('Loaded %s from %s', key, model_path)
    return model.to(device)
