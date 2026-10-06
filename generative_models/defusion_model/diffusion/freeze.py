import logging


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def freeze_by_substring(model, include_substrings, exclude_substrings=()):
    for name, param in model.named_parameters():
        if any(token in name for token in exclude_substrings):
            continue
        if any(token in name for token in include_substrings):
            param.requires_grad = False
            logging.info('froze %s', name)


def log_trainable_parameters(model):
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            logging.debug('trainable %s %s', name, tuple(param.shape))
    logging.info('trainable params %s / %s', trainable, total)


def freeze_unet_layers(model, policy='none'):
    if policy == 'none':
        unfreeze_all(model)
    elif policy == 'all':
        freeze_all(model)
    elif policy == 'encoder':
        unfreeze_all(model)
        freeze_by_substring(model, include_substrings=('input_conv', 'down_'))
    elif policy == 'decoder':
        unfreeze_all(model)
        freeze_by_substring(model, include_substrings=('up_', 'output_'))
    elif policy == 'time':
        unfreeze_all(model)
        freeze_by_substring(model, include_substrings=('time_embed', 'class_embed'))
    else:
        raise ValueError(f'Unsupported freeze policy: {policy}')
    log_trainable_parameters(model)
