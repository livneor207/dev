import logging
from pathlib import Path

from defusion_model.diffusion.device import load_model
from defusion_model.train.helpers import (
    ExponentialMovingAverage,
    add_epoch_results,
    check_if_model_improved,
    declare_early_stopping_condition,
    forward_all_dataset,
    initial_best_result,
    maybe_save_epoch_samples,
    print_epoch_results,
    save_checkpoint,
    scheduler_step,
    update_epoch_result,
    write_training_report,
)


def train_loop(model, optimizer, train_data_loader, val_data_loader, diffusion,
               noise_loss=None, num_epochs=50, device='mps', scheduler=None,
               model_path='./model.pth', training_summary_df_path='./model.csv',
               sample_dir='./samples', max_opt=False, ema_decay=0.9999,
               sample_every=1, max_grad_norm=1.0, label_dropout=0.0, cfg_scale=1.0,
               sample_classes=None, sample_steps=50, patience_epochs=0, save_last_every=25):
    """Train a diffusion model, tracking both the best-val and the latest weights.

    ``patience_epochs=0`` disables early stopping, which is the right default here:
    validation noise-MSE is a poor proxy for sample quality, so a small val split
    plateaus (noisily) long before samples stop improving, and stopping on it
    discards a still-improving model. The best-val checkpoint is still written to
    ``model_path``; the latest weights go to ``model_last.pth`` beside it, and that
    is usually the one worth sampling from.
    """
    results_list = []
    patience = 0
    max_patience = patience_epochs if patience_epochs else None
    last_model_path = str(Path(model_path).with_name(Path(model_path).stem + '_last.pth'))
    best_model_score = initial_best_result(max_opt)
    single_label = train_data_loader.dataset.single_label
    class_names = getattr(train_data_loader.dataset, 'class_names', None)
    ema = ExponentialMovingAverage(model, decay=ema_decay)
    sample_dir = Path(sample_dir)
    extra_state = {
        'model_config': model.config_dict(),
        'diffusion_config': diffusion.config_dict(),
        'class_names': class_names,
        'ema': ema,
    }

    for epoch in range(num_epochs):

        train_acc, train_score, train_loss, train_pred, train_y_true = (
            forward_all_dataset(model, train_data_loader, diffusion,
                                noise_loss=noise_loss, device=device,
                                optimizer=optimizer, ema=ema, max_grad_norm=max_grad_norm,
                                label_dropout=label_dropout))

        # run a validation step
        val_acc, val_score, val_loss, val_pred, val_y_true = (
            forward_all_dataset(model, val_data_loader, diffusion,
                                noise_loss=noise_loss, device=device))

        # update model score base optimization task
        current_val = update_epoch_result(val_loss, val_score, max_opt=max_opt)

        # print current result
        print_epoch_results(epoch, train_acc, train_loss, train_score,
                            val_acc, val_loss, val_score, single_label=single_label)

        # accumulate epoch results
        add_epoch_results(results_list, epoch, train_acc, train_score, train_loss,
                          val_acc, val_score, val_loss)

        # save results so far
        train_results_df = write_training_report(
            results_list, training_summary_df_path=training_summary_df_path)

        # update learning rate
        scheduler_step(scheduler, val_loss)

        # save a sample grid from the EMA weights
        if sample_every and epoch % sample_every == 0:
            maybe_save_epoch_samples(
                model, diffusion, epoch, sample_dir, device,
                image_size=train_data_loader.dataset.image_size,
                num_classes=model.num_classes, ema=ema, class_names=class_names,
                sample_classes=sample_classes, cfg_scale=cfg_scale, sample_steps=sample_steps)

        # maximization/minimization and validation score improved
        extra_state['epoch'] = epoch
        extra_state['val_loss'] = val_loss
        best_model_score, patience = check_if_model_improved(
            best_model_score, current_val, max_opt, model, model_path, patience,
            extra_state=extra_state)

        is_final_epoch = epoch == num_epochs - 1
        if save_last_every and (epoch % save_last_every == 0 or is_final_epoch):
            save_checkpoint(model, last_model_path, extra_state)
            logging.info('saved latest checkpoint to %s (epoch %s)', last_model_path, epoch)

        if max_patience is not None and patience > max_patience:
            declare_early_stopping_condition(max_patience)
            break

    train_results_df = write_training_report(
        results_list, training_summary_df_path=training_summary_df_path)
    save_checkpoint(model, last_model_path, extra_state)
    logging.info('final weights saved to %s', last_model_path)
    model = load_model(model, model_path, device=device, use_ema=True)

    return model, train_results_df
