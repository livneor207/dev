"""
Persistent best-baseline registry so runs do not overwrite the winner.

Layout:
  runs/best_baseline.json          pointer + metrics + config snapshot
  runs/best_baseline/best.pt       copy of the winning checkpoint
  runs/best_baseline/config.json   copy of the winning config
  runs/best_baseline/history/      previous winners (never deleted)
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


REGISTRY_NAME = 'best_baseline.json'
BASELINE_DIR_NAME = 'best_baseline'
HISTORY_DIR_NAME = 'history'
MONITORED_MIN = 'val_avg_total_loss'
MONITORED_MAX = 'val_pr_auc'
ARTIFACTS = (
    'best.pt',
    'config.json',
    'metrics.csv',
    'best_threshold.json',
    'highest_errors.png',
    'highest_errors.json',
    'examples_val.png',
    'curves.png',
)


def runs_root(output_dir: str | Path) -> Path:
    """`runs/` directory that contains individual experiments."""
    output_dir = Path(output_dir).resolve()
    if output_dir.name == BASELINE_DIR_NAME:
        return output_dir.parent
    if output_dir.parent.name == 'runs' or output_dir.name == 'runs':
        return output_dir if output_dir.name == 'runs' else output_dir.parent
    # Fall back to <cwd>/runs
    return Path('runs')


def registry_path(output_dir: str | Path) -> Path:
    return runs_root(output_dir) / REGISTRY_NAME


def baseline_dir(output_dir: str | Path) -> Path:
    return runs_root(output_dir) / BASELINE_DIR_NAME


def load_best_baseline(output_dir: str | Path = 'runs/default') -> Optional[dict]:
    path = registry_path(output_dir)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def print_baseline(baseline: Optional[dict]) -> None:
    print("=" * 60)
    print("BEST BASELINE")
    print("=" * 60)
    if baseline is None:
        print("  None yet — this run will create the first baseline if it finishes.")
        print("=" * 60)
        return
    print(f"  run_dir:     {baseline.get('run_dir')}")
    print(f"  checkpoint:  {baseline.get('checkpoint')}")
    print(f"  updated_at:  {baseline.get('updated_at')}")
    metrics = baseline.get('metrics', {})
    for key in (MONITORED_MIN, 'val_accuracy', 'val_f1', 'val_angle_mae_deg',
                'val_center_mae', 'val_axes_mae', 'best_fbeta', 'best_threshold'):
        if key in metrics:
            print(f"  {key}: {metrics[key]}")
    config = baseline.get('config', {})
    if config:
        print("  config:")
        for key in ('backbone', 'depth', 'cls_loss', 'reg_loss', 'lr',
                    'lambda_cls', 'lambda_center', 'lambda_radii', 'lambda_angle',
                    'fine_grained', 'optimizer'):
            if key in config:
                print(f"    {key}: {config[key]}")
    print("=" * 60)


def _metric_value(metrics: dict, max_opt: bool) -> Optional[float]:
    key = MONITORED_MAX if max_opt else MONITORED_MIN
    value = metrics.get(key)
    return None if value is None else float(value)


def is_better(current: dict, baseline: Optional[dict], max_opt: bool = False) -> bool:
    """True if current should replace the stored baseline."""
    current_val = _metric_value(current, max_opt)
    if current_val is None:
        return False
    if baseline is None:
        return True
    best_val = _metric_value(baseline.get('metrics', {}), max_opt)
    if best_val is None:
        return True
    if max_opt:
        return current_val > best_val
    return current_val < best_val


def _archive_baseline(best_dir: Path) -> Optional[Path]:
    """Copy current baseline artifacts into history/. Never deletes them."""
    if not (best_dir / 'best.pt').exists():
        return None
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dest = best_dir / HISTORY_DIR_NAME / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS + ('baseline_snapshot.json',):
        src = best_dir / name
        if src.exists() and src.is_file():
            shutil.copy2(src, dest / name)
    return dest


def _copy_artifacts(run_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        src = run_dir / name
        if src.exists() and src.is_file():
            shutil.copy2(src, dest_dir / name)


def snapshot_config(config: dict) -> dict:
    """Keep a small, readable hyperparameter snapshot."""
    keys = (
        'backbone', 'grayscale', 'canonical_axes', 'depth', 'base_filters',
        'kernel_size', 'dropout',
        'head_hidden_dim', 'fine_grained', 'auto_tune',
        'cls_loss', 'reg_loss', 'smooth_l1_beta',
        'lambda_cls', 'lambda_center', 'lambda_radii', 'lambda_angle',
        'optimizer', 'lr', 'weight_decay', 'batch_size', 'epochs', 'seed',
        'output_dir',
    )
    return {k: config[k] for k in keys if k in config}


def maybe_update_best_baseline(
    run_dir: str | Path,
    metrics: dict,
    config: dict,
    max_opt: bool = False,
) -> dict:
    """
    Promote this run to the global baseline if it beats the stored winner.

    Previous baseline artifacts are archived, never deleted.
    """
    run_dir = Path(run_dir)
    previous = load_best_baseline(run_dir)
    promoted = is_better(metrics, previous, max_opt=max_opt)

    result = {
        'promoted': promoted,
        'previous': previous,
        'reason': '',
    }

    if not promoted:
        current_val = _metric_value(metrics, max_opt)
        best_val = _metric_value(previous['metrics'], max_opt) if previous else None
        result['reason'] = (
            f"Did not beat baseline ({MONITORED_MAX if max_opt else MONITORED_MIN}: "
            f"{current_val} vs {best_val}). Baseline left untouched."
        )
        print(result['reason'])
        return result

    best_dir = baseline_dir(run_dir)
    archived = _archive_baseline(best_dir)
    _copy_artifacts(run_dir, best_dir)

    record: dict[str, Any] = {
        'run_dir': str(run_dir),
        'checkpoint': str(best_dir / 'best.pt'),
        'config_path': str(best_dir / 'config.json'),
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'max_opt': max_opt,
        'monitored_metric': MONITORED_MAX if max_opt else MONITORED_MIN,
        'metrics': metrics,
        'config': snapshot_config(config),
        'archived_previous_to': str(archived) if archived else None,
    }

    dest_snapshot = best_dir / 'baseline_snapshot.json'
    dest_snapshot.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_snapshot, 'w') as f:
        json.dump(record, f, indent=2)
    with open(registry_path(run_dir), 'w') as f:
        json.dump(record, f, indent=2)

    result['reason'] = (
        f"Promoted {run_dir} to best baseline "
        f"({record['monitored_metric']}={_metric_value(metrics, max_opt)})."
    )
    if archived:
        result['reason'] += f" Previous archived at {archived}."
    print(result['reason'])
    return result


def metrics_from_training(
    results_df,
    train_meta: dict,
    max_opt: bool = False,
) -> dict:
    """Pull the best-epoch validation metrics from the training report."""
    if results_df is None or len(results_df) == 0:
        return {}
    if max_opt and 'val_pr_auc' in results_df.columns:
        idx = results_df['val_pr_auc'].idxmax()
    else:
        idx = results_df['val_avg_total_loss'].idxmin()
    row = results_df.loc[idx]
    metrics = {}
    for col in results_df.columns:
        if col.startswith('val_'):
            value = row[col]
            metrics[col] = float(value) if hasattr(value, 'item') or isinstance(value, (int, float)) else value
    metrics['best_epoch'] = int(row['epoch'])
    metrics['best_threshold'] = float(train_meta.get('best_threshold', 0.5))
    metrics['best_fbeta'] = float(train_meta.get('best_fbeta', 0.0))
    return metrics


if __name__ == '__main__':
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_a = tmp / 'runs' / 'exp_a'
        run_b = tmp / 'runs' / 'exp_b'
        run_a.mkdir(parents=True)
        run_b.mkdir(parents=True)
        for run in (run_a, run_b):
            (run / 'best.pt').write_bytes(b'ckpt')
            (run / 'config.json').write_text('{"lr": 0.001}')

        first = maybe_update_best_baseline(
            run_a,
            metrics={'val_avg_total_loss': 1.2, 'val_angle_mae_deg': 40.0},
            config={'lr': 0.001, 'backbone': 'cnn'},
        )
        assert first['promoted']
        assert load_best_baseline(run_a)['metrics']['val_avg_total_loss'] == 1.2

        skipped = maybe_update_best_baseline(
            run_b,
            metrics={'val_avg_total_loss': 1.5, 'val_angle_mae_deg': 30.0},
            config={'lr': 0.002, 'backbone': 'mlp'},
        )
        assert not skipped['promoted']
        assert load_best_baseline(run_b)['metrics']['val_avg_total_loss'] == 1.2
        assert (tmp / 'runs' / 'best_baseline' / 'best.pt').exists()

        better = maybe_update_best_baseline(
            run_b,
            metrics={'val_avg_total_loss': 0.9, 'val_angle_mae_deg': 20.0},
            config={'lr': 0.002, 'backbone': 'mlp'},
        )
        assert better['promoted']
        history = list((tmp / 'runs' / 'best_baseline' / 'history').iterdir())
        assert history, 'previous baseline must be archived, not deleted'
        assert load_best_baseline(run_b)['metrics']['val_avg_total_loss'] == 0.9
        print('Baseline registry tests passed.')
