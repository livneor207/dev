# Manual Changes: Tversky Loss + Mask OR

Apply to `notebooks/02_training.ipynb`. If you see a JSON error, use **Kernel → Restart & Clear Output**, save, then edit.

---

## 1. Imports: Add TverskyLoss

```python
from monai.losses import DiceLoss, TverskyLoss
```

---

## 2. Config: Tversky loss

```python
loss_weights: Dict[str, float] = field(default_factory=lambda: {'dice': 0.33, 'bce': 0, 'focal': 0.33, 'tversky': 0.34})
tversky_loss_alpha: float = 0.2   # FP weight
tversky_loss_beta: float = 0.8   # FN weight (emphasizes recall)
```

---

## 3. CombinedSegLoss

**__init__**, after `self.dice = ...`:
```python
        self.tversky = TverskyLoss(sigmoid=True, alpha=cfg.tversky_loss_alpha, beta=cfg.tversky_loss_beta)
```

**forward**, before `return loss`:
```python
        if w.get('tversky', 0.0) > 0:
            loss = loss + w['tversky'] * self.tversky(logits, targets)
```

---

## 4. build_modality_mask_index: Mask OR

Replace the mask block with:
```python
        mask_map: Dict[str, List[Path]] = {}
        for m in sorted(mask_dir.glob('*.nii*')):
            stem = m.name.replace('_mask1.nii.gz', '').replace('_mask2.nii.gz', '').replace('_mask1.nii', '').replace('_mask2.nii', '')
            mask_map.setdefault(stem, []).append(m)

        for cid, paths in mod_paths.items():
            if cid not in mask_map or not all(mod in paths for mod in modalities):
                continue
            masks = sorted(mask_map[cid], key=lambda p: p.name)
            row: Dict[str, str] = {'case_id': cid, 'patient_id': cid.split('_')[0], 'label': str(masks[0])}
            row['label_2'] = str(masks[1]) if len(masks) > 1 else str(masks[0])
            for i, mod in enumerate(modalities):
                row[f'image_{i+1}'] = str(paths[mod])
            rows.append(row)
```

---

## 5. to_data_dicts: Include label_2

```python
    return [
        {**{k: getattr(r, k) for k in image_keys}, 'label': r.label, 'label_2': r.label_2, 'case_id': r.case_id}
        for r in df.itertuples(index=False)
    ]
```

---

## 6. get_transforms: Add _merge_labels_or

Add this function before `_make_concat_modalities`:
```python
def _merge_labels_or(data: Dict[str, Any]) -> Dict[str, Any]:
    """OR label | label_2 for combined mask (rater1 | rater2)."""
    d = dict(data)
    if 'label_2' not in d:
        return d
    a1 = (np.asarray(d['label']) > 0).astype(np.float32)
    a2 = (np.asarray(d['label_2']) > 0).astype(np.float32)
    d['label'] = np.maximum(a1, a2)
    del d['label_2']
    return d
```

In `get_transforms`, change:
```python
    load_keys = image_keys + ['label', 'label_2']
    common = [
        LoadImaged(keys=load_keys),
        _merge_labels_or,
        EnsureChannelFirstd(keys=image_keys + ['label']),
        Orientationd(keys=image_keys + ['label'], axcodes='RAS'),
        _make_concat_modalities(n),
        ...
    ]
```
