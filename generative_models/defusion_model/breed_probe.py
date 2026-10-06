"""Train a breed classifier on the real dataset, to judge generated images.

CLIP zero-shot brings its own biases, and similarity-to-the-training-set rewards
reproducing whatever is wrong with that set. This instead fits a supervised
classifier on the labelled data and reports its accuracy on a held-out split, so
its verdict on generated images comes with a known error rate.

A linear probe on frozen CLIP features is the right size of tool here: strong on
this dataset, trains in seconds, and cannot overfit the way a full fine-tune
could on ~180 images per class.

    python breed_probe.py fit                                  # train + report test accuracy
    python breed_probe.py score --images outputs/eval/base     # classify generated images
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path('data/lora_pets37')
PROBE_PATH = Path('outputs/breed_probe.npz')
CLIP_MODEL = 'openai/clip-vit-base-patch32'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(description='Supervised breed judge')
    sub = parser.add_subparsers(dest='command', required=True)

    fit = sub.add_parser('fit', help='Train the probe and report held-out accuracy')
    fit.add_argument('--data-dir', default=str(DATA_DIR))
    fit.add_argument('--probe-path', default=str(PROBE_PATH))
    fit.add_argument('--test-fraction', type=float, default=0.2)
    fit.add_argument('--epochs', type=int, default=400)
    fit.add_argument('--lr', type=float, default=1.0)
    fit.add_argument('--weight-decay', type=float, default=1e-4)
    fit.add_argument('--batch-size', type=int, default=64, help='Batch size for CLIP embedding')
    fit.add_argument('--seed', type=int, default=0)
    fit.add_argument('--device', default=None)

    score = sub.add_parser('score', help='Classify a folder of generated images')
    score.add_argument('--images', required=True, help='Folder of generated images')
    score.add_argument('--probe-path', default=str(PROBE_PATH))
    score.add_argument('--expect-from-filename', action='store_true', default=True,
                       help='Read the true breed from the leading part of each filename')
    score.add_argument('--device', default=None)
    return parser.parse_args()


def resolve_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    return model, processor


def as_tensor(output):
    if torch.is_tensor(output):
        return output
    for attribute in ('image_embeds', 'pooler_output'):
        value = getattr(output, attribute, None)
        if torch.is_tensor(value):
            return value
    raise TypeError(f'Cannot read embeddings from {type(output).__name__}')


@torch.no_grad()
def embed_images(paths, model, processor, device, batch_size=64):
    """L2-normalised CLIP image features for a list of paths."""
    features = []
    for start in range(0, len(paths), batch_size):
        batch = [Image.open(p).convert('RGB') for p in paths[start:start + batch_size]]
        inputs = processor(images=batch, return_tensors='pt').to(device)
        feats = as_tensor(model.get_image_features(**inputs)).float()
        features.append((feats / feats.norm(dim=-1, keepdim=True)).cpu())
        if start and start % (batch_size * 10) == 0:
            logger.info('  embedded %s/%s', start, len(paths))
    return torch.cat(features) if features else torch.empty(0)


def breed_of(path):
    return Path(path).stem.rsplit('_', 1)[0]


def fit(args):
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f'No images in {data_dir}')
    labels_text = [breed_of(p) for p in paths]
    classes = sorted(set(labels_text))
    class_index = {name: i for i, name in enumerate(classes)}
    labels = torch.tensor([class_index[name] for name in labels_text])
    logger.info('%s images, %s classes', len(paths), len(classes))

    model, processor = load_clip(device)
    logger.info('embedding real images with CLIP...')
    features = embed_images(paths, model, processor, device, args.batch_size)

    # Stratified split: every breed must appear in both halves, otherwise the
    # reported accuracy is dominated by whichever breeds happened to be sampled.
    generator = np.random.RandomState(args.seed)
    train_idx, test_idx = [], []
    for index in range(len(classes)):
        rows = np.where(labels.numpy() == index)[0]
        generator.shuffle(rows)
        cut = max(1, int(round(len(rows) * args.test_fraction)))
        test_idx.extend(rows[:cut])
        train_idx.extend(rows[cut:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    logger.info('train %s / test %s', len(train_idx), len(test_idx))

    x_train = features[train_idx].to(device)
    y_train = labels[train_idx].to(device)
    x_test = features[test_idx].to(device)
    y_test = labels[test_idx].to(device)

    # Multinomial logistic regression on frozen features. This is the right
    # capacity here: strong on this dataset, trains in seconds, and cannot overfit
    # the way a full fine-tune would on ~180 images per class. LBFGS converges in
    # a single .step() call.
    probe = torch.nn.Linear(features.shape[1], len(classes)).to(device)
    optimizer = torch.optim.LBFGS(probe.parameters(), lr=args.lr, max_iter=args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(probe(x_train), y_train)
        loss += args.weight_decay * probe.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        train_acc = (probe(x_train).argmax(-1) == y_train).float().mean().item()
        logits = probe(x_test)
        test_acc = (logits.argmax(-1) == y_test).float().mean().item()
        # Per-class accuracy is what makes the judge trustworthy: a weak score on
        # a breed the probe itself only gets right 78% of the time is judge error,
        # not model failure.
        per_class = {}
        for index, name in enumerate(classes):
            mask = y_test == index
            if mask.any():
                per_class[name] = (logits[mask].argmax(-1) == index).float().mean().item()

    logger.info('train accuracy %.3f | HELD-OUT TEST ACCURACY %.3f', train_acc, test_acc)
    print(f'\nheld-out accuracy per breed (n={len(test_idx)} images):')
    for name, accuracy in sorted(per_class.items(), key=lambda kv: kv[1]):
        print(f'  {name:<24} {accuracy:.2f}')

    out = Path(args.probe_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             weight=probe.weight.detach().cpu().numpy(),
             bias=probe.bias.detach().cpu().numpy(),
             classes=np.array(classes),
             test_accuracy=np.array(test_acc),
             per_class=json.dumps(per_class))
    print(f'\nwrote {out}  (test accuracy {test_acc:.3f} -- this is how much to trust its verdicts)')


def score(args):
    device = resolve_device(args.device)
    blob = np.load(args.probe_path, allow_pickle=False)
    classes = [str(c) for c in blob['classes']]
    weight = torch.tensor(blob['weight'], device=device)
    bias = torch.tensor(blob['bias'], device=device)
    probe_accuracy = float(blob['test_accuracy'])

    folder = Path(args.images)
    paths = sorted(p for p in folder.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f'No images in {folder}')

    model, processor = load_clip(device)
    features = embed_images(paths, model, processor, device).to(device)
    logits = features @ weight.T + bias
    probabilities = logits.softmax(-1)
    predictions = logits.argmax(-1).tolist()

    correct = total = 0
    rows = []
    for path, prediction, probs in zip(paths, predictions, probabilities):
        predicted = classes[prediction]
        confidence = float(probs[prediction])
        expected = breed_of(path) if args.expect_from_filename else None
        hit = expected is not None and expected in classes and predicted == expected
        if expected in classes:
            total += 1
            correct += int(hit)
            target_prob = float(probs[classes.index(expected)])
        else:
            target_prob = float('nan')
        rows.append((path.name, expected, predicted, confidence, target_prob, hit))

    print(f'\njudge: linear probe on CLIP features, held-out accuracy {probe_accuracy:.3f}')
    print(f'{"image":<34}{"expected":<20}{"predicted":<20}{"conf":>7}{"p(expected)":>13}')
    for name, expected, predicted, confidence, target_prob, hit in rows:
        mark = '' if hit else '  <-- miss'
        print(f'{name:<34}{str(expected):<20}{predicted:<20}{confidence:>7.2f}{target_prob:>13.2f}{mark}')
    if total:
        print(f'\nbreed accuracy on these images: {correct}/{total} = {correct/total:.1%}')
        print(f'(the judge itself is {probe_accuracy:.1%} accurate on real held-out photos)')
    counts = Counter(classes[p] for p in predictions)
    print('prediction spread:', dict(counts.most_common(6)))


def main():
    args = parse_args()
    if args.command == 'fit':
        fit(args)
    else:
        score(args)


if __name__ == '__main__':
    main()
