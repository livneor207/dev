"""Inject images and live loss data into template.html -> pet_diffusion.html."""
import base64, io, json, sys
from pathlib import Path

import pandas as pd
from PIL import Image

REPO = Path('/Users/or.livne/PycharmProjects/defusssion_model')
SP = Path(__file__).parent  # holds report_template.html
RUN_DIR = REPO / 'outputs/overfit64'
TOTAL_EPOCHS = 600
STEPS_PER_EPOCH = 16

images = {}


def add(key, path, size=None, mode=Image.LANCZOS, quality=82):
    im = Image.open(path).convert('RGB')
    if size:
        im = im.resize(size, mode)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=quality)
    images[key] = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


T2I = REPO / 'outputs/text2image'

# hero: the photoreal Australian Shepherd (RealVisXL, 1024px, 35 steps, CFG 7)
for i in range(4):
    add(f'aussie{i}', T2I / f'aussie_realvis_{i:02d}.png', (448, 448), quality=86)
for i in range(2):
    add(f'aussie_run{i}', T2I / f'aussie_running_{i:02d}.png', (448, 448), quality=86)

# the realism upgrade, same breed, same subject: distilled turbo vs guided SDXL
add('turbo_aussie', T2I / 'australian_shepherd_01.png', (420, 420), quality=88)
add('realvis_aussie', T2I / 'aussie_realvis_01.png', (420, 420), quality=88)

# comparison breeds, both models
for name in ('samoyed', 'pug'):
    add(f'sd_{name}', T2I / f'{name}_00.png', (320, 320))
    add(f'rv_{name}', T2I / f'{name}_realvis_00.png', (320, 320), quality=86)
add('sd_tabby_cat', T2I / 'tabby_cat_00.png', (320, 320))

# the x0-clipping bug
add('bug_noise', Path('/tmp/test_clamp0_cfg3.0.png'), (512, 128), Image.NEAREST, 90)
add('bug_fixed', Path('/tmp/test_clamp1_cfg3.0.png'), (512, 128), Image.NEAREST, 90)

# from-scratch samples, once a sampling pass has produced them
for key, name in (('scratch_samoyed', 'samoyed'), ('scratch_pug', 'pug')):
    path = RUN_DIR / 'generated' / f'{name}.png'
    if path.exists():
        im = Image.open(path)
        add(key, path, (im.width * 4, im.height * 4), Image.LANCZOS, 92)

# training preview grid from the latest epoch that produced one
previews = sorted((RUN_DIR / 'samples').glob('epoch_*.png'))
if previews:
    im = Image.open(previews[-1])
    add('scratch_preview', previews[-1], (im.width * 4, im.height * 4), Image.NEAREST, 92)

df = pd.read_csv(RUN_DIR / 'training_summary.csv')
loss = {
    'epochs': df['epoch'].tolist(),
    'train': [round(v, 5) for v in df['train_loss']],
    'val': [round(v, 5) for v in df['val_loss']],
}
run = {'total_epochs': TOTAL_EPOCHS, 'steps_per_epoch': STEPS_PER_EPOCH,
       'has_scratch': 'scratch_samoyed' in images,
       'has_preview': 'scratch_preview' in images,
       'preview_epoch': int(previews[-1].stem.split('_')[1]) if previews else None}

html = (SP / 'report_template.html').read_text()
html = (html.replace('__IMAGES__', json.dumps(images))
            .replace('__LOSS__', json.dumps(loss))
            .replace('__RUN__', json.dumps(run)))
out = SP / 'report.html'
out.write_text(html)
print(f'built {out} — {out.stat().st_size/1024/1024:.2f} MB, '
      f'{len(images)} images, {len(loss["epochs"])} epochs, scratch_samples={run["has_scratch"]}')
