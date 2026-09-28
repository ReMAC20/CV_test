import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import io
import json
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from src.orientation import OrientationModel, temperature_scale, sha256

DEV_WORDS = ['продажа', 'телефон', 'доставка', 'состояние', 'новый', 'цена', 'дом',
             'работает', 'комплект', 'скидка', 'товар', 'размер', 'Москва', 'улица',
             'new', 'price', 'delivery', 'phone', 'sale', 'model', 'product', 'home']
HOLDOUT_WORDS = ['объявление', 'велосипед', 'покупатель', 'магазин', 'автомобиль',
                 'сегодня', 'качество', 'ремонт', 'квартира', 'рублей', 'Санкт-Петербург',
                 'available', 'condition', 'warranty', 'contact', 'original', 'service']

def synthetic(split, count):
    # Разные словари и шрифты разделяют выборки до создания повёрнутых пар.
    rng = np.random.default_rng(42 if split == 'dev' else 2026)
    fonts = ['NotoSans.ttf', 'NotoSerif.ttf'] if split == 'dev' else ['RobotoMono.ttf']
    words = DEV_WORDS if split == 'dev' else HOLDOUT_WORDS
    images, labels = [], []
    for _ in range(count):
        size = int(rng.integers(14, 61))
        font = ImageFont.truetype(str(Path('assets') / rng.choice(fonts)), size)
        text = ' '.join(rng.choice(words, size=int(rng.integers(1, 6))))
        if rng.random() < .3:
            text += ' ' + str(rng.integers(1, 99999))
        if rng.random() < .25:
            text = text.upper()
        bbox = font.getbbox(text)
        padx, pady = int(rng.integers(1, 12)), int(rng.integers(1, 10))
        width, height = bbox[2] - bbox[0] + 2 * padx, bbox[3] - bbox[1] + 2 * pady
        light = rng.integers(160, 256, size=3)
        dark = rng.integers(0, 105, size=3)
        bg, fg = (light, dark) if rng.random() < .7 else (dark, light)
        arr = np.clip(bg + rng.normal(0, rng.uniform(0, 12), (height, width, 3)), 0, 255).astype('uint8')
        im = Image.fromarray(arr)
        ImageDraw.Draw(im).text((padx - bbox[0], pady - bbox[1]), text, font=font, fill=tuple(fg))
        im = im.rotate(float(rng.uniform(-5, 5)), resample=Image.Resampling.BICUBIC,
                       expand=True, fillcolor=tuple(bg))
        if rng.random() < .5:
            im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(.15, 1))))
        target_h = int(rng.integers(12, 65))
        im = im.resize((max(8, int(im.width * target_h / im.height)), target_h), Image.Resampling.BILINEAR)
        if rng.random() < .5:
            stream = io.BytesIO()
            im.save(stream, format='JPEG', quality=int(rng.integers(35, 91)))
            stream.seek(0)
            im = Image.open(stream).convert('RGB')
        upright = np.array(im)
        # Метки известны из генерации; тестовые изображения здесь не используются.
        images.extend([upright, np.rot90(upright, 2).copy()])
        labels.extend([0, 1])
    return images, np.array(labels)

def metrics(y, p):
    brier = float(np.mean((p - y) ** 2))
    return dict(brier=brier, score=1 - brier, accuracy=float(np.mean((p >= .5) == y)))

def raw_predict(model, images):
    start = time.perf_counter()
    p = np.concatenate([model.raw(images[i:i+64]) for i in range(0, len(images), 64)])
    return p, time.perf_counter() - start

def transform(raw, temperature, tta):
    p = temperature_scale(raw, temperature)
    # Соседние элементы — одна строка в ориентациях 0° и 180°.
    return (p + 1 - p.reshape(-1, 2)[:, ::-1].reshape(-1)) / 2 if tta else p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-lines', type=int, default=600)
    parser.add_argument('--holdout-lines', type=int, default=400)
    args = parser.parse_args()
    dev_images, dev_y = synthetic('dev', args.dev_lines)
    hold_images, hold_y = synthetic('holdout', args.holdout_lines)
    candidates, raw_cache = [], {}
    for name in ['lcnet_025', 'lcnet_100']:
        model = OrientationModel(name)
        model.raw(dev_images[:4])
        raw, seconds = raw_predict(model, dev_images)
        raw_cache[name] = raw
        for tta in [False, True]:
            for temp in [.75, 1.0, 1.25, 1.5, 2.0]:
                p = transform(raw, temp, tta)
                candidates.append(dict(model=name, temperature=temp, tta=tta,
                                       **metrics(dev_y, p), dev_seconds=seconds,
                                       model_bytes=(Path('models') / f'{name}.onnx').stat().st_size))
        print(name, 'finished development', flush=True)
    # Выбор только по development; при равенстве предпочитаем меньшую модель.
    best = min(candidates, key=lambda r: (r['brier'], r['model_bytes'], r['tta']))
    config = {k: best[k] for k in ['model', 'temperature', 'tta']}
    Path('config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    selected = OrientationModel(config['model'])
    # Holdout оцениваем после фиксации конфигурации.
    raw_hold, hold_seconds = raw_predict(selected, hold_images)
    hold_p = transform(raw_hold, config['temperature'], config['tta'])
    result = dict(seed_dev=42, seed_holdout=2026, dev_lines=args.dev_lines,
                  holdout_lines=args.holdout_lines, candidates=candidates, selected=config,
                  holdout=metrics(hold_y, hold_p), holdout_raw_seconds=hold_seconds,
                  caveat='Synthetic domain only; not an estimate of hidden test performance.',
                  fonts={p.name: sha256(p) for p in Path('assets').glob('*.ttf')})
    Path('reports').mkdir(exist_ok=True)
    Path('reports/validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed('reports/validation_predictions.npz', y=hold_y, p=hold_p,
                        dev_y=dev_y, **raw_cache)
    print(json.dumps(dict(selected=config, holdout=result['holdout']), indent=2))

if __name__ == '__main__':
    main()
