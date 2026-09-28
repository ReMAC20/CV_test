"""Create submission in exactly the order and ID format of the supplied template."""
import argparse
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from src.orientation import OrientationModel, read_rgb, sha256

def read_ids(path):
    # Порядок и формат идентификаторов берём из шаблона.
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['image_id', 'p_180']:
            raise ValueError('Template must contain exactly image_id,p_180')
        ids = [row['image_id'] for row in reader]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('IDs must be nonempty and unique')
    return ids

def resolve_images(directory, ids):
    files = list(Path(directory).glob('*'))
    index = {}
    for path in files:
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}:
            if path.stem in index:
                raise ValueError(f'Ambiguous image stem: {path.stem}')
            index[path.stem] = path
    paths = []
    for image_id in ids:
        if Path(image_id).name != image_id or '/' in image_id or '\\' in image_id:
            raise ValueError('Image IDs must be filenames, not paths')
        key = Path(image_id).stem
        if key not in index:
            raise FileNotFoundError(image_id)
        paths.append(index[key])
    return paths

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', default='test/images')
    parser.add_argument('--sample', default='sample_submission.csv')
    parser.add_argument('--output', default='submission.csv')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--report', default='reports/inference.json')
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch-size and threads must be positive')
    output = Path(args.output).resolve()
    protected = Path(args.images).resolve()
    # Не допускаем перезаписи шаблона или исходных изображений.
    if output == Path(args.sample).resolve() or protected == output or protected in output.parents:
        raise ValueError('Refusing to write into original data')
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    ids = read_ids(args.sample)
    paths = resolve_images(args.images, ids)
    model = OrientationModel(config['model'], args.model_dir, args.threads)
    # Прогрев сессии не включаем в измерение скорости.
    model.predict([read_rgb(paths[0])], config['temperature'], config['tta'])
    start = time.perf_counter()
    probabilities = []
    for i in range(0, len(paths), args.batch_size):
        images = [read_rgb(p) for p in paths[i:i + args.batch_size]]
        probabilities.extend(model.predict(images, config['temperature'], config['tta']).tolist())
        if i % (args.batch_size * 20) == 0:
            print(f'{min(i + args.batch_size, len(paths))}/{len(paths)}', flush=True)
    elapsed = time.perf_counter() - start
    p = np.array(probabilities)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probability')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Заменяем итоговый CSV только после успешной записи всех строк.
    temporary = output.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['image_id', 'p_180'])
        writer.writerows((image_id, f'{value:.8f}') for image_id, value in zip(ids, p))
    temporary.replace(output)
    report = dict(rows=len(ids), seconds=elapsed, images_per_second=len(ids)/elapsed,
                  mean_p180=float(p.mean()), uncertain_fraction=float(np.mean((p > .25) & (p < .75))),
                  config=config, batch_size=args.batch_size, threads=args.threads,
                  python=platform.python_version(), platform=platform.platform(),
                  onnxruntime=ort.__version__, numpy=np.__version__,
                  submission_sha256=sha256(output), sample_sha256=sha256(args.sample))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
