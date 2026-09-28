import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from src.orientation import sha256
from predict import read_ids, resolve_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    ids = read_ids('sample_submission.csv')
    paths = [Path('sample_submission.csv'), *resolve_images('test/images', ids)]
    manifest = {p.as_posix(): sha256(p) for p in paths}
    target = Path('reports/input_manifest.json')
    if args.verify:
        if manifest != json.loads(target.read_text(encoding='utf-8')):
            raise ValueError('Original input files changed')
        print(f'OK: all {len(paths)} original files unchanged')
    else:
        # Сохраняем исходный эталон: повторный запуск не должен его затереть.
        if target.exists():
            raise FileExistsError('Manifest already exists; use --verify')
        target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'Recorded SHA256 for {len(paths)} original files')

if __name__ == '__main__':
    main()
