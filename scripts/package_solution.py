import json
import zipfile
from pathlib import Path

# Явный список исключает исходные данные и локальное окружение из архива.
ROOT_FILES = ['README.md', 'submission.csv', 'predict.py', 'config.json',
              'requirements.txt', 'requirements-lock.txt', '.gitignore']
DIRECTORIES = ['src', 'scripts', 'tests', 'models', 'assets', '.vscode']
REPORTS = ['validation.json', 'inference.json', 'reproduction_inference.json',
           'reproducibility.json', 'input_manifest.json', 'data_summary.json', 'environment-freeze.txt']

def main():
    files = [Path(p) for p in ROOT_FILES] + [Path('reports') / p for p in REPORTS]
    for folder in DIRECTORIES:
        files.extend(p for p in Path(folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not json.loads(Path('reports/reproducibility.json').read_text())['byte_identical']:
        raise ValueError('Reproduction has not passed')
    target = Path('dist/text-orientation-solution.zip')
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, path.as_posix())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive failed CRC verification')
        names = archive.namelist()
        if any(n.startswith(('test/', '.venv/', '.wheels/')) or n == 'sample_submission.csv' for n in names):
            raise ValueError('Protected input or environment in archive')
    print(f'{target}: {len(files)} files, {target.stat().st_size:,} bytes; CRC OK')

if __name__ == '__main__':
    main()
