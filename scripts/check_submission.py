import argparse
import csv
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict import read_ids
from src.orientation import sha256

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', default='submission.csv')
    parser.add_argument('--compare')
    args = parser.parse_args()
    expected = read_ids('sample_submission.csv')
    with open(args.submission, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['image_id', 'p_180']:
            raise ValueError('Wrong columns')
        rows = list(reader)
    if [r['image_id'] for r in rows] != expected or len(rows) != 20000:
        raise ValueError('Wrong ID order or row count')
    for row in rows:
        p = float(row['p_180'])
        if not math.isfinite(p) or not 0 <= p <= 1 or None in row:
            raise ValueError(f'Invalid row: {row}')
    digest = sha256(args.submission)
    if args.compare and digest != sha256(args.compare):
        raise ValueError('Reproduction differs byte-for-byte')
    print(f'OK: {len(rows)} rows; SHA256={digest}')

if __name__ == '__main__':
    main()
