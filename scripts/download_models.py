import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import urllib.request
from src.orientation import MODEL_HASHES, sha256

BASE = 'https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv5/cls/'
FILES = {
    'lcnet_025': 'ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx',
    'lcnet_100': 'ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx',
}

def main():
    Path('models').mkdir(exist_ok=True)
    for name, remote in FILES.items():
        path = Path('models') / f'{name}.onnx'
        if path.exists() and sha256(path) == MODEL_HASHES[name]:
            print(f'Already verified: {path}')
            continue
        temp = path.with_suffix('.download')
        urllib.request.urlretrieve(BASE + remote, temp)
        if sha256(temp) != MODEL_HASHES[name]:
            raise ValueError(f'Unexpected download checksum: {name}')
        temp.replace(path)
        print(f'Downloaded and verified: {path}')

if __name__ == '__main__':
    main()
