from pathlib import Path
import hashlib
import cv2
import numpy as np
import onnxruntime as ort

MODEL_HASHES = {
    'lcnet_025': '54379ae5174d026780215fc748a7f31910dee36818e63d49e17dc598ecc82df7',
    'lcnet_100': '7d3c02ef6c7da8ae08b4347cc7695b2081aae68c325d64375724ecf39c99e743',
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_rgb(path):
    # Чтение через байты поддерживает пути с кириллицей в Windows.
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise ValueError(f'Cannot decode image: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def preprocess(rgb):
    # Предобработка из конфигурации весов PaddleX: прямой resize без padding.
    x = cv2.resize(rgb, (160, 80), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))

def temperature_scale(p, temperature=1.0):
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    # Ограничение защищает logit от бесконечностей; T > 1 смягчает уверенность.
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    logit = (np.log(p) - np.log1p(-p)) / temperature
    return 1 / (1 + np.exp(-logit))

class OrientationModel:
    def __init__(self, name='lcnet_100', model_dir='models', threads=4):
        path = Path(model_dir) / f'{name}.onnx'
        if sha256(path) != MODEL_HASHES[name]:
            raise ValueError(f'Model checksum mismatch: {path}')
        cv2.setNumThreads(1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def raw(self, images):
        # Модель уже содержит softmax; второй столбец соответствует 180°.
        batch = np.stack([preprocess(im) for im in images])
        p = np.asarray(self.session.run(None, {self.input_name: batch})[0])
        if p.shape != (len(images), 2) or not np.isfinite(p).all():
            raise ValueError(f'Unexpected output: {p.shape}')
        if np.any(p < -1e-6) or np.any(p > 1 + 1e-6) or not np.allclose(p.sum(1), 1, atol=1e-4):
            raise ValueError('Expected two-class softmax probabilities')
        return p[:, 1].astype(np.float64)

    def predict(self, images, temperature=1.0, tta=True):
        original = temperature_scale(self.raw(images), temperature)
        if not tta:
            return original
        rotated = temperature_scale(self.raw([np.rot90(im, 2).copy() for im in images]), temperature)
        # 1 - p для повёрнутой копии возвращает вероятность к исходной ориентации.
        return (original + 1 - rotated) / 2
