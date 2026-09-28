import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from predict import read_ids, resolve_images
from src.orientation import OrientationModel, read_rgb, temperature_scale

class ContractTests(unittest.TestCase):
    def test_template_order_and_extension_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            sample = folder / 'sample.csv'
            sample.write_text('image_id,p_180\nz,0.5\na.png,0.5\n', encoding='utf-8')
            (folder/'z.png').touch()
            (folder/'a.png').touch()
            self.assertEqual(read_ids(sample), ['z', 'a.png'])
            self.assertEqual([p.name for p in resolve_images(folder, read_ids(sample))], ['z.png', 'a.png'])

    def test_duplicate_and_missing_ids_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            sample = Path(folder)/'sample.csv'
            sample.write_text('image_id,p_180\na,0.5\na,0.5\n')
            with self.assertRaises(ValueError):
                read_ids(sample)
            with self.assertRaises(FileNotFoundError):
                resolve_images(folder, ['missing'])

    def test_invalid_image_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'invalid.png'
            path.write_bytes(b'not an image')
            with self.assertRaises(ValueError):
                read_rgb(path)

    def test_temperature_preserves_complement(self):
        p = np.array([0, .01, .4, .5, .9, 1.0])
        np.testing.assert_allclose(temperature_scale(p, 1.5) + temperature_scale(1-p, 1.5), 1, atol=1e-10)

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = OrientationModel('lcnet_100')
        im = Image.new('RGB', (850, 70), 'white')
        font = ImageFont.truetype('assets/NotoSans.ttf', 42)
        ImageDraw.Draw(im).text((8, 2), 'Delivery service and telephone', font=font, fill='black')
        cls.upright = np.array(im)

    def test_class_mapping_on_generated_text(self):
        p = self.model.predict([self.upright, np.rot90(self.upright, 2).copy()], 1.5)
        self.assertLess(p[0], .5)
        self.assertGreater(p[1], .5)
        self.assertAlmostEqual(float(p.sum()), 1, places=7)

    def test_batch_independence_and_determinism(self):
        images = [self.upright, np.zeros((19, 27, 3), dtype=np.uint8)]
        p = self.model.predict(images, 1.5)
        q = np.array([self.model.predict([im], 1.5)[0] for im in images])
        np.testing.assert_allclose(p, q, atol=1e-6)
        np.testing.assert_array_equal(p, self.model.predict(images, 1.5))

    def test_wrong_model_checksum_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'lcnet_100.onnx').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                OrientationModel('lcnet_100', folder)

if __name__ == '__main__':
    unittest.main()
