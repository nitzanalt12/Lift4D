import unittest
import numpy as np
from evaluation.export_input_views import camera_description,namespace_file
from pathlib import Path
import tempfile


class Export(unittest.TestCase):
    def test_native_square_crop_camera(self):
        c=camera_description(854,480,1200)
        self.assertEqual((c['cx'],c['cy']),(427,240))
        self.assertEqual(c['native_crop_top'],187)
        self.assertEqual(c['native_crop_left'],0)
        np.testing.assert_array_equal(c['world_to_camera'],np.eye(4))
        c=camera_description(481,854,1200)
        self.assertEqual(c['cx'],241)  # Native integer crop: preserve half-pixel asymmetry.
        self.assertEqual(c['cy'],427)
    def test_safe_config_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cfg_args';p.write_text('Namespace(K=3, hyper_dim=8, use_hash=False)')
            self.assertEqual(namespace_file(p)['hyper_dim'],8)
            p.write_text('__import__("os").system("false")')
            with self.assertRaises(ValueError):namespace_file(p)


if __name__=='__main__':unittest.main()
