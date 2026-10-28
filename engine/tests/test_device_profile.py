import unittest

from rec_mlx.kernels.device import profile_for_device


class DeviceProfileTests(unittest.TestCase):
    def test_m4_max_uses_the_40_core_worker_pool(self):
        profile = profile_for_device({"architecture": "applegpu_g16s", "device_name": "Apple M4 Max"})
        self.assertEqual((profile.workers, profile.mlp_workers, profile.head_workers), (76, 64, 64))
        self.assertEqual(profile.prefill_chunk_size, 2048)
        self.assertFalse(profile.specialize_mixed_experts)

    def test_m4_pro_keeps_the_default_pool(self):
        profile = profile_for_device({"architecture": "applegpu_g16s", "device_name": "Apple M4 Pro"})
        self.assertEqual((profile.workers, profile.mlp_workers, profile.head_workers), (36, 32, 32))


if __name__ == "__main__":
    unittest.main()
