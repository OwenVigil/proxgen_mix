import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare import prepare


class PrepareTests(unittest.TestCase):
    def test_observation_action_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, cache = root / "tiny.tar.xz", root / "cache"
            payload = io.BytesIO()
            obs = np.zeros((3, 3, 64, 64), dtype=np.float32)
            obs[0, 0] = 12
            obs[1, 2] = 34
            obs[2] = 255
            np.save(payload, {"observations": obs, "actions": np.array([[3], [5]])})
            payload.seek(0)
            member = tarfile.TarInfo("coinrun/date_21_2_144_10.00.npy")
            member.size = len(payload.getbuffer())
            with tarfile.open(archive, "w:xz") as tar:
                tar.addfile(member, payload)
            prepare(archive, cache)

            images = np.fromfile(cache / "rgb.bin", dtype=np.uint8).reshape(-1, 64, 64, 3)
            np.testing.assert_array_equal(images[:, 0, 0], [[12, 0, 0], [0, 0, 34]])
            np.testing.assert_array_equal(np.fromfile(cache / "action.bin", dtype=np.uint8), [3, 5])
            np.testing.assert_array_equal(np.load(cache / "episodes.npy"), [[144, 2]])
            self.assertEqual(json.loads((cache / "meta.json").read_text())["steps"], 2)
