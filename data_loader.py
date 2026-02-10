from pathlib import Path
import re
import numpy as np
import cv2

_IMAGE_RE = re.compile(r".*_(\d{5})\.jpg")


class SyncDataset:
    def __init__(
        self,
        root: Path,
        camera_prefix: str,
    ):
        """
        root/
          individual/
          lidar_bins/
        camera_prefix example:
          'racecar_camera_camera_1_image_raw'
        """
        self.image_dir = root / "individual"
        self.lidar_dir = root / "lidar_bins"
        self.camera_prefix = camera_prefix

        self.samples = self._index_samples()

    def _index_samples(self):
        samples = {}

        # --- index images ---
        for img_path in self.image_dir.glob(f"{self.camera_prefix}_*.jpg"):
            m = _IMAGE_RE.match(img_path.name)
            if not m:
                continue
            idx = int(m.group(1))
            samples.setdefault(idx, {})["image"] = img_path

        # --- index lidar ---
        for bin_path in self.lidar_dir.glob("*.bin"):
            idx = int(bin_path.stem)
            samples.setdefault(idx, {})["lidar"] = bin_path

        # --- keep only synced pairs ---
        synced = {
            idx: s
            for idx, s in samples.items()
            if "image" in s and "lidar" in s
        }

        return dict(sorted(synced.items()))

    def __len__(self):
        return len(self.samples)

    def indices(self):
        return list(self.samples.keys())

    def load_image(self, idx):
        img = cv2.imread(str(self.samples[idx]["image"]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to load image {idx}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_lidar(self, idx):
        """
        Assumes KITTI-style float32 x,y,z,intensity
        """
        bin_path = self.samples[idx]["lidar"]
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points[:, :3]  # xyz only
