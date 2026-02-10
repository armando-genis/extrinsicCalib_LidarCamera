import os
import yaml
import numpy as np
import cv2
from pathlib import Path

from data_loader import SyncDataset
from rerun_viz import RerunVisualizer
from ArUcoBoardModule import ArUcoBoardConfig

def _to_mat_3x3(block):
    if isinstance(block, dict) and "data" in block:
        return np.array(block["data"], dtype=np.float64).reshape(3, 3)
    return np.array(block, dtype=np.float64).reshape(3, 3)


def _to_vec(block):
    if isinstance(block, dict) and "data" in block:
        return np.array(block["data"], dtype=np.float64).reshape(-1, 1)
    return np.array(block, dtype=np.float64).reshape(-1, 1)

class CameraUndistorter:
    def __init__(self, K: np.ndarray, D: np.ndarray, frame_size_wh, is_fisheye: bool):
        self.K = K
        self.D = D
        self.frame_size = tuple(frame_size_wh)  # (w, h)
        self.is_fisheye = is_fisheye
        self.map1 = None
        self.map2 = None
        self._compute_maps()

    def _compute_maps(self):
        w, h = self.frame_size
        if w <= 0 or h <= 0:
            self.map1, self.map2 = None, None
            return

        if self.is_fisheye:
            if self.D.shape[0] != 4:
                self.D = self.D[:4].reshape(4, 1)
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.K, (w, h), cv2.CV_16SC2
            )
        else:
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.K, (w, h), cv2.CV_16SC2
            )

    def ensure_size(self, w, h):
        if (w, h) != (self.frame_size[0], self.frame_size[1]):
            self.frame_size = (w, h)
            self._compute_maps()

    def undistort(self, img: np.ndarray) -> np.ndarray:
        if self.map1 is None or self.map2 is None:
            return img
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

    def get_k(self) -> np.ndarray:
        return self.K

    def get_d(self) -> np.ndarray:
        return self.D

class LidarCameraSync():
    def __init__(self, cam_yaml: str, aruco_board_yaml: str):

        self.cam_yaml = cam_yaml
        self.aruco_board_yaml = aruco_board_yaml
        
        self.D = None
        self.K = None

        self.undistorter = self._load_undistorter(cam_yaml)
        self.board = ArUcoBoardConfig(aruco_board_yaml)

        self.dataset = None

        self.viz = RerunVisualizer()
    
    def dataloader(self, dataset_path: str, camera_prefix: str):
        self.dataset = SyncDataset(dataset_path, camera_prefix)


    def _resolve_yaml_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        cwd_path = os.path.join(os.getcwd(), path)
        if os.path.exists(cwd_path):
            return cwd_path
        return path

    def _is_fisheye_yaml(self, data: dict) -> bool:
        camera_type = str(data.get("camera_type", "")).strip().lower()
        if camera_type == "fisheye":
            return True
        dist_model = str(data.get("distortion_model", "")).strip().lower()
        return dist_model in ("equidistant", "fisheye")

    def _load_undistorter(self, yaml_path: str) -> CameraUndistorter:
        yaml_path = self._resolve_yaml_path(yaml_path)

        if not os.path.exists(yaml_path):
            self.get_logger().warn(f"cam_yaml not found: {yaml_path}. Undistortion disabled.")
            K = np.eye(3, dtype=np.float64)
            D = np.zeros((4, 1), dtype=np.float64)
            self._K = K
            self._D = D
            return CameraUndistorter(K, D, (0, 0), is_fisheye=True)

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        w = int(data.get("image_width", 0))
        h = int(data.get("image_height", 0))

        K = _to_mat_3x3(data["camera_matrix"])
        D = _to_vec(data["distortion_coefficients"])
        is_fisheye = self._is_fisheye_yaml(data)

        self._K = K
        self._D = D

        print(f"Loaded {yaml_path} | size=({w}x{h}) | fisheye={is_fisheye} | D_len={D.shape[0]}")

        return CameraUndistorter(K, D, (w, h), is_fisheye=is_fisheye)

    def process_frame(self, original_image):
        h, w = original_image.shape[:2]
        self.undistorter.ensure_size(w, h)
        undist = self.undistorter.undistort(original_image)
        ok, ids, corners = self.board.detect_markers_in_image(undist)
        if ok: 
            self.board.draw_marker_corners_into_image(ids, corners, undist)
            # if there are 4 markers, estimate board pose
            if len(ids) >= self.board.min_marker_detection:

                # estimate board pose
                pose_ok, rvec, tvec = self.board.estimate_board_pose(
                    ids,
                    corners,
                    self._K,
                    np.zeros((self._D.shape[0], 1)),
                )

                if pose_ok:
                    self.board.draw_board_pose(
                        undist,
                        rvec,
                        tvec,
                        self._K,
                        np.zeros((self._D.shape[0], 1)),
                        axis_length=0.25,
                    )
        return undist, original_image

    def run(self):
        for idx in self.dataset.indices():
            image = self.dataset.load_image(idx)
            undist, original_image = self.process_frame(image)
            lidar = self.dataset.load_lidar(idx)
            self.viz.log_image(idx, undist)
            self.viz.log_pointcloud(idx, lidar)



if __name__ == '__main__':
    cam_yaml = "config/cam0.yaml"
    aruco_board_yaml = "config/calibration_board_config.yaml"
    lidar_camera_sync = LidarCameraSync(cam_yaml, aruco_board_yaml)

    dataset_path = Path("sync_lidar_camera_aruco")
    camera_prefix = "racecar_camera_camera_0_image_raw"
    lidar_camera_sync.dataloader(dataset_path, camera_prefix)
    lidar_camera_sync.run()