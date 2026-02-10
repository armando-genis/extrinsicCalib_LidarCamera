import os
import yaml
import numpy as np
import cv2
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_loader import SyncDataset
from rerun_viz import RerunVisualizer
from ArUcoBoardModule import ArUcoBoardConfig
from LidarDataProcessor import LidarDataProcessor

def _to_mat_3x3(block):
    if isinstance(block, dict) and "data" in block:
        return np.array(block["data"], dtype=np.float64).reshape(3, 3)
    return np.array(block, dtype=np.float64).reshape(3, 3)


def _to_vec(block):
    if isinstance(block, dict) and "data" in block:
        return np.array(block["data"], dtype=np.float64).reshape(-1, 1)
    return np.array(block, dtype=np.float64).reshape(-1, 1)

class CameraUndistorter:
    def __init__(self, K, D, frame_size_wh):
        self.K = K.astype(np.float64)
        self.D = D.astype(np.float64).reshape(4, 1)
        self.frame_size = tuple(frame_size_wh)

        self.map1 = None
        self.map2 = None
        self._compute_maps()

    def _compute_maps(self):
        w, h = self.frame_size
        R = np.eye(3)

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, R, self.K, (w, h), cv2.CV_16SC2
        )

    def ensure_size(self, w, h):
        if (w, h) != self.frame_size:
            self.frame_size = (w, h)
            self._compute_maps()

    def undistort(self, img):
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

    def get_K(self):
        return self.K

    def get_zero_distortion(self):
        return np.zeros((5, 1), dtype=np.float64)


class LidarCameraSync():
    def __init__(self, cam_yaml: str, aruco_board_yaml: str):

        self.cam_yaml = cam_yaml
        self.aruco_board_yaml = aruco_board_yaml
        
        self.undistorter = self._load_undistorter(cam_yaml)
        self.board = ArUcoBoardConfig(aruco_board_yaml)

        self.dataset = None
        self.lidar_processor = None

        self.viz = RerunVisualizer()
    
    def dataloader(self, dataset_path: str, camera_prefix: str):
        self.dataset = SyncDataset(dataset_path, camera_prefix)

    def set_lidar_processor(self, lidar_yaml: str):
        self.lidar_processor = LidarDataProcessor(lidar_yaml)
        self.lidar_processor.set_board_size(self.board.get_board_size_width(), self.board.get_board_size_height())


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
            return CameraUndistorter(K, D, (0, 0))

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        w = int(data.get("image_width", 0))
        h = int(data.get("image_height", 0))

        K = _to_mat_3x3(data["camera_matrix"])
        D = _to_vec(data["distortion_coefficients"])
        is_fisheye = self._is_fisheye_yaml(data)

        print(f"Loaded {yaml_path} | size=({w}x{h}) | fisheye={is_fisheye} | D_len={D.shape[0]}")

        return CameraUndistorter(K, D, (w, h))

    def process_frame(self, original_image):
        h, w = original_image.shape[:2]
        self.undistorter.ensure_size(w, h)

        undist = self.undistorter.undistort(original_image)

        K = self.undistorter.get_K()
        D_zero = self.undistorter.get_zero_distortion()

        ok, ids, corners = self.board.detect_markers_in_image(undist)
        if ok:
            self.board.draw_marker_corners_into_image(ids, corners, undist)

            if len(ids) >= self.board.min_marker_detection:
                pose_ok, rvec, tvec = self.board.estimate_board_pose(
                    ids, corners, K, D_zero
                )

                if pose_ok:
                    corners_cam, corner_ids = self.board.compute_board_corners_in_camera_frame(rvec, tvec)
                    self.board.draw_board_pose(undist, rvec, tvec, K, D_zero, axis_length=0.25)
                    self.board.draw_board_contour_and_circles(undist, rvec, tvec, K, D_zero)
                    self.board.draw_corners_id(undist, rvec, tvec, K, D_zero, corner_ids=corner_ids)

                    return undist, original_image, rvec, tvec, corners_cam, corner_ids, ids

        return undist, original_image, None, None, None, None, None

    def process_lidar(self, lidar: np.ndarray) -> Dict[str, Any]:
        out = self.lidar_processor.process(lidar)
        return out

    def extrinsic_calibration(self, rvec: np.ndarray, tvec: np.ndarray, 
                            corners_cam: np.ndarray, board_info: Dict[str, Any], 
                            undist: np.ndarray, ids: List[int],
                            lidar_corner_ids: List[int], camera_corner_ids: List[int]):


        if rvec is not None and tvec is not None and board_info["corners_3d"] is not None and corners_cam is not None and undist is not None and len(ids) >= self.board.min_marker_detection:
            # Lidar boarder information
            lidar_corners = np.array(board_info["corners_3d"])
            lidar_rvec = board_info["R"]
            lidar_tvec = board_info["t"]

            # Camera boarder information
            camera_rvec = rvec
            camera_tvec = tvec

            # Reshaping the points correctly
            camera_corners = np.array(corners_cam, dtype=np.float32)
            lidar_corners = np.array(lidar_corners, dtype=np.float32)

            print(f"Camera corners: {camera_corners}")
            print(f"Lidar corners: {lidar_corners}")

            # print length of camera and lidar corners
            print(f"Length of camera corners: {len(camera_corners)}")
            print(f"Length of lidar corners: {len(lidar_corners)}")

            print(f"Lidar corner ids: {lidar_corner_ids}")
            print(f"Camera corner ids: {camera_corner_ids}")

            # Camera matrix (Intrinsic parameters)
            K = self.undistorter.get_K()
            
            # Initialize distortion coefficients to zero (fisheye model needs 4 distortion coefficients)
            D_zero = np.zeros((4, 1), dtype=np.float32)  # [k1, k2, k3, k4] all zero

            if len(camera_corners) >= 4 and len(lidar_corners) >= 4:

                print(f"Camera corners shape before reshaping: {camera_corners.shape}")
                print(f"Lidar corners shape before reshaping: {lidar_corners.shape}")   

                # convert and shape image points properly
                img_pts = np.array(camera_corners[:, :2], dtype=np.float32).reshape(-1, 1, 2)

                # convert and shape object points properly
                obj_pts = np.array(lidar_corners, dtype=np.float32).reshape(-1, 1, 3)

                print(f"Image points shape: {img_pts.shape}")
                print(f"Object points shape: {obj_pts.shape}")
                print(f"Image points dtype: {img_pts.dtype}")
                print(f"Object points dtype: {obj_pts.dtype}")

                success, rvec_lidar_to_cam, tvec_lidar_to_cam = cv2.solvePnP(
                    obj_pts,  # 3D LiDAR points
                    img_pts,  # Undistorted 2D camera points
                    K,  # Identity matrix for undistorted camera
                    D_zero,  # Zero distortion coefficients
                    flags=cv2.SOLVEPNP_ITERATIVE  # Use iterative solution for PnP
                )

                if success:
                    print("PnP calibration successful")
                    print("rvec_lidar_to_cam:", rvec_lidar_to_cam)
                    print("tvec_lidar_to_cam:", tvec_lidar_to_cam)

                    # Reprojection error
                    proj_pts, _ = cv2.projectPoints(obj_pts, rvec_lidar_to_cam, tvec_lidar_to_cam, K, D_zero)
                    error = cv2.norm(img_pts, proj_pts, cv2.NORM_L2) / len(proj_pts)
                    print(f"Reprojection error: {error}")
                else:
                    print("PnP calibration failed")
            else:
                print("Not enough points for PnP calibration. At least 4 points required.")


    def run(self):
        for idx in self.dataset.indices():
            image = self.dataset.load_image(idx)
            undist, original_image, rvec, tvec, corners_cam, camera_corner_ids, ids = self.process_frame(image)
            lidar = self.dataset.load_lidar(idx)
            processed_lidar = self.process_lidar(lidar)

            # ROI cloud: clusters that passed board-size filter (candidate board regions)
            board = processed_lidar["board"]
            roi_cloud_xyz = board["roi_cloud_xyz"]
            lidar_corner_ids = board["corner_ids"]


            # extrinsic calibration
            self.extrinsic_calibration(rvec, tvec, corners_cam, board, undist, ids, lidar_corner_ids, camera_corner_ids)

            self.viz.log_image(idx, undist)
            self.viz.log_pointcloud(idx, processed_lidar["xyz"])

            self.viz.log_roi_pointcloud(idx, roi_cloud_xyz)
            # Lidar origin (axes at 0,0,0)
            self.viz.log_lidar_origin(idx)
            # Detected board: corners and pose axes
            if board["corners_3d"] is not None:
                self.viz.log_corners_3d(
                    idx, board["corners_3d"], corner_ids=board.get("corner_ids")
                )
            if board["R"] is not None and board["t"] is not None:
                self.viz.log_pose_axes(idx, board["R"], board["t"])

if __name__ == '__main__':
    cam_yaml = "config/cam0.yaml"
    aruco_board_yaml = "config/calibration_board_config.yaml"
    lidar_camera_sync = LidarCameraSync(cam_yaml, aruco_board_yaml)

    # dataset setup (Lidar and Camera folders)
    dataset_path = Path("sync_lidar_camera_aruco")
    camera_prefix = "racecar_camera_camera_0_image_raw"
    lidar_camera_sync.dataloader(dataset_path, camera_prefix)

    # lidar setup
    lidar_yaml = "config/lidar_processor.yaml"
    lidar_camera_sync.set_lidar_processor(lidar_yaml)


    lidar_camera_sync.run()