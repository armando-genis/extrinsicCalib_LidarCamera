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
                    corners_cam_2d, _ = self.board.compute_board_corners_in_image(rvec, tvec, K, D_zero)
                    self.board.draw_board_pose(undist, rvec, tvec, K, D_zero, axis_length=0.25)
                    self.board.draw_board_contour_and_circles(undist, rvec, tvec, K, D_zero)
                    self.board.draw_corners_id(undist, rvec, tvec, K, D_zero, corner_ids=corner_ids)
                    # corners_cam are already 3D in camera frame (meters); use directly for viz
                    camera_points_meters = np.asarray(corners_cam, dtype=np.float64)


                    return undist, original_image, rvec, tvec, corners_cam, corner_ids, ids, camera_points_meters, corners_cam_2d

        return undist, original_image, None, None, None, None, None, None, None

    def process_lidar(self, lidar: np.ndarray) -> Dict[str, Any]:
        out = self.lidar_processor.process(lidar)
        return out

    def points_to_robot_frame(self, points: np.ndarray) -> np.ndarray:
        R_cv_to_robot = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0],
        ])
        return (R_cv_to_robot @ points.T).T

    def transform_rvec_tvec_to_robot_frame(self, rvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R_cv_to_robot = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0],
        ])
        R_cam_cv, _ = cv2.Rodrigues(rvec)
        t_cam_cv = tvec.reshape(3)
        R_cam_robot = R_cv_to_robot @ R_cam_cv
        t_cam_robot = R_cv_to_robot @ t_cam_cv
        return R_cam_robot, t_cam_robot

    def run(self):
        for idx in self.dataset.indices():
            image = self.dataset.load_image(idx)
            undist, original_image, rvec, tvec, corners_cam, camera_corner_ids, ids, camera_points_meters, corners_cam_2d = self.process_frame(image)
            lidar = self.dataset.load_lidar(idx)
            processed_lidar = self.process_lidar(lidar)

            # ROI cloud: clusters that passed board-size filter (candidate board regions)
            board = processed_lidar["board"]
            roi_cloud_xyz = board["roi_cloud_xyz"]
            lidar_corner_ids = board["corner_ids"]


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

            # Camera frame: 3D points (board corners in camera coords), camera axes, and board pose (rvec/tvec)
            if camera_points_meters is not None:

                camera_points_robot = self.points_to_robot_frame(camera_points_meters)

                self.viz.log_camera_points_3d(idx, camera_points_robot, corner_ids=camera_corner_ids)
            self.viz.log_camera_axes(idx)
            if rvec is not None and tvec is not None:
                R_cam_robot, t_cam_robot = self.transform_rvec_tvec_to_robot_frame(rvec, tvec)

                self.viz.log_pose_axes(idx, R_cam_robot, t_cam_robot,
                                        entity_path="camera/board_pose")

            lidar_pts = board["corners_3d"]
            # cam_pts = camera_points_robot   #Compute extrinsic in robot frame, later have to convert extrinsic back to OpenCV frame
            cam_pts = corners_cam   #Extrinsic is LiDAR → OpenCV camera
            
            R_ext, t_ext = self.compute_extrinsic_svd(
                np.asarray(lidar_pts),
                np.asarray(cam_pts)
            )

            print("OpenCV Extrinsic R:\n", R_ext)
            print("OpenCV Extrinsic t:\n", t_ext)

            R_cv_to_robot = np.array([
                [ 0,  0,  1],
                [-1,  0,  0],
                [ 0, -1,  0],
            ], dtype=np.float64)

            R_ext_robot = R_cv_to_robot @ R_ext
            t_ext_robot = R_cv_to_robot @ t_ext
            
            self.viz.log_pose_axes(
                idx,
                R_ext_robot,
                t_ext_robot,
                entity_path="extrinsic/lidar_to_camera"
            )

            print("Robot Frame Extrinsic R:\n", R_ext_robot)
            print("Robot Frame Extrinsic t:\n", t_ext_robot)

            # compute reprojection error with the OpenCV extrinsic 
            K = self.undistorter.get_K()
            D_zero = self.undistorter.get_zero_distortion()

            errors, mean_error, std_error = self.compute_reprojection_error(
                lidar_pts,
                corners_cam_2d,
                R_ext,
                t_ext,
                K,
                D_zero
            )

            print("Reprojection error per point (px):", errors)
            print("Mean reprojection error (px):", mean_error)
            print("RMS reprojection error (px):", std_error)

            lidar_in_cam = (R_ext @ lidar_pts.T).T + t_ext
            err_m = np.linalg.norm(lidar_in_cam - corners_cam, axis=1)
            print("3D alignment error (m):", err_m, "mean:", err_m.mean())

            colored_pts, colors = self.colorize_lidar(
                processed_lidar["xyz"],
                undist,      # UNDISTORTED image
                R_ext,
                t_ext,
                K
            )

            self.viz.log_pointcloud(
                idx,
                colored_pts,
                colors=colors
            )

    def compute_extrinsic_svd(self, lidar_pts: np.ndarray,
                            cam_pts: np.ndarray):

        assert lidar_pts.shape == cam_pts.shape
        assert lidar_pts.shape[0] >= 3

        # centroids
        centroid_lidar = np.mean(lidar_pts, axis=0)
        centroid_cam = np.mean(cam_pts, axis=0)

        # remove centroids
        X = lidar_pts - centroid_lidar
        Y = cam_pts - centroid_cam

        # covariance
        H = X.T @ Y

        # SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # reflection fix
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = centroid_cam - R @ centroid_lidar

        return R, t

    def lidar_to_opencv_frame(self, lidar_pts: np.ndarray, R_ext: np.ndarray, t_ext: np.ndarray) -> np.ndarray:
        lidar_pts = np.asarray(lidar_pts, dtype=np.float64)

        lidar_in_cam = (R_ext @ lidar_pts.T).T + t_ext.reshape(1,3)

        return lidar_in_cam

    def compute_reprojection_error(self, lidar_pts, image_pts, R_ext, t_ext, K, D):
        rvec_ext, _ = cv2.Rodrigues(R_ext)

        proj_pts, _ = cv2.projectPoints(
            np.asarray(lidar_pts, dtype=np.float64),
            rvec_ext,
            t_ext.reshape(3,1),
            K,
            D
        )

        proj_pts = proj_pts.reshape(-1, 2)
        image_pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)

        errors = np.linalg.norm(proj_pts - image_pts, axis=1)
        return errors, float(np.mean(errors)), float(np.sqrt(np.mean(errors**2)))

    # Project LiDAR points into the image and color them using the image pixels.
    def project_lidar_to_image(self,
                            lidar_xyz,
                            R_ext,
                            t_ext,
                            K):

        lidar_xyz = np.asarray(lidar_xyz, dtype=np.float64)

        rvec_ext, _ = cv2.Rodrigues(R_ext)

        D_zero = np.zeros((4,1), dtype=np.float64)

        proj_pts, _ = cv2.projectPoints(
            lidar_xyz,
            rvec_ext,
            t_ext.reshape(3,1),
            K,
            D_zero
        )

        proj_pts = proj_pts.reshape(-1, 2)

        return proj_pts

    def colorize_lidar(self,
                    lidar_xyz,
                    image,
                    R_ext,
                    t_ext,
                    K):

        h, w = image.shape[:2]

        proj_pts = self.project_lidar_to_image(
            lidar_xyz,
            R_ext,
            t_ext,
            K
        )

        colors = []
        valid_points = []

        for pt3d, (u, v) in zip(lidar_xyz, proj_pts):

            u_int = int(round(u))
            v_int = int(round(v))

            # Only keep points inside image
            if 0 <= u_int < w and 0 <= v_int < h:

                # Also ensure point is in front of camera
                P_cam = R_ext @ pt3d + t_ext
                if P_cam[2] <= 0:
                    continue

                # Image from decoder is typically RGB; use as-is so Rerun shows correct colors
                color = np.asarray(image[v_int, u_int], dtype=np.uint8)
                colors.append(color)
                valid_points.append(pt3d)

        return np.array(valid_points), np.array(colors, dtype=np.uint8) if colors else np.zeros((0, 3), dtype=np.uint8)



if __name__ == '__main__':
    cam_yaml = "config/cam0.yaml"
    aruco_board_yaml = "config/calibration_board_config.yaml"
    lidar_camera_sync = LidarCameraSync(cam_yaml, aruco_board_yaml)

    # dataset setup (Lidar and Camera folders)
    dataset_path = Path("sync_one")
    camera_prefix = "racecar_camera_camera_0_image_raw"
    lidar_camera_sync.dataloader(dataset_path, camera_prefix)

    # lidar setup
    lidar_yaml = "config/lidar_processor.yaml"
    lidar_camera_sync.set_lidar_processor(lidar_yaml)


    lidar_camera_sync.run()