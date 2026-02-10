#!/usr/bin/env python3
import os
import yaml
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
from rerunModule import RerunModule
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


class LidarCameraSync(Node):
    def __init__(self):
        super().__init__('lidar_camera_sync')

        # ---- Parameters ----
        self.declare_parameter('image_topic', '/racecar/camera/camera_0/image_raw')
        self.declare_parameter('lidar_topic', '/velodyne_points')
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('slop', 0.05)
        self.declare_parameter('allow_headerless', False)

        self.declare_parameter('cam_yaml', 'cam0.yaml')
        self.declare_parameter('aruco_board_yaml', 'calibration_board_config.yaml')

        image_topic = self.get_parameter('image_topic').value
        lidar_topic = self.get_parameter('lidar_topic').value
        queue_size = int(self.get_parameter('queue_size').value)
        slop = float(self.get_parameter('slop').value)
        allow_headerless = bool(self.get_parameter('allow_headerless').value)

        cam_yaml = self.get_parameter('cam_yaml').value
        aruco_board_yaml = self.get_parameter('aruco_board_yaml').value

        self.bridge = CvBridge()
        self.undistorter = self._load_undistorter(cam_yaml)

        # ---- message_filters subscribers ----
        self.img_sub = Subscriber(self, Image, image_topic)
        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.img_sub, self.lidar_sub],
            queue_size=queue_size,
            slop=slop,
            allow_headerless=allow_headerless
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info(
            f"Syncing:\n  Image: {image_topic}\n  LiDAR: {lidar_topic}\n"
            f"  queue_size={queue_size}, slop={slop}s\n"
        )

        self.board = ArUcoBoardConfig(aruco_board_yaml)

        self.rerun = RerunModule(app_id="lidar_camera", spawn=True)

    def get_k(self) -> np.ndarray:
        return self._K

    def get_d(self) -> np.ndarray:
        return self._D
        

    # ---- This matches your suggested approach ----
    def _decode_image_to_bgr(self, msg: Image) -> np.ndarray:
        """
        Process incoming image:
          - If encoding is 'mjpeg': decode bytes via cv2.imdecode
          - Else if encoding is rgb8/bgr8/mono8: use cv_bridge -> bgr8
        """
        enc = (msg.encoding or "").lower()

        try:
            if enc == 'mjpeg' or enc == 'jpeg' or enc == 'jpg':
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR
                if frame is None:
                    raise RuntimeError("Failed to decode MJPEG/JPEG with cv2.imdecode")
                return frame

        except Exception as e:
            raise RuntimeError(f"Decode failed (encoding={msg.encoding}): {e}") from e

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

        self.get_logger().info(
            f"Loaded {yaml_path} | size=({w}x{h}) | fisheye={is_fisheye} | D_len={D.shape[0]}"
        )

        return CameraUndistorter(K, D, (w, h), is_fisheye=is_fisheye)

    def synced_callback(self, img_msg: Image, pc_msg: PointCloud2):
        # compute sync dt
        t_img = img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9
        t_pc = pc_msg.header.stamp.sec + pc_msg.header.stamp.nanosec * 1e-9
        dt = abs(t_img - t_pc)

        # decode -> BGR
        try:
            cv_bgr = self._decode_image_to_bgr(img_msg)
        except Exception as e:
            self.get_logger().warn(str(e))
            return

        h, w = cv_bgr.shape[:2]

        # undistort
        self.undistorter.ensure_size(w, h)
        undist = self.undistorter.undistort(cv_bgr)

        ok, ids, corners, annotated = self.board.detect_markers_in_image(undist)
        if ok:
            self.board.draw_marker_corners_into_image(ids, corners, undist)

        self.get_logger().info(
            f"SYNC OK | dt={dt:.4f}s | img {w}x{h} enc={img_msg.encoding} | "
            f"pc fields={len(pc_msg.fields)} frame={pc_msg.header.frame_id}"
        )

        self.rerun.log_image(undist, entity_path="camera/image", stamp=img_msg.header.stamp)
        self.rerun.log_point_cloud(pc_msg, entity_path="lidar/points", stamp=pc_msg.header.stamp)


def main():
    rclpy.init()
    node = LidarCameraSync()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
