import rerun as rr
import rerun.blueprint as rrb
import numpy as np


def _rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    R = np.asarray(R, dtype=np.float64)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def _default_blueprint():
    """Two 3D views: Lidar only (left), Camera only (right). Each view shows only its own entities."""
    return rrb.Blueprint(
        rrb.Horizontal(
            # Left: only lidar point clouds and board (no camera data)
            rrb.Spatial3DView(
                origin="lidar",
                name="Lidar",
                contents=["lidar/points", "lidar/roi_cloud_xyz", "lidar/board_corners", "lidar/board_pose", "lidar/origin"],
            ),
            # Right: only camera 3D points, axes, and board pose (no lidar point clouds)
            rrb.Spatial3DView(
                origin="camera",
                name="Camera",
                contents=["camera/points_3d", "camera/axes", "camera/board_pose"],
            ),
        ),
        collapse_panels=False,
    )


class RerunVisualizer:
    def __init__(self, app_name="extrinsicCalibration"):
        rr.init(app_name, spawn=True, default_blueprint=_default_blueprint())

    def log_image(self, frame_idx: int, image: np.ndarray):
        # Set the “frame” timeline using the new API:
        rr.set_time("frame", sequence=frame_idx)
        rr.log(
            "camera/image",
            rr.Image(image),
        )

    def log_pointcloud(self, frame_idx: int, points_xyz: np.ndarray, colors: np.ndarray = None):
        rr.set_time("frame", sequence=frame_idx)
        if colors is None:
            # Gray so ROI (red) stands out in the same 3D view
            colors = np.full((points_xyz.shape[0], 3), [160, 160, 160], dtype=np.uint8)
        rr.log(
            "lidar/points",
            rr.Points3D(points_xyz, colors=colors),
        )

    def log_roi_pointcloud(self, frame_idx: int, points_xyz: np.ndarray):
        """Log the ROI cloud (clusters passing board-size filter) in red, same 3D view as lidar/points."""
        rr.set_time("frame", sequence=frame_idx)
        colors = np.full((points_xyz.shape[0], 3), [255, 0, 0], dtype=np.uint8)
        rr.log(
            "lidar/roi_cloud_xyz",
            rr.Points3D(points_xyz, colors=colors),
        )

    def log_corners_3d(
        self,
        frame_idx: int,
        corners_3d: np.ndarray,
        corner_ids: np.ndarray = None,
    ):
        """Log board corner points (4x3) in the lidar 3D view with optional id labels (e.g. yellow points, labels 1–4)."""
        rr.set_time("frame", sequence=frame_idx)
        corners_3d = np.asarray(corners_3d, dtype=np.float32)
        if corners_3d.size == 0:
            return
        n = corners_3d.shape[0]
        colors = np.full((n, 3), [255, 255, 0], dtype=np.uint8)
        labels = None
        if corner_ids is not None and np.size(corner_ids) >= n:
            ids = np.asarray(corner_ids, dtype=np.int32).ravel()[:n]
            labels = [str(int(i)) for i in ids]
        rr.log(
            "lidar/board_corners",
            rr.Points3D(corners_3d, colors=colors, radii=0.02, labels=labels),
        )

    def log_pose_axes(self, frame_idx: int, R: np.ndarray, t: np.ndarray, entity_path: str = "lidar/board_pose", axis_length: float = 0.4):
        """Log a 3D coordinate frame as visible line segments. R (3,3), t (3,) in lidar frame. X=red, Y=green, Z=blue."""
        rr.set_time("frame", sequence=frame_idx)
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).ravel()
        L = axis_length
        # Three segments: origin -> tip of each axis (in world frame: t + L * R col)
        origin = t
        x_tip = t + L * R[:, 0]
        y_tip = t + L * R[:, 1]
        z_tip = t + L * R[:, 2]
        strips = [
            np.array([origin, x_tip], dtype=np.float32),
            np.array([origin, y_tip], dtype=np.float32),
            np.array([origin, z_tip], dtype=np.float32),
        ]
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        rr.log(entity_path, rr.LineStrips3D(strips, colors=colors))

    def log_lidar_origin(self, frame_idx: int, axis_length: float = 0.4):
        """Log the lidar origin axes as visible line segments at (0,0,0). X=red, Y=green, Z=blue."""
        rr.set_time("frame", sequence=frame_idx)
        L = axis_length
        strips = [
            np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, L]], dtype=np.float32),
        ]
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        rr.log("lidar/origin", rr.LineStrips3D(strips, colors=colors))

    def log_camera_points_3d(
        self,
        frame_idx: int,
        points_xyz: np.ndarray,
        corner_ids=None,
    ):
        """Log board corners in camera frame (camera/points_3d) for a separate 3D view."""
        rr.set_time("frame", sequence=frame_idx)
        points_xyz = np.asarray(points_xyz, dtype=np.float32)
        if points_xyz.size == 0:
            return
        n = points_xyz.shape[0]
        colors = np.full((n, 3), [0, 200, 255], dtype=np.uint8)
        labels = None
        if corner_ids is not None and np.size(corner_ids) >= n:
            ids = np.asarray(corner_ids, dtype=np.int32).ravel()[:n]
            labels = [str(int(i)) for i in ids]
        rr.log(
            "camera/points_3d",
            rr.Points3D(points_xyz, colors=colors, radii=0.02, labels=labels),
        )

    def log_camera_axes(self, frame_idx: int, axis_length: float = 0.4):
        """Log camera coordinate frame at origin (camera frame). X=red, Y=green, Z=blue."""
        rr.set_time("frame", sequence=frame_idx)
        L = axis_length
        strips = [
            np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0], [0.0, L, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, L]], dtype=np.float32),
        ]
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        rr.log("camera/axes", rr.LineStrips3D(strips, colors=colors))
