#!/usr/bin/env python3
"""
ArUco board configuration loader.
Reads YAML target config (board_width, board_height, marker_size, marker_ids,
marker_positions, cutouts, min_marker_detection, cad_model_mesh, cad_model_cloud)
compatible with the multisensor_calibration C++ CalibrationTarget format.
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import yaml

try:
    import cv2
    from cv2 import aruco
except ImportError:
    cv2 = None
    aruco = None

from utils_annotation import quad_corners_square

# Geometry id for circular cutouts (matches C++ CircularCutout)
CUTOUT_GEOMETRY_ID_CIRCULAR = 1
CIRCULAR_CUTOUT_NUM_COEFFICIENTS = 3  # center_x, center_y, radius


def _parse_opencv_matrix(node) -> List:
    """
    Parse OpenCV-style matrix from YAML: { rows, cols, dt, data }.
    Returns a flat list of values (row-major).
    """
    if node is None:
        return []
    if isinstance(node, list):
        return node
    if isinstance(node, (int, float)):
        return [node]
    # OpenCV format
    data = node.get("data")
    if data is None:
        return []
    return list(data)


def _matrix_to_rows(data: List, cols: int) -> List[List[float]]:
    """Split flat row-major data into rows of length cols."""
    if not data or cols <= 0:
        return []
    return [data[i : i + cols] for i in range(0, len(data), cols)]


class ArUcoBoardConfig:
    """
    ArUco calibration board configuration loaded from a YAML file.
    Constructor takes the path to the YAML file (or directory containing it).
    """

    def __init__(self, yaml_path: str):
        """
        Load board configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML config file, or to the directory
                       containing the config (then a default filename is not applied;
                       you must pass the full path to the file).
        """
        self._yaml_dir = os.path.dirname(os.path.abspath(yaml_path))

        self.board_width: float = 0.0
        self.board_height: float = 0.0
        self.marker_size: float = 0.0
        self.marker_ids: List[int] = []
        self.marker_positions: List[Tuple[float, float, float]] = []  # (x, y, z=0)
        self.cutouts: List[dict] = []  # [ {"geometry_id": int, "coefficients": [float, ...] }, ... ]
        self.min_marker_detection: int = 1
        self.cad_model_mesh: str = ""
        self.cad_model_cloud: str = ""

        self.aruco_dictionary = None  # cv2.aruco.Dictionary
        self.aruco_board = None  # cv2.aruco.Board
        self._detector_params = None  # cv2.aruco.DetectorParameters (created with board)

        self.board_color = (2, 75, 185)

        self._load(yaml_path)

    def _load(self, yaml_path: str) -> None:
        path = Path(yaml_path)
        if path.is_dir():
            print("ArUcoBoardConfig: yaml_path is a directory; expected a file path. Not loading.")
            return
        if not path.exists():
            print(f"ArUcoBoardConfig: file not found: {yaml_path}")
            return

        try:
            with open(path, "r") as f:
                raw = f.read()
            # Strip OpenCV/ROS "%YAML:1.0" or "%YAML 1.0" first line so PyYAML parses the rest.
            lines = raw.splitlines()
            if lines and lines[0].strip().startswith("%YAML"):
                raw = "\n".join(lines[1:])
            data = yaml.safe_load(raw)
        except Exception as e:
            print(f"ArUcoBoardConfig: failed to load YAML {yaml_path}: {e}")
            return

        if not data or not isinstance(data, dict):
            return

        # Board size
        self.board_width = float(data.get("board_width") or 0.0)
        self.board_height = float(data.get("board_height") or 0.0)
        if self.board_width <= 0 or self.board_height <= 0:
            print("ArUcoBoardConfig: Board size ('board_width', 'board_height') in target configuration file not specified or invalid!")

        # Marker size
        self.marker_size = float(data.get("marker_size") or 0.0)
        if self.marker_size == 0.0:
            print("ArUcoBoardConfig: Size of ArUco markers ('marker_size') in target configuration file not specified!")

        # Marker IDs (single-column matrix -> list of ints)
        marker_ids_node = data.get("marker_ids")
        ids_data = _parse_opencv_matrix(marker_ids_node)
        self.marker_ids = [int(x) for x in ids_data]
        if not self.marker_ids:
            print("ArUcoBoardConfig: Ids of ArUco markers ('marker_ids') in target configuration file not specified!")

        # Marker positions (rows x 2 or 3, row-major; z=0 if only x,y)
        marker_pos_node = data.get("marker_positions")
        pos_data = _parse_opencv_matrix(marker_pos_node)
        rows_node = marker_pos_node if isinstance(marker_pos_node, dict) else {}
        n_rows = int(rows_node.get("rows", 0)) if isinstance(rows_node, dict) else 0
        n_cols = int(rows_node.get("cols", 2)) if isinstance(rows_node, dict) else 2
        if n_rows > 0 and n_cols > 0:
            row_list = _matrix_to_rows(pos_data, n_cols)
            for row in row_list:
                x = float(row[0])
                y = float(row[1]) if len(row) > 1 else 0.0
                z = float(row[2]) if len(row) > 2 else 0.0
                self.marker_positions.append((x, y, z))
        elif pos_data:
            # Flat list: assume [x1,y1, x2,y2, ...] or [x1,y1,z1, ...]
            step = 3 if len(pos_data) % 3 == 0 else 2
            for i in range(0, len(pos_data), step):
                x = float(pos_data[i])
                y = float(pos_data[i + 1]) if i + 1 < len(pos_data) else 0.0
                z = float(pos_data[i + 2]) if step == 3 and i + 2 < len(pos_data) else 0.0
                self.marker_positions.append((x, y, z))

        if not self.marker_positions:
            print("ArUcoBoardConfig: Positions of ArUco markers ('marker_positions') in target configuration file not specified!")

        if len(self.marker_ids) != len(self.marker_positions):
            print("ArUcoBoardConfig: Unequal number of ArUco marker IDs ('marker_ids') and marker positions ('marker_positions') in target configuration file!")

        # Cutouts: single-row matrix; each cutout = geometry_id + coefficients
        cutouts_node = data.get("cutouts")
        if cutouts_node:
            cutouts_data = _parse_opencv_matrix(cutouts_node)
            if cutouts_data:
                col_idx = 0
                while col_idx < len(cutouts_data):
                    geom_id = int(cutouts_data[col_idx])
                    col_idx += 1
                    if geom_id == CUTOUT_GEOMETRY_ID_CIRCULAR:
                        n_coeffs = CIRCULAR_CUTOUT_NUM_COEFFICIENTS
                    else:
                        # Unknown geometry: assume 0 coefficients to avoid eating columns
                        n_coeffs = 0
                    coeffs = []
                    for _ in range(n_coeffs):
                        if col_idx < len(cutouts_data):
                            coeffs.append(float(cutouts_data[col_idx]))
                            col_idx += 1
                    self.cutouts.append({"geometry_id": geom_id, "coefficients": coeffs})

        # Min marker detection
        self.min_marker_detection = int(data.get("min_marker_detection") or 1)
        if self.min_marker_detection < 1 or self.min_marker_detection > len(self.marker_ids):
            print("ArUcoBoardConfig: min_marker_detection exceeds limits [1, <number of marker ids>]. Truncating at limits.")
            self.min_marker_detection = max(
                1, min(self.min_marker_detection, len(self.marker_ids) or 1)
            )

        # CAD paths (relative -> absolute w.r.t. YAML directory)
        self.cad_model_mesh = (data.get("cad_model_mesh") or "").strip()
        if not self.cad_model_mesh:
            print("ArUcoBoardConfig: cad_model_mesh is empty; no file path to a CAD model provided. This may hinder pose optimization.")
        else:
            self.cad_model_mesh = os.path.normpath(
                os.path.join(self._yaml_dir, self.cad_model_mesh)
            )

        self.cad_model_cloud = (data.get("cad_model_cloud") or "").strip()
        if not self.cad_model_cloud:
            print("ArUcoBoardConfig: cad_model_cloud is empty; no file path to a CAD model provided. This may hinder pose optimization.")
        else:
            self.cad_model_cloud = os.path.normpath(
                os.path.join(self._yaml_dir, self.cad_model_cloud)
            )

        print(
            f"ArUcoBoardConfig: Successfully read calibration board config from {yaml_path} "
            f"(board {self.board_width:.2f}x{self.board_height:.2f} m, {len(self.marker_ids)} markers, marker_size={self.marker_size:.3f} m)"
        )
        self.create_aruco_board()

    def get_board_size_width(self) -> float:
        return self.board_width

    def get_board_size_height(self) -> float:
        return self.board_height

    def create_aruco_board(self) -> None:
        """
        Create the ArUco dictionary (DICT_6X6_250) and Board from loaded
        marker_ids and marker_positions. Board corners are built as
        top-left, top-right, bottom-right, bottom-left per marker.
        """
        if aruco is None or cv2 is None:
            print("ArUcoBoardConfig: OpenCV or cv2.aruco not available; ArUco board not created.")
            return
        if not self.marker_ids or not self.marker_positions:
            print("ArUcoBoardConfig: No marker_ids or marker_positions; ArUco board not created.")
            return
        if len(self.marker_ids) != len(self.marker_positions):
            print("ArUcoBoardConfig: marker_ids and marker_positions length mismatch; ArUco board not created.")
            return
        if self.marker_size <= 0.0:
            print("ArUcoBoardConfig: marker_size not set; ArUco board not created.")
            return

        # ArUco dictionary
        self.aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

        # Corner points of each marker: top-left, top-right, bottom-right, bottom-left (same as C++)
        n_markers = len(self.marker_ids)
        board_corners = np.zeros((n_markers, 4, 3), dtype=np.float32)
        for i in range(n_markers):
            x, y, z = self.marker_positions[i]
            # top-left
            board_corners[i, 0, :] = (x, y, z)
            # top-right
            board_corners[i, 1, :] = (x + self.marker_size, y, z)
            # bottom-right
            board_corners[i, 2, :] = (x + self.marker_size, y - self.marker_size, z)
            # bottom-left
            board_corners[i, 3, :] = (x, y - self.marker_size, z)

        ids_array = np.array(self.marker_ids, dtype=np.int32)

        try:
            # OpenCV 4.7+: Board.create() or Board() constructor; older: Board_create()
            if hasattr(aruco.Board, "create"):
                self.aruco_board = aruco.Board.create(
                    board_corners, self.aruco_dictionary, ids_array
                )
            elif hasattr(aruco, "Board_create"):
                self.aruco_board = aruco.Board_create(
                    board_corners, self.aruco_dictionary, ids_array
                )
            else:
                # Board(objPoints, dictionary, ids) constructor
                self.aruco_board = aruco.Board(
                    board_corners, self.aruco_dictionary, ids_array
                )
            print(f"ArUcoBoardConfig: ArUco board created successfully ({n_markers} markers, DICT_6X6_250).")
            self._create_detector_params()
        except (cv2.error, TypeError, AttributeError) as ex:
            print(f"ArUcoBoardConfig: cv2.aruco.Board creation failed: {ex}")
            return

    def _create_detector_params(self) -> None:
        """Create default ArUco detector parameters (used by detect_markers_in_image)."""
        if aruco is None:
            self._detector_params = None
        else:
            # Use the newer API: constructor exists in opencv >=4.7
            try:
                self._detector_params = aruco.DetectorParameters()
            except AttributeError:
                # Fallback if only old create exists
                self._detector_params = aruco.DetectorParameters_create()


    def detect_markers_in_image(
        self,
        camera_image: np.ndarray,
    ) -> Tuple[bool, List[int], List[Tuple[Tuple[float, float], ...]], np.ndarray]:
        """Detect ArUco markers in image (cv2.aruco)."""

        empty_ids: List[int] = []
        empty_corners: List[Tuple[Tuple[float, float], ...]] = []
        if camera_image is None or camera_image.size == 0:
            return False, empty_ids, empty_corners, camera_image.copy() if camera_image is not None else np.array([])
        if aruco is None or self.aruco_dictionary is None:
            return False, empty_ids, empty_corners, camera_image.copy()

        # Detect markers using newer ArucoDetector if available
        if hasattr(aruco, "ArucoDetector") and self._detector_params is not None:
            detector = aruco.ArucoDetector(self.aruco_dictionary, self._detector_params)
            tmp_corners, ids, rejected = detector.detectMarkers(camera_image)
        else:
            # Older API
            tmp_corners, ids, rejected = aruco.detectMarkers(
                camera_image, self.aruco_dictionary, parameters=self._detector_params
            )

        detected_ids: List[int] = []
        if ids is not None:
            detected_ids = [int(i) for i in ids.flatten()]

        detected_corners: List[Tuple[Tuple[float, float], ...]] = []
        for corners in tmp_corners or []:
            # corners shape depends; flatten to [4][2]
            pts = tuple((float(p[0]), float(p[1])) for p in corners.reshape(-1, 2))
            detected_corners.append(pts)

        return True, detected_ids, detected_corners


    def draw_marker_corners_into_image(
        self,
        marker_ids: List[int],
        marker_corners: List[Tuple[Tuple[float, float], ...]],
        image: np.ndarray,
    ) -> None:
        """
        Draw each detected marker as a quad with corner brackets (from utils_annotation)
        using the actual 4 corner positions, and a filled circle at each corner.
        Edges align with the circles (no axis-aligned bounding box). Modifies image in place.
        Circle radius = 1% of smallest image dimension.

        Args:
            marker_ids: List of marker ids (same length as marker_corners).
            marker_corners: List of 4 (x,y) corners per marker.
            image: BGR image to draw on (modified in place).
        """
        if image is None or image.size == 0 or len(marker_ids) != len(marker_corners):
            return
        h, w = image.shape[:2]
        radius = max(1, min(h, w) // 100)

        # Color lookup: one BGR color per board marker id (by index)
        if self.marker_ids:
            np.random.seed(42)
            colors = np.random.randint(0, 256, (max(len(self.marker_ids), 1), 3), dtype=np.uint8)
            id_to_color = {mid: tuple(int(c) for c in colors[i]) for i, mid in enumerate(self.marker_ids)}
        else:
            id_to_color = {}

        for i, marker_id in enumerate(marker_ids):
            color = id_to_color.get(marker_id, (0, 255, 0))  # default green BGR
            corners = marker_corners[i]
            quad_corners_square(image, list(corners), color)
            for corner in corners:
                x, y = int(round(corner[0])), int(round(corner[1]))
                cv2.circle(image, (x, y), radius, color, -1)

    @property
    def yaml_dir(self) -> str:
        """Directory containing the loaded YAML file (for resolving relative paths)."""
        return self._yaml_dir

    def board_size(self) -> Tuple[float, float]:
        """Return (board_width, board_height) in meters."""
        return (self.board_width, self.board_height)


    def estimate_board_pose(
        self,
        detected_marker_ids,
        detected_marker_corners,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ):
        """
        Estimate the ArUco board pose with respect to the camera.

        Args:
            detected_marker_ids: List of detected marker IDs
            detected_marker_corners: List of 4 (x, y) corners per marker
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: distortion coefficients (zeros if image is undistorted)

        Returns:
            pose_ok (bool)
            rvec (3x1 np.ndarray) or None
            tvec (3x1 np.ndarray) or None
        """

        # --- Basic validation
        if (
            self.aruco_board is None
            or not detected_marker_ids
            or len(detected_marker_ids) != len(detected_marker_corners)
        ):
            return False, None, None

        # --- Convert marker corners to OpenCV format
        marker_corners_cv = [
            np.asarray(corners, dtype=np.float32).reshape(4, 2)
            for corners in detected_marker_corners
        ]

        # --- Marker IDs must be (N, 1)
        marker_ids_cv = np.asarray(detected_marker_ids, dtype=np.int32).reshape(-1, 1)

        # --- Initial pose (OpenCV updates these in-place)
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        # --- Pose estimation
        try:
            result = cv2.aruco.estimatePoseBoard(
                marker_corners_cv,
                marker_ids_cv,
                self.aruco_board,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                False,  # useExtrinsicGuess
            )
        except cv2.error as e:
            print("estimate_board_pose: cv2.error:", e)
            return False, None, None

        # --- Handle OpenCV Python binding differences
        if isinstance(result, tuple):
            retval, rvec, tvec = result
            retval = int(retval)
        else:
            retval = int(result)

        # --- Validate result
        if (
            retval <= 0
            or np.isnan(np.linalg.norm(rvec))
            or np.isnan(np.linalg.norm(tvec))
        ):
            return False, None, None

        return True, rvec, tvec


    def draw_board_pose(
        self,
        image: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        axis_length: float = 0.2,
    ) -> None:
        """
        Draw the ArUco board coordinate frame into the image.

        Args:
            image: BGR image (modified in place)
            rvec: (3x1) rotation vector (Rodrigues)
            tvec: (3x1) translation vector (meters)
            camera_matrix: camera intrinsic matrix (3x3)
            dist_coeffs: distortion coefficients (should be zeros for undistorted image)
            axis_length: axis length in meters
        """
        if image is None or rvec is None or tvec is None:
            return
        try:
            cv2.drawFrameAxes(
                image,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                axis_length,
                8,
            )
        except cv2.error as e:
            print("draw_board_pose cv2.error:", e)

    def compute_board_corners_in_camera_frame(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute exactly 4 board corners in camera frame from the board pose (same convention as lidar).

        Uses the same corner order as LidarDataProcessor.compute_board_corners_from_pose so
        camera and lidar corners correspond to the same physical points. Numbering:

            1 ------------------ 2
            |                         |
            |                         |
            3 ------------------ 4

        Board frame (index 0..3 = corners 1..4):
          corner 1: (-w/2, +h/2, 0)  top-left
          corner 2: (+w/2, +h/2, 0)  top-right
          corner 3: (-w/2, -h/2, 0)  bottom-left
          corner 4: (+w/2, -h/2, 0)  bottom-right

        Args:
            rvec: (3,1) or (3,) rotation vector (Rodrigues) from estimate_board_pose
            tvec: (3,1) or (3,) translation in meters (camera frame)

        Returns:
            corners_cam: (4, 3) float32, exactly 4 corners in camera frame
            corner_ids: (4,) int32, ids [1, 2, 3, 4] for each corner (empty array on failure)
        """
        empty_corners = np.zeros((0, 3), dtype=np.float32)
        empty_ids = np.array([], dtype=np.int32)
        if rvec is None or tvec is None:
            return empty_corners, empty_ids
        w = self.board_width
        h = self.board_height
        if w <= 0 or h <= 0:
            return empty_corners, empty_ids

        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        t = np.asarray(tvec, dtype=np.float64).reshape(3,)

        hw = w * 0.5
        hh = h * 0.5
        # Order: 1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right (exactly 4 corners)
        corners_board = np.array(
            [
                [-hw,  hh, 0.0],   # 1 top-left
                [ hw,  hh, 0.0],   # 2 top-right
                [-hw, -hh, 0.0],   # 3 bottom-left
                [ hw, -hh, 0.0],   # 4 bottom-right
            ],
            dtype=np.float64,
        )
        corners_cam = (corners_board @ R.T) + t[None, :]
        corner_ids = np.array([1, 2, 3, 4], dtype=np.int32)
        return corners_cam.astype(np.float32), corner_ids

    def compute_board_corners_in_image(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the 4 board corners in 2D image (pixel) coordinates.

        Same corner order and IDs as compute_board_corners_in_camera_frame:
            1 top-left, 2 top-right, 3 bottom-left, 4 bottom-right.

        Args:
            rvec: (3,1) or (3,) rotation vector from estimate_board_pose
            tvec: (3,1) or (3,) translation from estimate_board_pose
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: distortion coefficients (use zeros for undistorted image)

        Returns:
            pts_2d: (4, 2) float32, pixel coordinates (u, v) for each corner
            corner_ids: (4,) int32, ids [1, 2, 3, 4] (empty array on failure)
        """
        empty_pts = np.zeros((0, 2), dtype=np.float32)
        empty_ids = np.array([], dtype=np.int32)
        corners_cam, corner_ids = self.compute_board_corners_in_camera_frame(rvec, tvec)
        if corners_cam.shape[0] != 4:
            return empty_pts, empty_ids
        dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        rvec_zero = np.zeros((3, 1), dtype=np.float64)
        tvec_zero = np.zeros((3, 1), dtype=np.float64)
        pts_2d, _ = cv2.projectPoints(
            corners_cam.reshape(4, 1, 3),
            rvec_zero,
            tvec_zero,
            camera_matrix,
            dist,
        )
        pts_2d = pts_2d.reshape(4, 2).astype(np.float32)
        return pts_2d, corner_ids

    def _get_circular_cutouts_board_frame(self) -> List[Tuple[np.ndarray, float]]:
        """Return list of (center_xyz, radius) in board frame for each circular cutout. center z=0."""
        out: List[Tuple[np.ndarray, float]] = []
        for cut in self.cutouts:
            if cut.get("geometry_id") != CUTOUT_GEOMETRY_ID_CIRCULAR:
                continue
            coeffs = cut.get("coefficients", [])
            if len(coeffs) < CIRCULAR_CUTOUT_NUM_COEFFICIENTS:
                continue
            cx, cy, radius = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
            center = np.array([cx, cy, 0.0], dtype=np.float64)
            out.append((center, radius))
        return out

    def draw_board_contour_and_circles(
        self,
        image: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        *,
        quad_thickness: int = 2,
        draw_corner_points: bool = True,
        corner_radius_px: int = 10,
        circles_thickness: int = 2,
    ) -> None:
        """
        Draw the board contour (quad + corner points) and circular cutouts in the image.

        Does not draw the pose axes; use draw_board_pose for that. Modifies image in place.

        Args:
            image: BGR image (modified in place)
            rvec: (3,1) or (3,) rotation vector from estimate_board_pose
            tvec: (3,1) or (3,) translation from estimate_board_pose
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: distortion coefficients (use zeros for undistorted image)
            quad_color: (B, G, R) for board outline and corner points
            quad_thickness: line thickness for the quad edges
            draw_corner_points: if True, draw a filled circle at each corner
            corner_radius_px: radius in pixels for corner points
            circles_color: (B, G, R) for circular cutout outlines
            circles_thickness: line thickness for cutout circles
        """
        if image is None or rvec is None or tvec is None:
            return
        dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        rvec_zero = np.zeros((3, 1), dtype=np.float64)
        tvec_zero = np.zeros((3, 1), dtype=np.float64)

        # Board contour (quad + corner points)
        corners_cam, _ = self.compute_board_corners_in_camera_frame(rvec, tvec)
        if corners_cam.shape[0] == 4:
            pts_2d, _ = cv2.projectPoints(
                corners_cam.reshape(4, 1, 3),
                rvec_zero,
                tvec_zero,
                camera_matrix,
                dist,
            )
            pts_2d = pts_2d.reshape(4, 2).astype(np.int32)
            for i in range(4):
                j = (i + 1) % 4
                cv2.line(image, tuple(pts_2d[i]), tuple(pts_2d[j]), self.board_color, quad_thickness, cv2.LINE_AA)
            if draw_corner_points:
                for pt in pts_2d:
                    cv2.circle(image, tuple(pt), corner_radius_px, self.board_color, -1, cv2.LINE_AA)

        # Circular cutouts
        circles = self._get_circular_cutouts_board_frame()
        if circles:
            R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
            t = np.asarray(tvec, dtype=np.float64).reshape(3,)
            for center_board, radius in circles:
                center_cam = (center_board.reshape(1, 3) @ R.T) + t[None, :]
                edge_board = center_board + np.array([radius, 0.0, 0.0], dtype=np.float64)
                edge_cam = (edge_board.reshape(1, 3) @ R.T) + t[None, :]
                pts = np.vstack([center_cam, edge_cam]).astype(np.float32).reshape(2, 1, 3)
                pts_2d, _ = cv2.projectPoints(pts, rvec_zero, tvec_zero, camera_matrix, dist)
                center_2d = pts_2d[0].ravel().astype(np.int32)
                radius_px = int(np.linalg.norm(pts_2d[1].ravel() - pts_2d[0].ravel()) + 0.5)
                if radius_px > 0:
                    cv2.circle(image, tuple(center_2d), radius_px, self.board_color, circles_thickness, cv2.LINE_AA)

    def draw_corners_id(
        self,
        image: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        *,
        corner_ids: Optional[np.ndarray] = None,
        font_scale: float = 0.8,
        font_thickness: int = 2,
        text_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Draw corner IDs (1, 2, 3, 4) at each board corner in the image.
        Modifies image in place.

        Args:
            image: BGR image (modified in place)
            rvec: (3,1) or (3,) rotation vector from estimate_board_pose
            tvec: (3,1) or (3,) translation from estimate_board_pose
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: distortion coefficients (use zeros for undistorted image)
            corner_ids: (4,) optional ids to draw; if None, uses [1, 2, 3, 4]
            font_scale: cv2.putText font scale
            font_thickness: cv2.putText thickness
            text_color: (B, G, R); if None, uses self.board_color
        """
        if image is None or rvec is None or tvec is None:
            return
        corners_cam, ids = self.compute_board_corners_in_camera_frame(rvec, tvec)
        if corners_cam.shape[0] != 4:
            return
        if corner_ids is not None and corner_ids.size == 4:
            ids = np.asarray(corner_ids, dtype=np.int32)
        dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        rvec_zero = np.zeros((3, 1), dtype=np.float64)
        tvec_zero = np.zeros((3, 1), dtype=np.float64)
        pts_2d, _ = cv2.projectPoints(
            corners_cam.reshape(4, 1, 3),
            rvec_zero,
            tvec_zero,
            camera_matrix,
            dist,
        )
        pts_2d = pts_2d.reshape(4, 2)
        color = text_color if text_color is not None else self.board_color
        for i in range(4):
            pt = pts_2d[i].astype(np.int32)
            label = str(int(ids[i]))
            # Offset text slightly so it sits next to the corner point
            text_pt = (pt[0] + 8, pt[1] - 8)
            cv2.putText(
                image, label, tuple(text_pt),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, font_thickness, cv2.LINE_AA,
            )

