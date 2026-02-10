from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

import open3d as o3d

class LidarDataProcessor:
    def __init__(self, yaml_path: str):
        self.cfg = self.load_yaml(yaml_path)
        self.validate_cfg(self.cfg)
        self.board_width = None
        self.board_height = None
        print(f"LidarDataProcessor initialized with config: {self.cfg}")
    
    def set_board_size(self, board_width: float, board_height: float):
        self.board_width = board_width
        self.board_height = board_height

    def process(self, xyz: np.ndarray) -> Dict[str, Any]:
        # validate xyz
        xyz = self.validate_xyz(xyz)
        raw_xyz = xyz.copy()

        # box filter (xmin/xmax, ymin/ymax, zmin/zmax)
        xyz = self.box_filter(xyz)

        # apply transform
        xyz = self.apply_transform(xyz)

        # apply preproc filters
        xyz = self.apply_preproc_filters(xyz)

        # normal estimation
        pcd = self.to_o3d(xyz)
        normals = self.estimate_normals(pcd)

        # region growing
        cluster_indices = self.region_growing(pcd, normals)

        # find board
        board = self.find_board(cluster_indices, xyz)

        return {
            "raw_xyz": raw_xyz.astype(np.float32, copy=False),
            "xyz": xyz.astype(np.float32, copy=False),
            "board": board,
        }

    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("YAML must define a mapping (dict) at the root.")
        return data

    @staticmethod
    def validate_cfg(cfg: Dict[str, Any]) -> None:
        required_blocks = ["box_filter", "transform", "preproc_filters",
                           "normal_estimation", "region_growing", "board_size_filter", "plane_ransac"]
        for b in required_blocks:
            if b not in cfg:
                raise ValueError(f"Missing YAML block: '{b}'")

        # transform
        T = cfg["transform"].get("T_4x4", None)
        if T is not None:
            T = np.asarray(T, dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError("transform.T_4x4 must be a 4x4 matrix (or null).")

    @staticmethod
    def validate_xyz(xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Expected Nx3 array, got shape {xyz.shape}")
        # remove NaN/Inf like PCL removeNaNFromPointCloud
        mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[mask]
        return xyz.astype(np.float32, copy=False)

    @staticmethod
    def to_o3d(xyz: np.ndarray) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        return pcd

    def box_filter(self, xyz: np.ndarray) -> np.ndarray:
        b = self.cfg["box_filter"]
        xmin = float(b.get("xmin", -np.inf))
        xmax = float(b.get("xmax", np.inf))
        ymin = float(b.get("ymin", -np.inf))
        ymax = float(b.get("ymax", np.inf))
        zmin = float(b.get("zmin", -np.inf))
        zmax = float(b.get("zmax", np.inf))
        keep = (
            (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax)
            & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
            & (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
        )
        return xyz[keep]

    def apply_transform(self, xyz: np.ndarray) -> np.ndarray:
        T_list = self.cfg["transform"].get("T_4x4", None)
        if T_list is None:
            return xyz
        T = np.asarray(T_list, dtype=np.float64)
        ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
        hom = np.hstack([xyz.astype(np.float64), ones])  # Nx4
        out = (T @ hom.T).T[:, :3]
        return out.astype(np.float32)

    def apply_transform(self, xyz: np.ndarray) -> np.ndarray:
        T_list = self.cfg["transform"].get("T_4x4", None)
        if T_list is None:
            return xyz
        T = np.asarray(T_list, dtype=np.float64)
        ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
        hom = np.hstack([xyz.astype(np.float64), ones])  # Nx4
        out = (T @ hom.T).T[:, :3]
        return out.astype(np.float32)

    def apply_preproc_filters(self, xyz: np.ndarray) -> np.ndarray:
        if xyz.shape[0] == 0:
            return xyz

        pcd = self.to_o3d(xyz)
        fcfg = self.cfg["preproc_filters"]

        # voxel downsample
        voxel_size = float(fcfg.get("voxel_size", 0.0))
        if voxel_size > 1e-6:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        # statistical outlier removal
        sor = fcfg.get("statistical_outlier_removal", {}) or {}
        if bool(sor.get("enabled", False)):
            nb = int(sor.get("nb_neighbors", 20))
            std = float(sor.get("std_ratio", 2.0))
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)

        # radius outlier removal
        ror = fcfg.get("radius_outlier_removal", {}) or {}
        if bool(ror.get("enabled", False)):
            nbp = int(ror.get("nb_points", 16))
            rad = float(ror.get("radius", 0.5))
            pcd, _ = pcd.remove_radius_outlier(nb_points=nbp, radius=rad)

        return np.asarray(pcd.points, dtype=np.float32)


    def estimate_normals(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        if len(pcd.points) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        ncfg = self.cfg["normal_estimation"]
        method = str(ncfg.get("method", "knn")).lower()

        if method == "radius":
            radius = float(ncfg.get("radius", 0.2))
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius)
            )
        else:
            knn = int(ncfg.get("knn", 30))
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
            )

        pcd.normalize_normals()
        return np.asarray(pcd.normals, dtype=np.float32)

    def region_growing(self, pcd: o3d.geometry.PointCloud, normals: np.ndarray) -> List[np.ndarray]:
        """
        Approximate PCL RegionGrowing:
          - Build KNN graph
          - Grow if:
              angle(n_u, n_v) <= smoothness_deg
              curvature(v) <= curvature_thresh
        Curvature proxy: 1 - ||mean_neighbor_normal||
        """
        pts = np.asarray(pcd.points, dtype=np.float32)
        N = pts.shape[0]
        if N == 0:
            return []

        rcfg = self.cfg["region_growing"]
        k = max(5, int(rcfg.get("number_neighbors", 30)))
        smooth_deg = float(rcfg.get("smoothness_deg", 7.5))
        curvature_thresh = float(rcfg.get("curvature_thresh", 0.12))

        tree = o3d.geometry.KDTreeFlann(pcd)

        neighbors: List[np.ndarray] = []
        for i in range(N):
            _, idxs, _ = tree.search_knn_vector_3d(pcd.points[i], k)
            neighbors.append(np.asarray(idxs, dtype=np.int32))

        # curvature proxy
        curv = np.zeros((N,), dtype=np.float32)
        for i in range(N):
            nn = normals[neighbors[i]]
            m = nn.mean(axis=0)
            curv[i] = 1.0 - float(np.linalg.norm(m) + 1e-9)

        smooth_rad = math.radians(smooth_deg)
        cos_thresh = float(math.cos(smooth_rad))

        visited = np.zeros((N,), dtype=bool)
        clusters: List[np.ndarray] = []

        for seed in range(N):
            if visited[seed]:
                continue
            # like PCL using indices that are not NaN; we already removed NaNs.
            if curv[seed] > curvature_thresh:
                visited[seed] = True
                continue

            q = [seed]
            visited[seed] = True
            cluster = [seed]

            while q:
                u = q.pop()
                nu = normals[u]
                nu = nu / (np.linalg.norm(nu) + 1e-9)

                for v in neighbors[u]:
                    if visited[v]:
                        continue
                    if curv[v] > curvature_thresh:
                        visited[v] = True
                        continue

                    nv = normals[v]
                    nv = nv / (np.linalg.norm(nv) + 1e-9)

                    if float(np.dot(nu, nv)) >= cos_thresh:
                        visited[v] = True
                        q.append(int(v))
                        cluster.append(int(v))

            cluster = np.asarray(cluster, dtype=np.int32)
            if cluster.size >= int(rcfg.get("cluster_size_min", 200)):
                clusters.append(cluster)

        return clusters

    def compute_obb(self, cluster_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pcd = self.to_o3d(cluster_xyz)
        obb = pcd.get_oriented_bounding_box()
        center = np.asarray(obb.center, dtype=np.float32)
        R = np.asarray(obb.R, dtype=np.float32)
        extent = np.asarray(obb.extent, dtype=np.float32)
        return center, R, extent

    def passes_board_size_filter(self, extent: np.ndarray) -> bool:

        if self.board_width is None or self.board_height is None:
            raise ValueError("Board size not set. Call set_board_size(1.2, 0.6) before process().")

        bcfg = self.cfg["board_size_filter"]
        bw = self.board_width
        bh = self.board_height

        wmin = bw - float(bcfg.get("width_tol_min", 0.05))
        wmax = bw + float(bcfg.get("width_tol_max", 0.05))
        hmin = bh - float(bcfg.get("height_tol_min", 0.05))
        hmax = bh + float(bcfg.get("height_tol_max", 0.05))

        # Choose two largest extents as planar dimensions
        dims = np.sort(np.asarray(extent, dtype=np.float64))[-2:]
        h = float(dims[0])
        w = float(dims[1])

        area = w * h
        min_area = wmin * hmin
        max_area = wmax * hmax

        aspect = max(w, h) / max(1e-9, min(w, h))
        max_aspect = max(wmax, hmax) / max(1e-9, min(wmin, hmin))

        return (min_area <= area <= max_area) and (aspect <= max_aspect)

    def fit_plane_and_project(self, cluster_xyz: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        pcfg = self.cfg["plane_ransac"]
        dist = float(pcfg.get("distance_threshold", 0.02))
        ransac_n = int(pcfg.get("ransac_n", 3))
        iters = int(pcfg.get("num_iterations", 1000))

        pcd = self.to_o3d(cluster_xyz)
        if len(pcd.points) < 3:
            return None, None

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=dist,
            ransac_n=ransac_n,
            num_iterations=iters,
        )
        if len(inliers) < 3:
            return None, None

        a, b, c, d = [float(x) for x in plane_model]
        n = np.array([a, b, c], dtype=np.float64)
        nn = np.linalg.norm(n) + 1e-12
        n = n / nn
        d = d / nn

        pts = cluster_xyz.astype(np.float64)
        signed_dist = (pts @ n) + d
        planar = pts - signed_dist[:, None] * n[None, :]

        plane_abcd = np.array([n[0], n[1], n[2], d], dtype=np.float32)
        return plane_abcd, planar.astype(np.float32)

    def choose_best_board_candidate(
        self,
        clusters_out: List[Dict[str, Any]],
        board_w: float = 1.2,
        board_h: float = 0.6
    ) -> Optional[Dict[str, Any]]:
        """
        Pick the best candidate among clusters_out using:
          1) how close OBB (planar dims) are to (board_w, board_h)
          2) larger plane support (more points) as tie-break

        Expects each cluster dict to include:
          - "obb_extent": (3,)
          - "planar_xyz": (N,3)  (projected points)
          - "cluster_xyz": (N,3)
        """
        if not clusters_out:
            return None

        bw = float(board_w)
        bh = float(board_h)
        target_area = bw * bh

        best = None
        best_score = float("inf")

        for c in clusters_out:
            extent = np.asarray(c["obb_extent"], dtype=np.float64).reshape(3,)
            dims = np.sort(extent)[-2:]  # take 2 largest as planar dims
            h = float(dims[0])
            w = float(dims[1])

            # Evaluate both assignments (w~bw,h~bh) and swapped, take min.
            e1 = (w - bw) ** 2 + (h - bh) ** 2
            e2 = (w - bh) ** 2 + (h - bw) ** 2
            dim_err = min(e1, e2)

            area = w * h
            area_err = (area - target_area) ** 2

            # Prefer candidates with more planar points (more support)
            npts = int(np.asarray(c["planar_xyz"]).shape[0])

            # Score: dimension fit + small area penalty - support bonus
            # (support bonus keeps bigger/more complete board ahead)
            score = dim_err + 0.25 * area_err - 1e-6 * npts

            if score < best_score:
                best_score = score
                best = c

        return best


    def estimate_board_pose_from_planar_cluster(
        self,
        planar_xyz: np.ndarray,
        plane_abcd: np.ndarray,
        board_w: float = 1.2,
        board_h: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate a stable 6-DoF pose (R,t) of the board in LiDAR frame from:
          - planar_xyz: Nx3 points projected onto the plane of the board
          - plane_abcd: [a,b,c,d] with (a,b,c) being unit normal, ax+by+cz+d=0

        Returns:
          R: (3,3) rotation matrix, columns are [x_axis(width), y_axis(height), z_axis(normal)]
          t: (3,) translation (board center)

        Approach:
          - z axis = plane normal (unit)
          - center = centroid of planar_xyz
          - in-plane axes = PCA on planar points (after removing center)
          - choose which PCA axis is width vs height based on board_w vs board_h
        """
        pts = np.asarray(planar_xyz, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
            raise ValueError("planar_xyz must be Nx3 with N>=3")

        plane = np.asarray(plane_abcd, dtype=np.float64).reshape(4,)
        n = plane[:3]
        n_norm = np.linalg.norm(n) + 1e-12
        z_axis = n / n_norm  # unit normal

        # Center (translation)
        center = pts.mean(axis=0)

        # PCA for in-plane directions
        X = pts - center[None, :]
        # covariance
        C = (X.T @ X) / max(1, (X.shape[0] - 1))
        eigvals, eigvecs = np.linalg.eigh(C)  # ascending eigvals
        # two largest eigenvectors span the plane
        v1 = eigvecs[:, 2]  # largest variance
        v2 = eigvecs[:, 1]  # second largest

        # Make them orthonormal and lie in plane
        v1 = v1 / (np.linalg.norm(v1) + 1e-12)
        # ensure v2 orthogonal to v1 and z
        v2 = v2 - np.dot(v2, v1) * v1
        v2 = v2 - np.dot(v2, z_axis) * z_axis
        v2 = v2 / (np.linalg.norm(v2) + 1e-12)

        # Ensure right-handed frame: x × y = z
        # First decide which is width axis (x) vs height axis (y).
        # Use spread along v1/v2 compared to expected board dimensions.
        bw = float(board_w)
        bh = float(board_h)

        # Project points onto v1 and v2 to estimate extents
        s1 = X @ v1
        s2 = X @ v2
        ext1 = float(s1.max() - s1.min())
        ext2 = float(s2.max() - s2.min())

        # Decide assignment by closeness to (bw,bh) (allow swap)
        err_a = (ext1 - bw) ** 2 + (ext2 - bh) ** 2
        err_b = (ext1 - bh) ** 2 + (ext2 - bw) ** 2

        if err_a <= err_b:
            x_axis = v1
            y_axis = v2
        else:
            x_axis = v2
            y_axis = v1

        # Rebuild y to ensure orthonormal right-handed with z
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
        # Final x again to eliminate numerical drift
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)

        R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)
        t = center.astype(np.float64)
        return R.astype(np.float32), t.astype(np.float32)



    def compute_board_corners_from_pose(
        self,
        R: np.ndarray,
        t: np.ndarray,
        w: float = 1.2,
        h: float = 0.6
    ):
        """
        Compute board corners in LiDAR frame with stable canonical ordering.

        LiDAR convention:
        +X forward
        +Y left
        +Z up

        Output order (ALWAYS):
            1 -------- 2
            |          |
            3 -------- 4
        """

        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3)

        hw, hh = 0.5 * w, 0.5 * h

        # --- Step 1: Raw geometry (do NOT trust orientation yet) ---
        corners_board = np.array([
            [-hw,  hh, 0.0],
            [ hw,  hh, 0.0],
            [-hw, -hh, 0.0],
            [ hw, -hh, 0.0],
        ], dtype=np.float64)

        corners_lidar = (corners_board @ R.T) + t
        center = corners_lidar.mean(axis=0)

        # --- Step 2: Build canonical board frame in LiDAR ---

        # Board normal
        normal = R[:, 2]
        if normal[2] < 0:   # force upward (+Z)
            normal = -normal
        normal /= np.linalg.norm(normal)

        # Board up = projection of LiDAR +Z onto board plane
        lidar_up = np.array([0.0, 0.0, 1.0])
        up = lidar_up - np.dot(lidar_up, normal) * normal
        up /= np.linalg.norm(up)

        # Board right = in-plane orthogonal
        right = np.cross(up, normal)

        # Enforce LiDAR convention:
        # +Y is LEFT → board-right must point toward -Y
        if right[1] > 0:
            right = -right

        right /= np.linalg.norm(right)

        # --- Step 3: Stable corner assignment ---
        rel = corners_lidar - center
        up_proj = rel @ up
        right_proj = rel @ right

        top = up_proj > 0
        bottom = ~top

        top_pts = corners_lidar[top]
        bot_pts = corners_lidar[bottom]

        top_r = right_proj[top]
        bot_r = right_proj[bottom]

        # Final canonical order
        c1 = top_pts[np.argmin(top_r)]     # top-left
        c2 = top_pts[np.argmax(top_r)]     # top-right
        c3 = bot_pts[np.argmin(bot_r)]     # bottom-left
        c4 = bot_pts[np.argmax(bot_r)]     # bottom-right

        corners_ordered = np.vstack([c1, c2, c3, c4]).astype(np.float32)
        corner_ids = np.array([1, 2, 3, 4], dtype=np.int32)

        return corners_ordered, corner_ids

    def find_board(self, cluster_indices, xyz) -> Dict[str, Any]:
        clusters_out: List[Dict[str, Any]] = []
        roi_parts: List[np.ndarray] = []

        for inds in cluster_indices:
            if inds.size < 3:
                continue
            if not (self.cfg["region_growing"]["cluster_size_min"] <= int(inds.size) <=
                    self.cfg["region_growing"]["cluster_size_max"]):
                continue

            cluster_xyz = xyz[inds]
            obb_center, obb_R, obb_extent = self.compute_obb(cluster_xyz)

            if not self.passes_board_size_filter(obb_extent):
                continue

            roi_parts.append(cluster_xyz)

            plane_abcd, planar_xyz = self.fit_plane_and_project(cluster_xyz)
            if plane_abcd is None:
                continue

            clusters_out.append({
                "indices": inds.astype(np.int32, copy=False),
                "cluster_xyz": cluster_xyz.astype(np.float32, copy=False),
                "obb_center": obb_center.astype(np.float32, copy=False),
                "obb_R": obb_R.astype(np.float32, copy=False),
                "obb_extent": obb_extent.astype(np.float32, copy=False),
                "plane_abcd": plane_abcd.astype(np.float32, copy=False),
                "planar_xyz": planar_xyz.astype(np.float32, copy=False),
            })

        roi_cloud_xyz = (
            np.concatenate(roi_parts, axis=0).astype(np.float32, copy=False)
            if roi_parts else np.zeros((0, 3), dtype=np.float32)
        )

        # --- NEW: choose best candidate + compute pose + corners
        bw = self.board_width
        bh = self.board_height

        best = self.choose_best_board_candidate(clusters_out, board_w=bw, board_h=bh)

        pose_R = None
        pose_t = None
        corners_3d = None
        corner_ids = None

        if best is not None:
            pose_R, pose_t = self.estimate_board_pose_from_planar_cluster(
                best["planar_xyz"], best["plane_abcd"], board_w=bw, board_h=bh
            )
            corners_3d, corner_ids = self.compute_board_corners_from_pose(pose_R, pose_t, w=bw, h=bh)

        return {
            "roi_cloud_xyz": roi_cloud_xyz,
            "candidates": clusters_out,
            "best_candidate": best,
            "R": pose_R,          # (3,3) or None
            "t": pose_t,          # (3,) or None
            "corners_3d": corners_3d,  # (4,3) or None
            "corner_ids": corner_ids,  # (4,) int32 [1,2,3,4] or None
        }