import argparse
import os
import cv2
import numpy as np
import yaml


# -------------------------
# Args (images only, auto only)
# -------------------------
parser = argparse.ArgumentParser(description="Camera Intrinsic Calibration (ChArUco, images folder auto)")
parser.add_argument("-type", "--type", dest="CAMERA_TYPE", default="fisheye", type=str, help="Camera Type: fisheye/normal")
parser.add_argument("-path", "--path", dest="INPUT_PATH", default="./data/", type=str, help="Input images folder")

parser.add_argument("-subpix", "--subpix", dest="SUBPIX_REGION", default=5, type=int, help="Subpix window half-size (pixels)")
parser.add_argument("-min_pts", "--min_pts", dest="MIN_CHARUCO_CORNERS", default=6, type=int, help="Min ChArUco corners to accept image")

parser.add_argument("-store", "--store", dest="STORE_FLAG", action="store_true", help="Store valid images")
parser.add_argument("-store_path", "--store_path", dest="STORE_PATH", default="./selected/", type=str, help="Path to store valid images")

parser.add_argument("--save_undist", action="store_true", help="Save undistorted images")
parser.add_argument("--undist_path", default="./undist/", type=str, help="Output folder for undistorted images")

parser.add_argument("--yaml", "-o", dest="YAML_PATH", default=None, type=str, help="Output YAML file for calibration (default: calibration.yaml)")


args = parser.parse_args()


# -------------------------
# Board Config (ChArUco)
# -------------------------
class BoardConfig:
    ROWS = 8
    COLS = 11
    CHECKER_SIZE = 20.0  # mm
    MARKER_SIZE = 15.0   # mm
    ARUCO_DICT = cv2.aruco.DICT_4X4_250

    @classmethod
    def get_charuco_board(cls):
        dictionary = cv2.aruco.getPredefinedDictionary(cls.ARUCO_DICT)
        try:
            board = cv2.aruco.CharucoBoard(
                (cls.COLS, cls.ROWS),
                cls.CHECKER_SIZE / 1000.0,
                cls.MARKER_SIZE / 1000.0,
                dictionary,
            )
        except AttributeError:
            board = cv2.aruco.CharucoBoard_create(
                cls.COLS, cls.ROWS,
                cls.CHECKER_SIZE / 1000.0,
                cls.MARKER_SIZE / 1000.0,
                dictionary,
            )
        return board, dictionary


# -------------------------
# Helpers
# -------------------------
def get_all_images(folder: str):
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise RuntimeError(f"INPUT_PATH does not exist: {folder}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    files = []

    for root, _, names in os.walk(folder):
        for name in names:
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                files.append(os.path.join(root, name))

    files.sort()
    if not files:
        raise RuntimeError(f"No images found under: {folder}")

    print(f"[INFO] Found {len(files)} images under {folder}")
    return files



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_valid_image(src_path: str, img: np.ndarray, out_dir: str):
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(src_path))
    cv2.imwrite(out_path, img)


# -------------------------
# Calibration containers
# -------------------------
class CalibData:
    def __init__(self):
        self.type = None
        self.camera_mat = None
        self.dist_coeff = None
        self.rvecs = None
        self.tvecs = None
        self.map1 = None
        self.map2 = None
        self.reproj_err = None
        self.ok = False
        self.frame_size = None  # (w,h)


# -------------------------
# ChArUco calibrator
# -------------------------
class InCalibrator:
    def __init__(self, camera_type: str):
        if camera_type not in ("fisheye", "normal"):
            raise ValueError("CAMERA_TYPE must be 'fisheye' or 'normal'")

        self.camera_type = camera_type
        self.data = CalibData()

        self.board, self.dictionary = BoardConfig.get_charuco_board()
        self.objpoints = []  # list of (N,1,3)
        self.imgpoints = []  # list of (N,1,2)

    @staticmethod
    def _aruco_detector(dictionary):
        try:
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, params)
            return detector, params
        except AttributeError:
            params = cv2.aruco.DetectorParameters_create()
            return None, params

    def detect_charuco(self, bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        detector, params = self._aruco_detector(self.dictionary)
        if detector is not None:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=params)

        if ids is None or len(ids) == 0:
            return False, None, None, corners, ids, None, None

        # ---- Interpolate ChArUco corners: support old/new OpenCV return formats ----
        try:
            out = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
        except Exception:
            out = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board, None, None)

        if isinstance(out[0], (int, np.integer)):
            # OpenCV 4.5.x style: (retval, charucoCorners, charucoIds)
            _, charuco_corners, charuco_ids = out[:3]
        else:
            # Newer style: (charucoCorners, charucoIds, ...)
            charuco_corners, charuco_ids = out[:2]

        if charuco_ids is None or charuco_corners is None:
            return False, None, None, corners, ids, None, None

        # Force expected dtype/shape
        charuco_corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
        charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)

        if charuco_corners.shape[0] == 0 or charuco_ids.shape[0] == 0:
            return False, None, None, corners, ids, charuco_corners, charuco_ids

        if charuco_ids.shape[0] < args.MIN_CHARUCO_CORNERS:
            return False, None, None, corners, ids, charuco_corners, charuco_ids

        # Subpix refine (safe now)
        win = (args.SUBPIX_REGION, args.SUBPIX_REGION)
        cv2.cornerSubPix(
            gray,
            charuco_corners,
            win,
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )

        # Object points for these IDs
        try:
            all_obj = self.board.getChessboardCorners()
        except AttributeError:
            all_obj = self.board.chessboardCorners

        obj = all_obj[charuco_ids.flatten()].reshape(-1, 1, 3).astype(np.float32)
        img = charuco_corners.reshape(-1, 1, 2).astype(np.float32)

        return True, obj, img, corners, ids, charuco_corners, charuco_ids

    def draw(self, bgr, ar_c, ar_i, ch_c=None, ch_i=None):
        out = bgr.copy()
        if ar_i is not None and len(ar_i) > 0:
            cv2.aruco.drawDetectedMarkers(out, ar_c, ar_i)
        if ch_c is not None and ch_i is not None:
            try:
                cv2.aruco.drawDetectedCornersCharuco(out, ch_c, ch_i)
            except Exception:
                pass
        return out

    def update(self, frame_size_wh):
        w, h = frame_size_wh
        self.data.frame_size = (w, h)

        if self.camera_type == "fisheye":
            self._calibrate_fisheye(w, h)
        else:
            self._calibrate_normal(w, h)

        self._calc_reproj_error()
        self._compute_undistort_maps()

    def _calibrate_fisheye(self, w, h):
        d = self.data
        d.type = "FISHEYE"

        K = np.eye(3, dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)

        flags = cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC

        try:
            ok, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                self.objpoints, self.imgpoints, (w, h),
                K, D, flags=flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 1e-7),
            )
        except cv2.error as e:
            raise RuntimeError(
                "Fisheye calibrate failed (InitExtrinsics). This usually happens when you have too few views "
                "or not enough pose diversity. Add more images (tilt/rotate board, different distances)."
            ) from e

        d.ok = bool(ok) and cv2.checkRange(K) and cv2.checkRange(D)
        d.camera_mat, d.dist_coeff, d.rvecs, d.tvecs = K, D, rvecs, tvecs


    def _calibrate_normal(self, w, h):
        d = self.data
        d.type = "NORMAL"

        K = np.eye(3, dtype=np.float64)
        D = np.zeros((5, 1), dtype=np.float64)

        ok, K, D, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            (w, h),
            K,
            D,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 1e-7),
        )

        d.ok = bool(ok) and cv2.checkRange(K) and cv2.checkRange(D)
        d.camera_mat, d.dist_coeff, d.rvecs, d.tvecs = K, D, rvecs, tvecs

    def _calc_reproj_error(self):
        d = self.data
        if not d.ok:
            d.reproj_err = None
            return

        errs = []
        for i in range(len(self.objpoints)):
            obj = self.objpoints[i]
            img = self.imgpoints[i]

            if self.camera_type == "fisheye":
                proj, _ = cv2.fisheye.projectPoints(obj, d.rvecs[i], d.tvecs[i], d.camera_mat, d.dist_coeff)
            else:
                proj, _ = cv2.projectPoints(obj, d.rvecs[i], d.tvecs[i], d.camera_mat, d.dist_coeff)

            e = cv2.norm(proj, img, cv2.NORM_L2) / len(proj)
            errs.append(float(e))

        d.reproj_err = errs

    def _compute_undistort_maps(self):
        d = self.data
        w, h = d.frame_size
        if self.camera_type == "fisheye":
            d.map1, d.map2 = cv2.fisheye.initUndistortRectifyMap(
                d.camera_mat, d.dist_coeff, np.eye(3), d.camera_mat, (w, h), cv2.CV_16SC2
            )
        else:
            d.map1, d.map2 = cv2.initUndistortRectifyMap(
                d.camera_mat, d.dist_coeff, np.eye(3), d.camera_mat, (w, h), cv2.CV_16SC2
            )

    def undistort(self, bgr: np.ndarray):
        d = self.data
        if d.map1 is None or d.map2 is None:
            return bgr
        return cv2.remap(bgr, d.map1, d.map2, cv2.INTER_LINEAR)


# -------------------------
# Main (AUTO only)
# -------------------------
def main():
    images = get_all_images(args.INPUT_PATH)
    print("[INFO] First 10 files:\n  " + "\n  ".join(images[:10]))

    calibrator = InCalibrator(args.CAMERA_TYPE)

    frame_size = None
    processed = 0
    valid = 0
    valid_image_paths = []  # only these get undistorted and stored as fisheye|undist

    for p in images:
        processed += 1
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Could not read: {p}")
            continue

        if frame_size is None:
            h, w = img.shape[:2]
            frame_size = (w, h)

        ok, obj, imgp, ar_c, ar_i, ch_c, ch_i = calibrator.detect_charuco(img)

        # visualize
        vis = calibrator.draw(img, ar_c, ar_i, ch_c, ch_i)
        cv2.namedWindow("raw_frame", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("raw_frame", vis)

        if ok:
            valid += 1
            valid_image_paths.append(p)
            calibrator.objpoints.append(obj)
            calibrator.imgpoints.append(imgp)

            if args.STORE_FLAG:
                save_valid_image(p, img, args.STORE_PATH)

            print(f"[OK] valid={valid}/{processed}  file={os.path.basename(p)}  corners={len(imgp)}")
        else:
            n_aruco = 0 if ar_i is None else len(ar_i)
            n_char  = 0 if ch_i is None else len(ch_i)
            print(f"[SKIP] valid={valid}/{processed}  file={os.path.basename(p)}  aruco={n_aruco}  charuco={n_char}")

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()

    if frame_size is None:
        raise RuntimeError("Could not determine frame size from images.")
    if valid == 0:
        raise RuntimeError("No valid ChArUco detections found.")

    # IMPORTANT: require enough views for fisheye
    min_views = 10 if args.CAMERA_TYPE == "fisheye" else 5
    if valid < min_views:
        raise RuntimeError(
            f"Not enough valid images for {args.CAMERA_TYPE} calibration: got {valid}, need at least {min_views}. "
            "Take more images with different tilts/positions."
        )

    # Calibrate ONCE using all valid images
    calibrator.update(frame_size)


    if args.save_undist:
        ensure_dir(args.undist_path)
        # Only for valid images: one image = left fisheye, right undistorted
        for p in valid_image_paths:
            img = cv2.imread(p)
            if img is None:
                continue

            und = calibrator.undistort(img)
            # side-by-side: fisheye | undist
            composite = np.hstack([img, und])

            base = os.path.splitext(os.path.basename(p))[0]
            out_name = base + "_fisheye_undist.jpg"
            out_path = os.path.join(args.undist_path, out_name)
            cv2.imwrite(out_path, composite)

        print(f"[INFO] Saved valid images (fisheye|undist in one image) to: {os.path.abspath(args.undist_path)} ({len(valid_image_paths)} images)")


    d = calibrator.data
    print("\nCalibration Complete")
    print(f"Total images found: {len(images)}")
    print(f"Images processed: {processed}")
    print(f"Valid images used: {valid}")
    print(f"Type: {d.type}")
    print(f"Frame size (w,h): {d.frame_size}")
    print(f"Camera Matrix (K):\n{d.camera_mat}")
    print(f"Dist Coeff (D):\n{d.dist_coeff}")
    if d.reproj_err:
        print(f"Mean Reprojection Error: {np.mean(d.reproj_err):.6f}")

    np.save("camera_K.npy", d.camera_mat)
    np.save("camera_D.npy", d.dist_coeff)

    # Save calibration to YAML
    yaml_path = args.YAML_PATH if args.YAML_PATH else "calibration.yaml"
    calib_yaml = {
        "image_width": int(d.frame_size[0]),
        "image_height": int(d.frame_size[1]),
        "camera_name": "camera",
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": d.camera_mat.flatten().tolist(),
        },
        "distortion_model": "plumb_bob" if d.type == "NORMAL" else "equidistant",
        "distortion_coefficients": {
            "rows": int(d.dist_coeff.shape[0]),
            "cols": 1,
            "data": d.dist_coeff.flatten().tolist(),
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": np.eye(3).flatten().tolist(),
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": np.hstack([d.camera_mat, np.zeros((3, 1))]).flatten().tolist(),
        },
        "camera_type": d.type.lower(),
        "num_images_used": valid,
        "mean_reprojection_error": float(np.mean(d.reproj_err)) if d.reproj_err else None,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(calib_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Calibration saved to YAML: {os.path.abspath(yaml_path)}")


if __name__ == "__main__":
    main()
