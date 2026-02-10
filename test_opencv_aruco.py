import cv2

print("OpenCV version:", cv2.__version__)

# Check for ArUco module and specific functions
has_aruco = hasattr(cv2, "aruco")
has_est_pose_board = has_aruco and hasattr(cv2.aruco, "estimatePoseBoard")
has_est_single_markers = has_aruco and hasattr(cv2.aruco, "estimatePoseSingleMarkers")

print("Has cv2.aruco module?:", has_aruco)
print("Has estimatePoseBoard?:", has_est_pose_board)
print("Has estimatePoseSingleMarkers?:", has_est_single_markers)

# Optionally print the function objects
if has_est_pose_board:
    print("cv2.aruco.estimatePoseBoard =", cv2.aruco.estimatePoseBoard)

if has_est_single_markers:
    print("cv2.aruco.estimatePoseSingleMarkers =", cv2.aruco.estimatePoseSingleMarkers)
