
import cv2
import numpy as np

def rectangle_corners_square(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8, shift=0, fill_alpha=0.3):
    # Convert coordinates to ints
    x1, y1 = map(int, pt1)
    x2, y2 = map(int, pt2)

    # Ensure correct order
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Draw semi-transparent filled rectangle
    if fill_alpha > 0:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)

    width, height = x2 - x1, y2 - y1
    max_side = max(width, height)
    corner_length = int(max_side * 0.1)
    corner_length = min(corner_length, width // 2, height // 2)

    # Draw corners
    cv2.line(img, (x1, y1), (x1 + corner_length, y1), color, thickness, lineType, shift)
    cv2.line(img, (x1, y1), (x1, y1 + corner_length), color, thickness, lineType, shift)
    cv2.line(img, (x2, y1), (x2 - corner_length, y1), color, thickness, lineType, shift)
    cv2.line(img, (x2, y1), (x2, y1 + corner_length), color, thickness, lineType, shift)
    cv2.line(img, (x1, y2), (x1 + corner_length, y2), color, thickness, lineType, shift)
    cv2.line(img, (x1, y2), (x1, y2 - corner_length), color, thickness, lineType, shift)
    cv2.line(img, (x2, y2), (x2 - corner_length, y2), color, thickness, lineType, shift)
    cv2.line(img, (x2, y2), (x2, y2 - corner_length), color, thickness, lineType, shift)

    return img


def quad_corners_square(img, pts, color, thickness=1, lineType=cv2.LINE_8, shift=0, fill_alpha=0.3, corner_fraction=0.1):
    """
    Draw a quadrilateral from 4 corner points (same style as rectangle_corners_square):
    semi-transparent fill, 4 edges, and L-shaped corner brackets at each vertex.
    pts: list/tuple of 4 (x,y) in order (e.g. top-left, top-right, bottom-right, bottom-left).
    Edges and brackets align exactly with the given corners (no axis-aligned bounding box).
    """
    if len(pts) != 4:
        return img
    pts = np.array(pts, dtype=np.float64)
    pts_int = np.round(pts).astype(np.int32)

    # Semi-transparent fill
    if fill_alpha > 0:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts_int], color)
        cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)

    # Four edges
    for i in range(4):
        a = tuple(pts_int[i])
        b = tuple(pts_int[(i + 1) % 4])
        cv2.line(img, a, b, color, thickness, lineType, shift)

    # L-shaped corner brackets at each vertex (along the two edges meeting at that corner)
    for i in range(4):
        p = pts[i]
        prev_ = pts[(i - 1) % 4]
        next_ = pts[(i + 1) % 4]
        e1 = p - prev_
        e2 = next_ - p
        len1 = max(1e-6, np.linalg.norm(e1))
        len2 = max(1e-6, np.linalg.norm(e2))
        bracket_len = min(len1, len2) * corner_fraction
        bracket_len = min(bracket_len, len1 / 2, len2 / 2)
        end1 = p - e1 / len1 * bracket_len
        end2 = p + e2 / len2 * bracket_len
        cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(end1).astype(int)), color, thickness, lineType, shift)
        cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(end2).astype(int)), color, thickness, lineType, shift)

    return img
