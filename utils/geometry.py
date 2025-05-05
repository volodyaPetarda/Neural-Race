import math
from typing import List, Tuple

import numpy
import numpy as np

from entities.figures import Segment, Rectangle, Vector2D, ccw, on_segment, Ray, dot
from utils.profile import timeit


def segment_intersect(segment1: Segment, segment2: Segment) -> Vector2D | None:
    a1 = segment1.start
    a2 = segment1.end
    b1 = segment2.start
    b2 = segment2.end

    o1 = ccw(a1, a2, b1)
    o2 = ccw(a1, a2, b2)
    o3 = ccw(b1, b2, a1)
    o4 = ccw(b1, b2, a2)
    epsilon = 1e-9

    # Case 1: General crossing
    if (o1 * o2 < -epsilon) and (o3 * o4 < -epsilon):
        dir_a = a2 - a1
        dir_b = b2 - b1
        denominator = dir_a.x * dir_b.y - dir_a.y * dir_b.x
        if abs(denominator) < 1e-9:
            return None  # Parallel
        t = ((b1.x - a1.x) * dir_b.y - (b1.y - a1.y) * dir_b.x) / denominator
        intersection = Vector2D(a1.x + t * dir_a.x, a1.y + t * dir_a.y)
        return intersection

    # Case 2: Collinear overlap
    if (abs(o1) < epsilon and abs(o2) < epsilon and
            abs(o3) < epsilon and abs(o4) < epsilon):
        # Bounding box checks
        seg1_min_x, seg1_max_x = sorted([a1.x, a2.x])
        seg2_min_x, seg2_max_x = sorted([b1.x, b2.x])
        seg1_min_y, seg1_max_y = sorted([a1.y, a2.y])
        seg2_min_y, seg2_max_y = sorted([b1.y, b2.y])

        if not (seg1_max_x >= seg2_min_x and seg2_max_x >= seg1_min_x and
                seg1_max_y >= seg2_min_y and seg2_max_y >= seg1_min_y):
            return None

        # Midpoint calculation
        dir_a = a2 - a1
        if dir_a.magnitude() < 1e-9:
            return a1 if (a1 == b1 or a1 == b2) else None
        t_b1 = dot(b1 - a1, dir_a) / dot(dir_a, dir_a)
        t_b2 = dot(b2 - a1, dir_a) / dot(dir_a, dir_a)
        t_start = max(min(t_b1, t_b2), 0.0)
        t_end = min(max(t_b1, t_b2), 1.0)
        if t_start > t_end:
            return None
        t_mid = (t_start + t_end) / 2
        return a1 + dir_a * t_mid

    # Case 3: Check endpoints with epsilon
    for p, s, e in [(a1, b1, b2), (a2, b1, b2), (b1, a1, a2), (b2, a1, a2)]:
        if on_segment(s, e, p, epsilon):
            return p

    return None


def segment_batch_intersect(segment: Segment, precomputed_segments: tuple) -> bool:
    seg_aabbs, seg_vecs, seg_starts = precomputed_segments
    a1 = np.array([segment.x1, segment.y1])
    a2 = np.array([segment.x2, segment.y2])
    dir_a = a2 - a1

    input_min = np.minimum(a1, a2)
    input_max = np.maximum(a1, a2)
    mask = (
            (seg_aabbs[:, 0] <= input_max[0]) &
            (seg_aabbs[:, 2] >= input_min[0]) &
            (seg_aabbs[:, 1] <= input_max[1]) &
            (seg_aabbs[:, 3] >= input_min[1])
    )

    if not np.any(mask):
        return False

    seg_starts = seg_starts[mask]
    seg_vecs = seg_vecs[mask]

    o1 = dir_a[0] * (seg_starts[:, 1] - a1[1]) - dir_a[1] * (seg_starts[:, 0] - a1[0])
    o2 = dir_a[0] * (seg_starts[:, 1] + seg_vecs[:, 1] - a1[1]) - dir_a[1] * (seg_starts[:, 0] + seg_vecs[:, 0] - a1[0])
    o3 = seg_vecs[:, 0] * (a1[1] - seg_starts[:, 1]) - seg_vecs[:, 1] * (a1[0] - seg_starts[:, 0])
    o4 = seg_vecs[:, 0] * (a2[1] - seg_starts[:, 1]) - seg_vecs[:, 1] * (a2[0] - seg_starts[:, 0])

    epsilon = 1e-9
    mask_general = (o1 * o2 < -epsilon) & (o3 * o4 < -epsilon)

    denominators = dir_a[0] * seg_vecs[:, 1] - dir_a[1] * seg_vecs[:, 0]
    valid_denom = np.abs(denominators) > epsilon
    mask_general &= valid_denom

    if not np.any(mask_general):
        return False

    t_num = (seg_starts[:, 0] - a1[0]) * seg_vecs[:, 1] - (seg_starts[:, 1] - a1[1]) * seg_vecs[:, 0]
    t = np.divide(t_num, denominators, where=mask_general, out=np.zeros_like(denominators))

    s_num = (seg_starts[:, 0] - a1[0]) * dir_a[1] - (seg_starts[:, 1] - a1[1]) * dir_a[0]
    s = np.divide(s_num, denominators, where=mask_general, out=np.zeros_like(denominators))

    valid = mask_general & (t >= -epsilon) & (t <= 1 + epsilon) & (s >= -epsilon) & (s <= 1 + epsilon)

    return np.any(valid)

def is_segment_rectangle_intersection(segment: Segment, rectangle: Rectangle) -> bool:
    def is_point_inside(p: Vector2D) -> bool:
        translated = Vector2D(p.x - rectangle.center.x, p.y - rectangle.center.y)
        cos_theta = math.cos(rectangle.angle)
        sin_theta = math.sin(rectangle.angle)
        local_x = translated.x * cos_theta + translated.y * sin_theta
        local_y = -translated.x * sin_theta + translated.y * cos_theta
        return (abs(local_x) <= rectangle.x_len / 2 and
                abs(local_y) <= rectangle.y_len / 2)

    if is_point_inside(segment.start) or is_point_inside(segment.end):
        return True

    for side in rectangle.get_sides():
        if segment_intersect(segment, side):
            return True

    return False

def is_batch_segments_rectangle_intersection(seg_points: numpy.array, rectangle: Rectangle) -> bool:
    center = np.array([rectangle.center.x, rectangle.center.y])
    x_half = rectangle.x_len / 2
    y_half = rectangle.y_len / 2
    cos_theta = np.cos(rectangle.angle)
    sin_theta = np.sin(rectangle.angle)
    rot_matrix = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

    translated = seg_points - center
    local_coords = translated @ rot_matrix.T  # Rotate to local coordinates

    # Check if any endpoints are inside the rectangle
    in_x = (local_coords[:, :, 0] >= -x_half) & (local_coords[:, :, 0] <= x_half)
    in_y = (local_coords[:, :, 1] >= -y_half) & (local_coords[:, :, 1] <= y_half)
    point_inside = (in_x & in_y).any(axis=1)
    if point_inside.any():
        return True

    # Check remaining segments for line-rectangle intersection
    mask = ~point_inside
    if not mask.any():
        return False

    p0 = local_coords[mask, 0]
    p1 = local_coords[mask, 1]
    d = p1 - p0

    dir_x = d[:, 0]
    dir_y = d[:, 1]

    # Precompute inverse directions to avoid division
    inv_dir_x = np.divide(1.0, dir_x, where=np.abs(dir_x) > 1e-9, out=np.full_like(dir_x, np.inf))
    inv_dir_y = np.divide(1.0, dir_y, where=np.abs(dir_y) > 1e-9, out=np.full_like(dir_y, np.inf))

    tx1 = (-x_half - p0[:, 0]) * inv_dir_x
    tx2 = (x_half - p0[:, 0]) * inv_dir_x
    tx_min = np.minimum(tx1, tx2)
    tx_max = np.maximum(tx1, tx2)

    ty1 = (-y_half - p0[:, 1]) * inv_dir_y
    ty2 = (y_half - p0[:, 1]) * inv_dir_y
    ty_min = np.minimum(ty1, ty2)
    ty_max = np.maximum(ty1, ty2)

    t_entry = np.maximum(tx_min, ty_min)
    t_exit = np.minimum(tx_max, ty_max)

    # Check for valid intersections
    valid = (
        (t_entry <= t_exit) &
        (t_entry <= 1.0) &
        (t_exit >= 0.0)
    )

    # Handle segments with direction components near zero
    mask_dir_x_zero = np.abs(dir_x) < 1e-9
    x_inside = (p0[:, 0] >= -x_half) & (p0[:, 0] <= x_half)
    valid &= ~(mask_dir_x_zero & ~x_inside)

    mask_dir_y_zero = np.abs(dir_y) < 1e-9
    y_inside = (p0[:, 1] >= -y_half) & (p0[:, 1] <= y_half)
    valid &= ~(mask_dir_y_zero & ~y_inside)

    return valid.any()

def ray_segment_intersection(
        ray: Ray,
        segment: Segment,
        epsilon: float = 1e-6
) -> Vector2D | None:
    d = ray.direction

    denominator = d.x * (segment.y1 - segment.y2) + d.y * (segment.x2 - segment.x1)

    if abs(denominator) < epsilon:
        if on_segment(segment.start, segment.end, ray.start):
            return ray.start
        return None

    t_numerator = (segment.x1 - ray.start.x) * (segment.y1 - segment.y2) - (
            ray.start.y - segment.y1) * (segment.x2 - segment.x1)
    t = t_numerator / denominator

    s_numerator = d.x * (segment.y1 - ray.start.y) - d.y * (segment.x1 - ray.start.x)
    s = s_numerator / denominator

    if t >= -epsilon and -epsilon <= s <= 1 + epsilon:
        intersection = Vector2D(
            ray.start.x + t * d.x,
            ray.start.y + t * d.y
        )
        return intersection

    return None

def get_rays_intersect_segment(rays: List[Ray], segment: Segment) -> list[Vector2D | None]:
    return [ray_segment_intersection(ray, segment) for ray in rays]

def get_batch_ray_intersect_segments(
        precomputed_rays: tuple,
        precomputed_segments: tuple,
        epsilon: float = 1e-6
) -> list[Vector2D | None]:
    n_rays = precomputed_rays[0].shape[0]
    n_segments = precomputed_segments[0].shape[0]

    if n_rays == 0 or n_segments == 0:
        return [None] * n_rays

    seg_aabbs, seg_vecs, seg_starts = precomputed_segments

    ray_starts, ray_dirs, inv_dirs = precomputed_rays

    aabb_mask = _calculate_aabb_intersections(
        ray_starts, inv_dirs, seg_aabbs
    )

    return _calculate_intersections(
        ray_starts, ray_dirs, seg_starts, seg_vecs, aabb_mask, epsilon
    )

def _precompute_segment_data(segments: List[Segment]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_segments = len(segments)
    seg_aabbs = np.empty((n_segments, 4), dtype=np.float64)
    seg_starts = np.empty((n_segments, 2), dtype=np.float64)
    seg_ends = np.empty((n_segments, 2), dtype=np.float64)

    for i, seg in enumerate(segments):
        x1, y1 = seg.x1, seg.y1
        x2, y2 = seg.x2, seg.y2
        seg_starts[i] = [x1, y1]
        seg_ends[i] = [x2, y2]
        seg_aabbs[i] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    seg_vecs = seg_ends - seg_starts
    return seg_aabbs, seg_vecs, seg_starts


def _precompute_ray_data(rays: List[Ray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rays = len(rays)
    ray_starts = np.empty((n_rays, 2), dtype=np.float64)
    ray_dirs = np.empty((n_rays, 2), dtype=np.float64)

    for i, ray in enumerate(rays):
        ray_starts[i] = [ray.start.x, ray.start.y]
        dir_vec = ray.direction
        ray_dirs[i] = [dir_vec.x, dir_vec.y]

    inv_dirs = np.divide(1.0, ray_dirs,
                         where=np.abs(ray_dirs) > 1e-9,
                         out=np.full_like(ray_dirs, np.inf))
    return ray_starts, ray_dirs, inv_dirs


def _calculate_aabb_intersections(
        ray_starts: np.ndarray,
        inv_dirs: np.ndarray,
        seg_aabbs: np.ndarray
) -> np.ndarray:
    tx1 = (seg_aabbs[np.newaxis, :, 0] - ray_starts[:, np.newaxis, 0]) * inv_dirs[:, np.newaxis, 0]
    tx2 = (seg_aabbs[np.newaxis, :, 2] - ray_starts[:, np.newaxis, 0]) * inv_dirs[:, np.newaxis, 0]
    tmin = np.minimum(tx1, tx2)
    tmax = np.maximum(tx1, tx2)

    ty1 = (seg_aabbs[np.newaxis, :, 1] - ray_starts[:, np.newaxis, 1]) * inv_dirs[:, np.newaxis, 1]
    ty2 = (seg_aabbs[np.newaxis, :, 3] - ray_starts[:, np.newaxis, 1]) * inv_dirs[:, np.newaxis, 1]
    tmin = np.maximum(tmin, np.minimum(ty1, ty2))
    tmax = np.minimum(tmax, np.maximum(ty1, ty2))

    return (tmax >= tmin) & (tmax >= 0)


def _calculate_intersections(
        ray_starts: np.ndarray,
        ray_dirs: np.ndarray,
        seg_starts: np.ndarray,
        seg_vecs: np.ndarray,
        aabb_mask: np.ndarray,
        epsilon: float
) -> list[Vector2D | None]:
    n_rays, n_segments = aabb_mask.shape

    p = ray_starts[:, np.newaxis, :]
    d = ray_dirs[:, np.newaxis, :]
    a = seg_starts[np.newaxis, :, :]
    v = seg_vecs[np.newaxis, :, :]

    denominator = d[..., 0] * v[..., 1] - d[..., 1] * v[..., 0]

    t_num = (a[..., 0] - p[..., 0]) * v[..., 1] - (a[..., 1] - p[..., 1]) * v[..., 0]
    s_num = (a[..., 0] - p[..., 0]) * d[..., 1] - (a[..., 1] - p[..., 1]) * d[..., 0]

    valid_mask = aabb_mask & (np.abs(denominator) > epsilon)
    t = np.full_like(denominator, -np.inf)
    s = np.full_like(denominator, -np.inf)
    np.divide(t_num, denominator, out=t, where=valid_mask)
    np.divide(s_num, denominator, out=s, where=valid_mask)

    valid_mask &= (t >= -epsilon) & (s >= -epsilon) & (s <= 1 + epsilon)

    intersection_points = p + t[..., np.newaxis] * d
    sq_distances = np.sum((intersection_points - p) ** 2, axis=2)
    sq_distances[~valid_mask] = np.inf

    min_indices = np.argmin(sq_distances, axis=1)
    min_distances = np.min(sq_distances, axis=1)

    results = []
    for i in range(n_rays):
        if min_distances[i] == np.inf:
            results.append(None)
        else:
            seg_idx = min_indices[i]
            results.append(Vector2D(*intersection_points[i, seg_idx]))

    return results

