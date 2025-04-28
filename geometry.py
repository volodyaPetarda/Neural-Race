import math
from typing import List

from figures import Segment, Rectangle, Vector2D, ccw, on_segment, Ray, dot


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

def is_segment_rectangle_intersection(segment: Segment, rectangle: Rectangle) -> bool:
    def is_point_inside(p: Vector2D) -> bool:
        translated = Vector2D(p.x - rectangle.center.x, p.y - rectangle.center.y)
        cos_theta = math.cos(rectangle.angle)
        sin_theta = math.sin(rectangle.angle)
        local_x = translated.x * cos_theta + translated.y * sin_theta
        local_y = -translated.x * sin_theta + translated.y * cos_theta
        return (abs(local_x) <= rectangle.x_len/2 and
                abs(local_y) <= rectangle.y_len/2)

    if is_point_inside(segment.start) or is_point_inside(segment.end):
        return True

    for side in rectangle.get_sides():
        if segment_intersect(segment, side):
            return True

    return False


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

def get_batch_ray_intersect_segments(rays: List[Ray], segments: List[Segment]) -> list[Vector2D | None]:
    result = []
    for ray in rays:
        min_len = float('inf')
        point = None
        for segment in segments:
            intersection = ray_segment_intersection(ray, segment, epsilon=1e-6)
            if intersection is None:
                continue
            cur_len = (intersection - ray.start).square_magnitude()
            if cur_len < min_len:
                min_len = cur_len
                point = intersection
        result.append(point)
    return result