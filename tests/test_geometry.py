import math
from typing import List, Optional, Tuple

import pytest
from entities.figures import Segment, Rectangle, Vector2D, Ray
from utils.geometry import is_segment_rectangle_intersection, ray_segment_intersection, segment_intersect, \
    is_batch_segments_rectangle_intersection, get_batch_ray_intersect_segments


@pytest.mark.parametrize("seg1, seg2, expected", [
    # Crossing segments
    (Segment(Vector2D(0, 0), Vector2D(2, 2)),
     Segment(Vector2D(0, 2), Vector2D(2, 0)),
     Vector2D(1, 1)),

    # Shared endpoint
    (Segment(Vector2D(0, 0), Vector2D(1, 1)),
     Segment(Vector2D(1, 1), Vector2D(2, 2)),
     Vector2D(1, 1)),

    # Collinear overlap
    (Segment(Vector2D(0, 0), Vector2D(4, 0)),
     Segment(Vector2D(1, 0), Vector2D(3, 0)),
     Vector2D(2, 0)),

    # No intersection
    (Segment(Vector2D(0, 0), Vector2D(1, 1)),
     Segment(Vector2D(2, 2), Vector2D(3, 3)),
     None)
])

def test_segment_intersect(seg1, seg2, expected):
    result = segment_intersect(seg1, seg2)
    if expected is None:
        assert result is None
    else:
        assert abs(result.x - expected.x) < 1e-9
        assert abs(result.y - expected.y) < 1e-9


@pytest.mark.parametrize(
    "rect_params, segment_coords, expected",
    [
        # Rectangle centered at (2,2), 2x2, rotated 45 degrees
        (
                (2, 2, 2, 2, 45),
                ((2, 2), (3, 3)),  # Diagonal inside
                True
        ),
        (
                (2, 2, 2, 2, 45),
                ((0, 2), (4, 2)),  # Horizontal through center
                True
        ),
        (
                (2, 2, 2, 2, 45),
                ((3, 1), (1, 3)),  # Crosses rotated rectangle
                True
        ),
        (
                (2, 2, 2, 2, 45),
                ((0, 0), (1, 1)),  # Outside diagonal
                False
        ),
        (
                (2, 2, 2, 2, 45),
                ((2.5, 2.5), (3.5, 3.5)),  # Endpoint inside
                True
        ),
        # Rectangle centered at (5,5), 4x2, rotated 30 degrees
        (
                (5, 5, 4, 2, 30),
                ((5, 5), (7, 6)),  # Along length direction
                True
        ),
        (
                (5, 5, 4, 2, 30),
                ((3, 4), (3, 6)),
                True
        ),
    ]
)
def test_rotated_intersections(rect_params, segment_coords, expected):
    cx, cy, x_len, y_len, angle_deg = rect_params
    rect = Rectangle(
        center=Vector2D(cx, cy),
        x_len=x_len,
        y_len=y_len,
        angle=math.radians(angle_deg)
    )

    start = Vector2D(segment_coords[0][0], segment_coords[0][1])
    end = Vector2D(segment_coords[1][0], segment_coords[1][1])
    segment = Segment(start, end)

    assert is_segment_rectangle_intersection(segment, rect) == expected


@pytest.mark.parametrize(
    "ray_start, ray_dir, segment_coords, expected_point",
    [
        # Луч пересекает отрезок внутри
        ((0, 0), (1, 1), ((2, 0), (0, 2)), (1.0, 1.0)),
        ((0, 0), (-1, -1), ((-2, 0), (0, -2)), (-1.0, -1.0)),
        # Луч начинается на отрезке
        ((1, 1), (2, 2), ((0, 0), (3, 3)), (1.0, 1.0)),
        # Луч направлен в противоположную сторону
        ((0, 0), (-1, -1), ((1, 1), (2, 2)), None),
        # Параллельные линии (нет пересечения)
        ((0, 0), (0, 1), ((1, 0), (1, 2)), None),
        # Луч совпадает с частью отрезка
        ((1, 1), (2, 2), ((0, 0), (3, 3)), (1.0, 1.0)),
        # Отрезок за пределами луча
        ((2, 2), (3, 3), ((0, 0), (1, 1)), None),
        # Касание конца отрезка
        ((0, 0), (2, 0), ((2, 0), (2, 2)), (2.0, 0.0)),
        # Вертикальный луч через середину отрезка
        ((2, 0), (0, 1), ((2, 2), (2, -2)), (2.0, 0.0)),
        # Граничный случай с epsilon (почти параллельны)
        ((0, 0), (1, 1e-7), ((2, 0), (2, 1)), (2.0, 2e-7)),
        # Точечный отрезок на луче
        ((1, 1), (2, 2), ((1, 1), (1, 1)), (1.0, 1.0)),
        # Точечный отрезок вне луча
        ((0, 0), (1, 0), ((2, 0), (2, 0)), None),
        ((2, 2), (-1, -1), ((0, 2), (2, 0)), (1.0, 1.0)),
        ((2, 2), (-1, -1), ((0, 5), (5, 0)), None),
    ]
)
def test_ray_segment_intersection(ray_start, ray_dir, segment_coords, expected_point):
    ray = Ray(
        start=Vector2D(*ray_start),
        direction=Vector2D(*ray_dir)
    )

    seg_start = Vector2D(*segment_coords[0])
    seg_end = Vector2D(*segment_coords[1])
    segment = Segment(seg_start, seg_end)

    result = ray_segment_intersection(ray, segment)

    if expected_point is None:
        assert result is None
    else:
        assert result is not None
        assert abs(result.x - expected_point[0]) < 1e-6
        assert abs(result.y - expected_point[1]) < 1e-6

def _compare_intersections(result: List[Optional[Vector2D]], expected: List[Optional[Tuple[float, float]]], tol=1e-6):
    assert len(result) == len(expected)
    for res_pt, exp_coords in zip(result, expected):
        if exp_coords is None:
            assert res_pt is None
        else:
            assert res_pt is not None
            assert isinstance(res_pt, Vector2D)
            assert abs(res_pt.x - exp_coords[0]) < tol
            assert abs(res_pt.y - exp_coords[1]) < tol

@pytest.mark.parametrize(
    "rays_params, segments_params, expected_intersections",
    [
        # No rays
        ([], [((0, 0), (1, 1))], []),
        # No segments
        ([((0, 0), (1, 0))], [], [None]),
        # Single ray, single segment (intersection)
        ([((0, 0), (1, 1))], [((2, 0), (0, 2))], [(1.0, 1.0)]),
        # Single ray, single segment (no intersection)
        ([((0, 0), (-1, -1))], [((1, 1), (2, 2))], [None]),
        # Multiple rays, multiple segments
        (
            [((0, 0), (1, 0)), ((0, 3), (1, -1)), ((-1, 0), (0, 1))], # Rays: horizontal, diagonal down, diagonal up
            [((2, -1), (2, 1)), ((1, 2), (3, 2))], # Segments: vertical at x=2, horizontal at y=2
            # Expected: Ray1 hits x=2 -> (2,0). Ray2 hits y=2 at (1,2) [dist sq=2] and x=2 at (2,1) [dist sq=8]. Closest is (1,2). Ray3 hits y=2 -> (1,2).
            [(2.0, 0.0), (1.0, 2.0), None] # <<< CORRECTED EXPECTED VALUE for Ray 2 and Ray 3
        ),
        # Ray intersects multiple segments (closest should be returned)
        (
            [((0, 0), (1, 0))], # Ray along positive x-axis
            [((2, -1), (2, 1)), ((4, -1), (4, 1))], # Two vertical segments
            [(2.0, 0.0)] # Closest intersection at x=2
        ),
        # Multiple rays, some intersect, some don't
        (
            [((0, 0), (1, 0)), ((5, 5), (0, 1)), ((0, 0), (-1, -1))],
            [((2, -1), (2, 1)), ((4, -1), (4, 1))],
            [(2.0, 0.0), None, None]
        ),
        # Ray starts on segment
        (
             [((1, 1), (1, 1))], # Ray starting at (1,1) going right-up
             [((3, 0), (0, 3))], # Segment passing through (1,5,1.5)
             [(1.5, 1.5)]
        ),
        # Parallel ray and segment
        (
            [((0, 0), (1, 0))],
            [((0, 1), (2, 1))],
            [None]
        ),
    ]
)
def test_get_batch_ray_intersect_segments(rays_params, segments_params, expected_intersections):
    rays = [Ray(Vector2D(*start), Vector2D(*direction)) for start, direction in rays_params]
    segments = [Segment(Vector2D(*start), Vector2D(*end)) for start, end in segments_params]

    result = get_batch_ray_intersect_segments(rays, segments)

    _compare_intersections(result, expected_intersections)


@pytest.mark.parametrize(
    "rect_params, segments_params, expected_result",
    [
        # No segments
        ((0, 0, 2, 2, 0), [], False),
        # Single segment intersecting (axis-aligned rect)
        ((0, 0, 2, 2, 0), [((-2, 0), (2, 0))], True),
        # Single segment not intersecting
        ((0, 0, 2, 2, 0), [((2, 2), (3, 3))], False),
        # Single segment endpoint inside
        ((0, 0, 2, 2, 0), [((0.5, 0.5), (3, 3))], True),
        # Single segment fully inside
        ((0, 0, 2, 2, 0), [((-0.5, -0.5), (0.5, 0.5))], True),
        # Multiple segments, none intersecting
        ((5, 5, 4, 2, 30), [((0, 0), (1, 1)), ((-1, -1), (-2, -2))], False),
        # Multiple segments, one intersecting (rotated rect)
        ((5, 5, 4, 2, 30), [((0, 0), (1, 1)), ((3, 4), (3, 6))], True),
        # Multiple segments, all intersecting
        ((5, 5, 4, 2, 30), [((5, 5), (7, 6)), ((3, 4), (3, 6))], True),
        # Multiple segments, one fully inside, others outside
        ((0, 0, 4, 4, 0), [((-1, -1), (1, 1)), ((5, 5), (6, 6)), ((-5, -5), (-6, -6))], True),
         # Edge case: Segment endpoint on boundary
        ((0, 0, 2, 2, 0), [((1, 1), (2, 2))], True),
         # Edge case: Segment collinear with side, partially overlapping
        ((0, 0, 2, 2, 0), [((0, 1), (2, 1))], True),
         # Edge case: Segment collinear with side, not overlapping
        ((0, 0, 2, 2, 0), [((2, 1), (3, 1))], False),
         # Rotated rectangle, complex case
        ( (2, 2, 2*math.sqrt(2), 2*math.sqrt(2), 45), # 4x4 square rotated 45 deg
          [((0,0), (1,0)), ((4,4), (5,5))], # One outside bottom-left, one outside top-right
          False
        ),
        ( (2, 2, 2*math.sqrt(2), 2*math.sqrt(2), 45),
          [((0,0), (1,0)), ((2,0), (2,4))], # One outside, one intersecting vertically
          True
        ),
    ]
)
def test_is_batch_segments_rectangle_intersection(rect_params, segments_params, expected_result):
    cx, cy, x_len, y_len, angle_deg = rect_params
    rectangle = Rectangle(
        center=Vector2D(cx, cy),
        x_len=x_len,
        y_len=y_len,
        angle=math.radians(angle_deg)
    )
    segments = [Segment(Vector2D(*start), Vector2D(*end)) for start, end in segments_params]

    result = is_batch_segments_rectangle_intersection(segments, rectangle)
    assert result == expected_result