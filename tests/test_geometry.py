import math

import pytest
from figures import Segment, Rectangle, Vector2D, Ray
from geometry import is_segment_rectangle_intersection, ray_segment_intersection, segment_intersect


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