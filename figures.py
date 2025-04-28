import copy
import math
from typing import List


class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def one_rotated(angle):
        return Vector2D(math.cos(angle), math.sin(angle))

    def perpendicular(self):
        return Vector2D(-self.y, self.x)

    def normalized(self):
        vector_len = self.magnitude()
        if vector_len != 0:
            return Vector2D(self.x / vector_len, self.y / vector_len)
        else:
            return Vector2D(0, 0)

    def magnitude(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def square_magnitude(self):
        return self.x * self.x + self.y * self.y

    def rotate(self, angle):
        return Vector2D(self.x * math.cos(angle) - self.y * math.sin(angle), self.x * math.sin(angle) + self.y * math.cos(angle))

    def __add__(self, other) -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other) -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, other) -> 'Vector2D':
        return Vector2D(self.x * other, self.y * other)

class Segment:
    def __init__(self, start: Vector2D, end: Vector2D):
        self._start = copy.deepcopy(start)
        self._end = copy.deepcopy(end)

    @property
    def x1(self):
        return self.start.x
    @property
    def y1(self):
        return self.start.y
    @property
    def x2(self):
        return self.end.x
    @property
    def y2(self):
        return self.end.y

    @property
    def start(self):
        return copy.deepcopy(self._start)
    @property
    def end(self):
        return copy.deepcopy(self._end)

class Rectangle:
    def __init__(self, center: Vector2D, x_len: float, y_len: float, angle: float):
        self.center = center
        self.x_len = x_len
        self.y_len = y_len
        self.angle = angle

    def get_sides(self) -> List[Segment]:
        hx = self.x_len / 2
        hy = self.y_len / 2
        cos_theta = math.cos(self.angle)
        sin_theta = math.sin(self.angle)
        cx = self.center.x
        cy = self.center.y

        corners = []
        for dx, dy in [(-hx, -hy), (-hx, hy), (hx, hy), (hx, -hy)]:
            rx = dx * cos_theta - dy * sin_theta
            ry = dx * sin_theta + dy * cos_theta
            corners.append(Vector2D(rx + cx, ry + cy))

        return [
            Segment(corners[0], corners[1]),
            Segment(corners[1], corners[2]),
            Segment(corners[2], corners[3]),
            Segment(corners[3], corners[0])
        ]

class Ray:
    def __init__(self, start: Vector2D, direction: Vector2D):
        self._start = copy.deepcopy(start)
        self._direction = copy.deepcopy(direction.normalized())

    @property
    def direction(self) -> Vector2D:
        return copy.deepcopy(self._direction)
    @property
    def start(self):
        return copy.deepcopy(self._start)

def dot(vector1: Vector2D, vector2: Vector2D) -> float:
    return vector1.x * vector2.x + vector1.y * vector2.y

def ccw(p1: Vector2D, p2: Vector2D, p3: Vector2D) -> float:
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)

def on_segment(p1: Vector2D, p2: Vector2D, p3: Vector2D, epsilon=1e-9) -> bool:
    if abs(ccw(p1, p2, p3)) > epsilon:
        return False
    return (
        (min(p1.x, p2.x) - epsilon <= p3.x <= max(p1.x, p2.x) + epsilon) and
        (min(p1.y, p2.y) - epsilon <= p3.y <= max(p1.y, p2.y) + epsilon)
    )

def get_batch_rays(origin: Vector2D, angle: float, num_rays: int) -> List[Ray]:
    return [Ray(origin, Vector2D.one_rotated(angle + 2 * math.pi * (i / num_rays))) for i in range(num_rays)]