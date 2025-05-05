from entities.figures import Vector2D


class CarState:
    def __init__(self, velocity: Vector2D, angle: float, ):
        self.velocity = velocity
        self.angle = angle