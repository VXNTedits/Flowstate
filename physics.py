from collections import defaultdict
from typing import Tuple, List, Set
import glfw
import glm
from model import Model

class Physics:
    def __init__(self, world, player):
        self.world = world
        self.player = player
        self.gravity = glm.vec3(0, -9.81, 0)

    def apply_gravity(self, player, delta_time: float):
        player.velocity += self.gravity * delta_time
        player.position += player.velocity * delta_time + 0.5 * self.gravity * delta_time ** 2

    def is_point_inside_convex_polyhedron(self, point: glm.vec3, convex_shape: Model) -> bool:
        vertices = convex_shape.vertices.reshape(-1, 3)
        indices = convex_shape.indices.reshape(-1, 3)
        for i in range(len(indices)):
            p1 = glm.vec3(vertices[indices[i][0]])
            p2 = glm.vec3(vertices[indices[i][1]])
            p3 = glm.vec3(vertices[indices[i][2]])
            normal = glm.normalize(glm.cross(p2 - p1, p3 - p1))
            if glm.dot(normal, point - p1) > 0:
                print(f"Point {point} is outside the triangle formed by {p1}, {p2}, {p3}")
                return False
        return True

    def check_collision(self, player: Model, world: Model) -> bool:
        player_position = self.player.position
        print(f"Checking collision for player position: {player_position}")
        for convex_shape in self.world.convex_components:
            if self.is_point_inside_convex_polyhedron(player_position, convex_shape):
                print(f"Collision detected with shape: {convex_shape.vertices}")
                return True
        return False

    def update(self, delta_time: float):
        self.apply_gravity(self.player, delta_time)
        if self.check_collision(self.player, self.world):
            print("Collision detected!")
