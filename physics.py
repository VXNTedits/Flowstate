from collections import defaultdict
from typing import Tuple, List, Set
import glfw
import glm

import shape
from aabb import AABB
from model import Model


class Physics:
    def __init__(self, world, player):
        self.world = world
        self.player = player
        self.gravity = glm.vec3(0, -9.81, 0)

    def apply_gravity(self, player, delta_time: float):
        player.velocity += self.gravity * delta_time
        player.position += player.velocity * delta_time + 0.5 * self.gravity * delta_time ** 2

    @staticmethod
    def project_shape(model: Model, axis: glm.vec3) -> Tuple[float, float]:
        min_proj = float('inf')
        max_proj = float('-inf')
        vertices = model.vertices.reshape(-1, 3)
        for vertex in vertices:
            proj = glm.dot(glm.vec3(vertex), axis)
            min_proj = min(min_proj, proj)
            max_proj = max(max_proj, proj)
        return min_proj, max_proj

    @staticmethod
    def overlap(proj1: Tuple[float, float], proj2: Tuple[float, float]) -> bool:
        return not (proj1[1] < proj2[0] or proj2[1] < proj1[0])

    @staticmethod
    def sat_collision(model1: Model, model2: Model) -> bool:
        axes1 = model1.get_normals(model1.compute_edges(model1.vertices.reshape(-1, 3)))
        axes2 = model2.get_normals(model2.compute_edges(model2.vertices.reshape(-1, 3)))
        axes = axes1 + axes2
        for axis in axes:
            proj1 = Physics.project_shape(model1, axis)
            proj2 = Physics.project_shape(model2, axis)
            if not Physics.overlap(proj1, proj2):
                return False
        return True

    def check_collision(self, player: Model, world: Model) -> bool:
        for shape1 in player.convex_components:
            for shape2 in world.convex_components:
                if Physics.sat_collision(shape1, shape2):
                    return True
        return False

    def update(self, delta_time: float):
        self.apply_gravity(self.player, delta_time)
        self.check_collision(self.player, self.world)
