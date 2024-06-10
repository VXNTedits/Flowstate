from collections import defaultdict
from typing import Tuple, List, Set
import glfw
import glm



class Physics:
    def __init__(self, world, player):
        self.world = world
        self.player = player
        self.gravity = glm.vec3(0, -9.81, 0)

    def apply_gravity(self, player, delta_time: float):
        player.velocity += self.gravity * delta_time
        player.position += player.velocity * delta_time + 0.5 * self.gravity * delta_time ** 2

    def update(self, delta_time: float):
        self.apply_gravity(self.player, delta_time)
        self.broad_phase_check()
        pass

    def check_collisions(self, player, world, objects=None):
        #TODO
        pass

    def broad_phase_check(self):
        return self.world.aabb.overlaps(self.player.aabb)