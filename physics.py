from collections import defaultdict
from typing import Tuple, List, Set
import glfw
import glm
import numpy as np

import player
import world
from model import Model

class Physics:
    EPSILON = 1e-6

    def __init__(self, world, player):
        self.world = world
        self.player = player
        self.gravity = glm.vec3(0, -9.81, 0)

    def apply_gravity(self, player, delta_time: float):
        self.player.velocity += self.gravity * delta_time
        self.player.position += self.player.velocity * delta_time + 0.5 * self.gravity * delta_time ** 2

    def is_player_below_world(self, player: glm.vec3):
        if player.y <= self.world.height:
            return True

    def is_point_inside_convex_polyhedron(self, point: glm.vec3, convex_shape: List[glm.vec3]) -> bool:
        indices = range(0, len(convex_shape), 3)  # Assuming the convex_shape is a flattened list of triangles
        for i in indices:
            p1 = convex_shape[i]
            p2 = convex_shape[i + 1]
            p3 = convex_shape[i + 2]
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
                print(f"Collision detected with shape: {convex_shape}")
                return True
        return False

    def check_voxel_collision(self, player: Model, world: Model) -> bool:
        player_position = self.player.position
        for voxel in self.world.voxels:
            if glm.distance(player_position, voxel) < world.voxel_size / 2:
                print(f"Collision detected with voxel at position: {voxel}")
                return True
        return False

    def get_aabb(self, vertices):
        min_x = min(vertices, key=lambda v: v.x).x
        min_y = min(vertices, key=lambda v: v.y).y
        min_z = min(vertices, key=lambda v: v.z).z
        max_x = max(vertices, key=lambda v: v.x).x
        max_y = max(vertices, key=lambda v: v.y).y
        max_z = max(vertices, key=lambda v: v.z).z

        return (min_x, min_y, min_z, max_x, max_y, max_z)

    def is_point_in_aabb(self, point, aabb):
        return (aabb[0] <= point.x <= aabb[3] and
                aabb[1] <= point.y <= aabb[4] and
                aabb[2] <= point.z <= aabb[5])

    def check_simple_collision(self, player, world):
        player_position = player.position
        world_bounding_box = world.calculate_bounding_box()
        world_aabb = self.get_aabb(world_bounding_box)
        world.world_aabb = world_aabb

        return self.is_point_in_aabb(player_position, world_aabb)

    def resolve_collision(self, world_in_collision, player_in_collision):
        if not self.check_simple_collision(player_in_collision, world_in_collision):
            return

        player_position = player_in_collision.position
        world_aabb = world_in_collision.world_aabb
        print(f"Player position: {player_position}, World AABB: {world_aabb}")

        # Y-axis collision resolution (gravity axis)
        if player_position.y <= world_aabb[1] + self.EPSILON:
            #print(f"player_position.y < world_aabb[1]: {player_position.y} < {world_aabb[1]}")
            player_in_collision.position.y = world_aabb[1]
            self.player.velocity.y = 0  # Reset vertical velocity to stop gravity from pulling through
        elif player_position.y >= world_aabb[4] - self.EPSILON:
            #print(f"player_position.y > world_aabb[4]: {player_position.y} > {world_aabb[4]}")
            player_in_collision.position.y = world_aabb[4]
            self.player.velocity.y = 0  # Reset vertical velocity to stop gravity from pulling through

        # X-axis collision resolution
        if player_position.x <= world_aabb[0] + self.EPSILON:
            #print(f"player_position.x < world_aabb[0]: {player_position.x} < {world_aabb[0]}")
            player_in_collision.position.x = world_aabb[0]
        elif player_position.x >= world_aabb[3] - self.EPSILON:
            #print(f"player_position.x > world_aabb[3]: {player_position.x} > {world_aabb[3]}")
            player_in_collision.position.x = world_aabb[3]

        # Z-axis collision resolution
        if player_position.z <= world_aabb[2] + self.EPSILON:
            #print(f"player_position.z < world_aabb[2]: {player_position.z} < {world_aabb[2]}")
            player_in_collision.position.z = world_aabb[2]
        elif player_position.z >= world_aabb[5] - self.EPSILON:
            #print(f"player_position.z > world_aabb[5]: {player_position.z} > {world_aabb[5]}")
            player_in_collision.position.z = world_aabb[5]

        # Update player position
        player.position = player_in_collision.position
        #print(f"Player position corrected to: {player_in_collision.position}", end="\r")

    def handle_collisions(self, player, world, delta_time):
        if self.check_voxel_collision(player, world):
            print("Voxel collision detected!")
        if self.check_simple_collision(player, world):
            self.resolve_collision(world, player)
            #print("Simple collision detected!")

    def update(self, delta_time: float):
        if not self.check_simple_collision(self.player, self.world):
            self.apply_gravity(self.player, delta_time)
        self.handle_collisions(self.player, self.world, delta_time)