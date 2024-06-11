from collections import defaultdict
from typing import Tuple, List, Set
import glfw
import glm
import numpy as np

import player
import world
from model import Model

class Physics:
    EPSILON = 1e-1

    def __init__(self, world, player):
        self.world = world
        self.player = player
        self.gravity = glm.vec3(0, -10, 0)

    def apply_gravity(self, delta_time: float):
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

    def check_simple_collision(self):
        player_position = self.player.position

        for obj in self.world.get_objects():
            obj_bounding_box = obj.calculate_bounding_box()
            obj_aabb = self.get_aabb(obj_bounding_box)
            obj.world_aabb = obj_aabb

            if self.is_point_in_aabb(player_position, obj_aabb):
                return True

        return False

    def resolve_collision(self, world, player):
        player_position = player.position

        for obj in world.objects:
            obj_bounding_box = obj.calculate_bounding_box()
            obj_aabb = self.get_aabb(obj_bounding_box)
            obj.world_aabb = obj_aabb

            if self.is_point_in_aabb(player_position, obj_aabb):
                # Y-axis collision resolution (gravity axis) - skip if jumping
                if not self.player.is_jumping:
                    if player_position.y <= obj_aabb[1] + self.EPSILON:
                        player.position.y = obj_aabb[1]
                        self.player.velocity.y = 0  # Reset vertical velocity to stop gravity from pulling through
                        self.player.is_grounded = True  # Player is on the ground
                    elif player_position.y >= obj_aabb[4] - self.EPSILON:
                        player.position.y = obj_aabb[4]
                        self.player.velocity.y = 0  # Reset vertical velocity to stop gravity from pulling through

                # X-axis collision resolution
                if player_position.x <= obj_aabb[0] + self.EPSILON:
                    player.position.x = obj_aabb[0]
                elif player_position.x >= obj_aabb[3] - self.EPSILON:
                    player.position.x = obj_aabb[3]

                # Z-axis collision resolution
                if player_position.z <= obj_aabb[2] + self.EPSILON:
                    player.position.z = obj_aabb[2]
                elif player_position.z >= obj_aabb[5] - self.EPSILON:
                    player.position.z = obj_aabb[5]

                # Update player position
                player.position = player_position

    def handle_collisions(self, player, world, delta_time):
        collision_detected = False

        for obj in world.objects:
            if self.check_voxel_collision(player, obj):
                print("Voxel collision detected!")
                collision_detected = True
            if self.check_simple_collision():
                self.resolve_collision(world, player)
                collision_detected = True

        if collision_detected:
            self.resolve_collision(world, player)

    def update(self, delta_time: float):
        if not self.check_simple_collision():
            self.apply_gravity(delta_time)
        self.handle_collisions(self.player, self.world, delta_time)
