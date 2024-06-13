import sys
from collections import defaultdict
from typing import Tuple, List, Set
import glfw
import glm

from model import Model


class Physics:
    EPSILON = 1e-6

    def __init__(self, world, player, interactables: list):
        self.world = world
        self.player = player
        self.gravity = glm.vec3(0, -10, 0)
        self.interactables = interactables

    def apply_gravity(self, delta_time: float):
        if not self.player.is_grounded:
            self.player.velocity += self.gravity * delta_time
            self.player.position += self.player.velocity * delta_time + 0.5 * self.gravity * delta_time ** 2
        #for item in self.interactables:
        #    item.velocity += self.gravity * delta_time
        #    item.position += item.velocity * delta_time + 0.5 * self.gravity * delta_time ** 2

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

    def is_point_in_aabb(self, point, aabb):
        (min_x, min_y, min_z), (max_x, max_y, max_z) = aabb
        return (min_x <= point.x <= max_x) and (min_y <= point.y <= max_y) and (min_z <= point.z <= max_z)

    def check_simple_collision(self):
        player_position = self.player.position

        for obj in self.world.get_objects():
            obj_bounding_box = obj.calculate_bounding_box()
            obj_aabb = self.get_aabb(obj_bounding_box)
            obj.world_aabb = obj_aabb

            if self.is_point_in_aabb(player_position, obj_aabb):
                return True

        return False

    def calculate_toi(self, pos, min_val, max_val, velocity):
        if velocity > 0:
            return (max_val - pos) / velocity
        elif velocity < 0:
            return (min_val - pos) / velocity
        return float('inf')

    def resolve_collision(self, obstacle_aabb):
        (min_x, min_y, min_z), (max_x, max_y, max_z) = obstacle_aabb
        player_pos = self.player.position
        player_velocity = self.player.velocity

        # Calculate overlap on each axis
        overlap_x = min(max_x - player_pos.x, player_pos.x - min_x)
        overlap_y = min(max_y - player_pos.y, player_pos.y - min_y)
        overlap_z = min(max_z - player_pos.z, player_pos.z - min_z)

        # Find the smallest overlap to determine the resolution axis
        min_overlap = min(overlap_x, overlap_y, overlap_z)

        if min_overlap == overlap_x:
            # Resolve along x-axis
            if player_pos.x > (max_x + min_x) / 2:
                # Player is to the right of the obstacle
                self.player.position.x += overlap_x
            else:
                # Player is to the left of the obstacle
                self.player.position.x -= overlap_x
            self.player.velocity.x = 0
        elif min_overlap == overlap_y:
            # Resolve along y-axis
            if player_pos.y > (max_y + min_y) / 2:
                # Player is above the obstacle
                if self.player.velocity.y < 0: # Player is falling
                    self.player.position.y += overlap_y
                    self.player.velocity.y = 0 # Stop player from falling
                    self.player.is_grounded = True
                else: # Player is already moving in the direction that would resolve the collision
                    pass#self.player.is_grounded = False
            else:
                # Player is below the obstacle
                self.player.position.y -= overlap_y
                self.player.velocity.y = 0
        else:
            # Resolve along z-axis
            if player_pos.z > (max_z + min_z) / 2:
                # Player is in front of the obstacle
                self.player.position.z += overlap_z
            else:
                # Player is behind the obstacle
                self.player.position.z -= overlap_z
            self.player.velocity.z = 0

    def handle_collisions(self, delta_time):
        for obj in self.world.objects:
            if self.check_linear_collision():
                #print('collision')
                self.resolve_collision(obj.aabb)
                break  # Stop checking further objects if a collision is detected

    def check_linear_collision(self):
        start_pos = self.player.previous_position
        end_pos = self.player.position
        for obj in self.world.get_objects():
            if self.is_line_segment_intersecting_aabb(start_pos, end_pos, obj.aabb):
                return True
        return False

    def is_line_segment_intersecting_aabb(self, start, end, aabb):
        (min_x, min_y, min_z), (max_x, max_y, max_z) = aabb

        def slab_check(p0, p1, min_b, max_b):
            inv_d = 1.0 / (p1 - p0) if (p1 - p0) != 0 else float('inf')
            t0 = (min_b - p0) * inv_d
            t1 = (max_b - p0) * inv_d
            if t0 > t1:
                t0, t1 = t1, t0
            return t0, t1

        tmin, tmax = 0.0, 1.0

        for axis in range(3):
            p0 = start[axis]
            p1 = end[axis]
            min_b = [min_x, min_y, min_z][axis]
            max_b = [max_x, max_y, max_z][axis]

            if p0 == p1:
                if p0 < min_b or p0 > max_b:
                    return False
            else:
                t0, t1 = slab_check(p0, p1, min_b, max_b)
                tmin = max(tmin, t0)
                tmax = min(tmax, t1)
                if tmin > tmax:
                    return False

        return True

    def update(self, delta_time: float):
        #if not self.check_linear_collision():  # self.check_simple_collision():
        self.apply_gravity(delta_time)
        self.handle_collisions(delta_time=delta_time)
