from collections import defaultdict
from typing import Tuple, List, Set
import glfw
import glm
import numpy as np

from model import Model

class Physics:
    def __init__(self, world, player):
        self.world = world
        self.player = player
        self.gravity = glm.vec3(0, -9.81, 0)

    def apply_gravity(self, player, delta_time: float):
        player.velocity += self.gravity * delta_time
        player.position += player.velocity * delta_time + 0.5 * self.gravity * delta_time ** 2

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
        #print(f"Checking voxel collision for player position: {player_position}")
        for voxel in self.world.voxels:
            if glm.distance(player_position, voxel) < world.voxel_size / 2:
                print(f"Collision detected with voxel at position: {voxel}")
                return True
        return False

    def update(self, delta_time: float):
        self.apply_gravity(self.player, delta_time)
        if self.is_player_below_world(self.player.position):
            print("Player below world")
            #resolve_collision()
        #if self.check_collision(self.player, self.world):
        #    print("Collision detected!")
        if self.check_voxel_collision(self.player, self.world):
            print("Voxel collision detected!")
        if self.check_simple_collision(self.player,self.world):
            print("Simple collision detected!")

    def check_simple_collision(self, player, world):
        collision = False
        box_player = player.bounding_box
        box_world = world.bounding_box

        # Convert player's bounding box to world coordinates
        player_vertices = []
        for mat_set in box_player:
            for mat in mat_set:
                # Extract the translation part of the matrix (assuming the matrix class has a method or property for this)
                translation = [mat[3][0], mat[3][1], mat[3][2]]
                player_vertices.append(translation)

        # Extract the bounding box vertices for the world
        world_vertices = []
        for vec in box_world:
            world_vertices.append([vec[0], vec[1], vec[2]])

        # Find the axis-aligned bounding boxes (AABB) for both player and world
        def get_aabb(vertices):
            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')

            for v in vertices:
                min_x = min(min_x, v[0])
                min_y = min(min_y, v[1])
                min_z = min(min_z, v[2])
                max_x = max(max_x, v[0])
                max_y = max(max_y, v[1])
                max_z = max(max_z, v[2])

            return (min_x, min_y, min_z, max_x, max_y, max_z)

        player_aabb = get_aabb(player_vertices)
        world_aabb = get_aabb(world_vertices)

        # Check for overlap in all three axes
        def aabb_overlap(aabb1, aabb2):
            return (aabb1[0] <= aabb2[3] and aabb1[3] >= aabb2[0] and
                    aabb1[1] <= aabb2[4] and aabb1[4] >= aabb2[1] and
                    aabb1[2] <= aabb2[5] and aabb1[5] >= aabb2[2])

        collision = aabb_overlap(player_aabb, world_aabb)

        return collision


# Assuming this is part of the Model class
def decompose_to_voxels(self, vertices: np.ndarray, voxel_size: float) -> List[glm.vec3]:
    # Convert vertices from ndarray to list[glm.vec3]
    vertices = vertices.reshape(-1, 3)
    vertices_list = [glm.vec3(vertex[0], vertex[1], vertex[2]) for vertex in vertices]

    # Calculate the bounding box from the given vertices
    min_corner = glm.vec3(
        min(vertex.x for vertex in vertices_list),
        min(vertex.y for vertex in vertices_list),
        min(vertex.z for vertex in vertices_list)
    )
    max_corner = glm.vec3(
        max(vertex.x for vertex in vertices_list),
        max(vertex.y for vertex in vertices_list),
        max(vertex.z for vertex in vertices_list)
    )

    # Calculate the number of voxels along each axis
    voxel_count_x = int((max_corner.x - min_corner.x) / voxel_size) + 1
    voxel_count_y = int((max_corner.y - min_corner.y) / voxel_size) + 1
    voxel_count_z = int((max_corner.z - min_corner.z) / voxel_size) + 1

    voxels = []
    for x in range(voxel_count_x):
        for y in range(voxel_count_y):
            for z in range(voxel_count_z):
                voxel_center = glm.vec3(
                    min_corner.x + x * voxel_size + voxel_size / 2,
                    min_corner.y + y * voxel_size + voxel_size / 2,
                    min_corner.z + z * voxel_size + voxel_size / 2
                )
                if self.is_point_inside_shape(voxel_center, vertices_list):
                    voxels.append(voxel_center)
    return voxels
