from collections import defaultdict
from typing import Tuple, List, Set
import glfw
import glm
import sys
from camera import Camera
from shader import Shader
from world import World
from player import Player
from renderer import Renderer
from input_handler import InputHandler
from model import Model
from text_renderer import TextRenderer
from interfaces import WorldInterface, PlayerInterface


class Physics:
    def __init__(self, world: WorldInterface, player: PlayerInterface):
        assert isinstance(world, WorldInterface), "World must implement WorldInterface."
        assert isinstance(player, PlayerInterface), "Player must implement PlayerInterface."
        self.world = world
        self.player = player
        self.colliders = []
        self.grid_size = 1.0  # Size of each grid cell
        self.grid = defaultdict(list)  # Spatial grid for partitioning
        self.add_model_collider(world)
        self.add_model_collider(player.model)
        self.gravity = glm.vec3(0, -9.81, 0)

    def add_collider(self, collider: Tuple[glm.vec3, glm.vec3, glm.vec3]):
        assert isinstance(collider, tuple) and len(collider) == 3, "Collider must be a tuple of three vertices."
        min_point, max_point = self.get_surface_aabb(collider)
        self.colliders.append((collider, min_point, max_point))
        self.add_to_grid(collider, min_point, max_point)

    def add_model_collider(self, model: WorldInterface):
        assert isinstance(model, WorldInterface), "Model must implement WorldInterface."
        surfaces = model.get_surfaces()
        assert all(isinstance(surface, tuple) and len(surface) == 3 for surface in surfaces), "Surfaces must be tuples of three vertices."
        for surface in surfaces:
            self.add_collider(surface)

    def add_to_grid(self, collider: Tuple[glm.vec3, glm.vec3, glm.vec3], min_point: glm.vec3, max_point: glm.vec3):
        assert isinstance(min_point, glm.vec3) and isinstance(max_point, glm.vec3), "Min and max points must be glm.vec3."
        min_cell = self.world_to_grid(min_point)
        max_cell = self.world_to_grid(max_point)
        for x in range(min_cell.x, max_cell.x + 1):
            for y in range(min_cell.y, max_cell.y + 1):
                for z in range(min_cell.z, max_cell.z + 1):
                    self.grid[(x, y, z)].append(collider)

    def world_to_grid(self, point: glm.vec3) -> glm.ivec3:
        assert isinstance(point, glm.vec3), "Point must be glm.vec3."
        return glm.ivec3(
            int(point.x // self.grid_size),
            int(point.y // self.grid_size),
            int(point.z // self.grid_size)
        )

    def apply_gravity(self, player: PlayerInterface, delta_time: float):
        assert isinstance(delta_time, float), "Delta time must be a float."
        assert hasattr(player, 'velocity') and hasattr(player, 'position'), "Player must have velocity and position attributes."
        assert isinstance(player.velocity, glm.vec3) and isinstance(player.position, glm.vec3), "Player's velocity and position must be glm.vec3."
        player.velocity += self.gravity * delta_time
        player.position += player.velocity * delta_time + 0.5 * self.gravity * delta_time ** 2

    def check_collision(self, model1: WorldInterface, model2: WorldInterface) -> bool:
        assert hasattr(model1, 'get_surfaces') and hasattr(model2, 'get_surfaces'), "Model objects must have get_surfaces methods."
        colliders1 = self.get_model_colliders(model1)
        colliders2 = self.get_model_colliders(model2)
        for surface1 in colliders1:
            for surface2 in colliders2:
                if self.check_surface_collision(surface1, surface2):
                    return True
        return False

    def get_model_colliders(self, model: WorldInterface) -> Set:
        assert hasattr(model, 'get_surfaces'), "Model object must have a get_surfaces method."
        colliders = set()
        for surface in model.get_surfaces():
            assert isinstance(surface, tuple) and len(surface) == 3, "Surface must be a tuple of three vertices."
            min_point, max_point = self.get_surface_aabb(surface)
            min_cell = self.world_to_grid(min_point)
            max_cell = self.world_to_grid(max_point)
            for x in range(min_cell.x, max_cell.x + 1):
                for y in range(min_cell.y, max_cell.y + 1):
                    for z in range(min_cell.z, max_cell.z + 1):
                        colliders.update(self.grid[(x, y, z)])
        return colliders

    def check_surface_collision(self, surface1: Tuple[glm.vec3, glm.vec3, glm.vec3], surface2: Tuple[glm.vec3, glm.vec3, glm.vec3]) -> bool:
        assert isinstance(surface1, tuple) and len(surface1) == 3, "Surface1 must be a tuple of three vertices."
        assert isinstance(surface2, tuple) and len(surface2) == 3, "Surface2 must be a tuple of three vertices."
        box1_min, box1_max = self.get_surface_aabb(surface1)
        box2_min, box2_max = self.get_surface_aabb(surface2)
        return self.check_aabb_collision(box1_min, box1_max, box2_min, box2_max)

    def get_surface_aabb(self, surface: Tuple[glm.vec3, glm.vec3, glm.vec3]) -> Tuple[glm.vec3, glm.vec3]:
        assert isinstance(surface, tuple) and len(surface) == 3, "Surface must be a tuple of three vertices."
        min_point = glm.vec3(float('inf'), float('inf'), float('inf'))
        max_point = glm.vec3(float('-inf'), float('-inf'), float('-inf'))

        for vertex in surface:
            assert isinstance(vertex, glm.vec3), "Vertex must be glm.vec3."
            min_point.x = min(min_point.x, vertex.x)
            min_point.y = min(min_point.y, vertex.y)
            min_point.z = min(min_point.z, vertex.z)
            max_point.x = max(max_point.x, vertex.x)
            max_point.y = max(max_point.y, vertex.y)
            max_point.z = max(max_point.z, vertex.z)

        return min_point, max_point

    def check_aabb_collision(self, box1_min: glm.vec3, box1_max: glm.vec3, box2_min: glm.vec3, box2_max: glm.vec3) -> bool:
        assert isinstance(box1_min, glm.vec3) and isinstance(box1_max, glm.vec3), "Box1 min and max must be glm.vec3."
        assert isinstance(box2_min, glm.vec3) and isinstance(box2_max, glm.vec3), "Box2 min and max must be glm.vec3."
        return (box1_min.x <= box2_max.x and box1_max.x >= box2_min.x) and \
               (box1_min.y <= box2_max.y and box1_max.y >= box2_min.y) and \
               (box1_min.z <= box2_max.z and box1_max.z >= box2_min.z)

    def resolve_collision(self, player: PlayerInterface, collider: Tuple[glm.vec3, glm.vec3, glm.vec3]):
        assert hasattr(player, 'velocity') and hasattr(player, 'position'), "Player must have velocity and position attributes."
        assert isinstance(player.velocity, glm.vec3) and isinstance(player.position, glm.vec3), "Player's velocity and position must be glm.vec3."
        player.position -= player.velocity  # Revert to previous position
        player.velocity = glm.vec3(0, 0, 0)  # Stop the player
