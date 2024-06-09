from typing import List, Tuple
import glm
from interfaces import WorldInterface
from model import Model

class World(WorldInterface, Model):
    def __init__(self, filepath: str, rotation_angles=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0)):
        super().__init__(filepath)
        self.set_orientation(rotation_angles)
        self.set_position(translation)

    def set_orientation(self, rotation_angles):
        # Apply rotations around x, y, z axes respectively
        rotation_matrix = glm.mat4(1.0)
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[0]), glm.vec3(1.0, 0.0, 0.0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[1]), glm.vec3(0.0, 1.0, 0.0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[2]), glm.vec3(0.0, 0.0, 1.0))
        self.model_matrix = rotation_matrix * self.model_matrix

    def set_position(self, translation):
        # Apply translation
        translation_matrix = glm.translate(glm.mat4(1.0), glm.vec3(translation[0], translation[1], translation[2]))
        self.model_matrix = translation_matrix * self.model_matrix

    def get_surfaces(self) -> List[Tuple[glm.vec3, glm.vec3, glm.vec3]]:
        surfaces = super().get_surfaces()
        print(f"World.get_surfaces: {surfaces}")
        assert surfaces is not None, "get_surfaces should not return None"
        assert all(isinstance(surface, tuple) and len(surface) == 3 for surface in surfaces), "Surfaces must be tuples of three vertices"
        return surfaces
