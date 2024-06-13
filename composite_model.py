import glm
from OpenGL.GL import *

class CompositeModel:
    def __init__(self):
        self.models = []
        self.model_matrices = []

    def add_model(self, model, relative_position=glm.vec3(0.0, 0.0, 0.0), relative_rotation=glm.vec3(0.0, 0.0, 0.0)):
        self.models.append((model, relative_position, relative_rotation))
        self.update_model_matrix()  # Update model matrices whenever a new sub-model is added


    def update_model_matrix(self, parent_matrix=glm.mat4(1.0)):
        self.model_matrices.clear()
        for model, rel_pos, rel_rot in self.models:
            # Create the translation matrix for the relative position
            translation_matrix = glm.translate(glm.mat4(1.0), rel_pos)

            # Create rotation matrices for the relative rotation
            rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(rel_rot.x), glm.vec3(1.0, 0.0, 0.0))
            rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rel_rot.y), glm.vec3(0.0, 1.0, 0.0))
            rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rel_rot.z), glm.vec3(0.0, 0.0, 1.0))

            # Combine the translation and rotation matrices
            model_matrix = parent_matrix * translation_matrix * rotation_matrix

            # Store the calculated model matrix
            self.model_matrices.append(model_matrix)

            # Update the sub-model's model matrix
            model.model_matrix = model_matrix


    def draw(self):
        for model, _, _ in self.models:
            model.draw()
