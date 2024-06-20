import glm
import numpy as np
from OpenGL.GL import *

from model import Model


class CompositeModel(Model):
    def __init__(self,
                 filepath,
                 mtl_filepath,
                 player=False,
                 draw_convex_only=False,
                 rotation_angles=glm.vec3(0, 0, 0),
                 translation=glm.vec3(0, 0, 0),
                 kd_override=None,
                 ks_override=None,
                 ns_override=None,
                 scale=1,
                 is_collidable=False,
                 shift_to_centroid=True
                 ):
        self.name = filepath.split('/')[-1].split('.')[0]
        self.composite_position = translation
        self.composite_rotation = rotation_angles
        self.shift_to_centroid = shift_to_centroid
        self.is_collidable = is_collidable
        self.scale = scale
        self.ns_override = ns_override
        self.ks_override = ks_override
        self.kd_override = kd_override
        self.draw_convex_only = draw_convex_only
        self.player = player
        self.models = []
        self.model_matrices = []
        super().__init__(
                         filepath,
                         mtl_filepath,
                         draw_convex_only=draw_convex_only,
                         rotation_angles=rotation_angles,
                         translation=translation,
                         kd_override=kd_override,
                         ks_override=ks_override,
                         ns_override=ns_override,
                         scale=scale,
                         is_collidable=is_collidable,
                         shift_to_centroid=shift_to_centroid
                         )

        self.models.append((self, self.composite_position, self.composite_rotation))
        self.composite_centroid = self.calculate_composite_centroid()

    def add_world_model(self, model, scale, relative_position=glm.vec3(0.0, 0.0, 0.0),
                        relative_rotation=glm.vec3(0.0, 0.0, 0.0)):
        self.models.append((model, relative_position, relative_rotation))
        model.set_scale(scale)
        model.set_position(relative_position)
        model.set_orientation(relative_rotation)
        model.init_model_matrix(relative_position, relative_rotation)
        self.update_flat_model_matrix()  # Ensure initial update without explicitly passing identity matrix

    def get_objects(self):
        # Return all models contained in this composite model
        return [model for model, _, _ in self.models]

    def update_composite_model_matrix(self, parent_matrix=None):
        self.model_matrices.clear()

        # Ensure parent_matrix is a glm.mat4
        if parent_matrix is None:
            parent_matrix = glm.mat4(1.0)
        elif isinstance(parent_matrix, glm.vec4):
            raise ValueError("parent_matrix should be a glm.mat4, not a glm.vec4")

        #print(f"Initial parent matrix:\n{parent_matrix}")

        # Check if there are models to process
        if not self.models:
            return

        # Process the root model (first model in the list)
        root_model, root_rel_pos, root_rel_rot = self.models[0]

        # Create translation matrix for the root model
        root_translation_matrix = glm.translate(glm.mat4(1.0), root_rel_pos)
        #print(f"Root translation matrix:\n{root_translation_matrix}")

        # Create rotation matrices for the root model and combine them
        root_rotation_x = glm.rotate(glm.mat4(1.0), glm.radians(root_rel_rot.x), glm.vec3(1.0, 0.0, 0.0))
        root_rotation_y = glm.rotate(root_rotation_x, glm.radians(root_rel_rot.y), glm.vec3(0.0, 1.0, 0.0))
        root_rotation_z = glm.rotate(root_rotation_y, glm.radians(root_rel_rot.z), glm.vec3(0.0, 0.0, 1.0))
        root_rotation_matrix = root_rotation_z
        #print(f"Root rotation matrix:\n{root_rotation_matrix}")

        # Create scale matrix for the root model
        root_scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(root_model.scale, root_model.scale, root_model.scale))
        #print(f"Root scale matrix:\n{root_scale_matrix}")

        # Combine parent matrix with root translation, rotation, and scale matrices
        root_model_matrix = parent_matrix * root_translation_matrix * root_rotation_matrix * root_scale_matrix
        #print(f"Combined root model matrix:\n{root_model_matrix}")

        # Store the computed root model matrix
        self.model_matrices.append(root_model_matrix)

        # Update the root model's matrix
        root_model.update_model_matrix(root_model_matrix)

        # If there are no child models, return after updating the root model
        if len(self.models) == 1:
            return

        # Use the root model matrix as the base for child models
        current_parent_matrix = root_model_matrix

        # Process child models (remaining models in the list)
        for model, relative_pos, relative_rot in self.models[1:]:
            # Create translation matrix
            translation_matrix = glm.translate(glm.mat4(1.0), relative_pos)
            #print(f"Translation matrix for child model:\n{translation_matrix}")

            # Create rotation matrices for each axis and combine them
            rotation_x = glm.rotate(glm.mat4(1.0), glm.radians(relative_rot.x), glm.vec3(1.0, 0.0, 0.0))
            rotation_y = glm.rotate(rotation_x, glm.radians(relative_rot.y), glm.vec3(0.0, 1.0, 0.0))
            rotation_z = glm.rotate(rotation_y, glm.radians(relative_rot.z), glm.vec3(0.0, 0.0, 1.0))
            rotation_matrix = rotation_z
            #print(f"Rotation matrix for child model:\n{rotation_matrix}")

            # Create scale matrix
            scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(model.scale, model.scale, model.scale))
            #print(f"Scale matrix for child model:\n{scale_matrix}")

            # Combine current parent matrix with translation, rotation, and scale matrices
            model_matrix = current_parent_matrix * translation_matrix * rotation_matrix * scale_matrix
            #print(f"Combined model matrix for child model:\n{model_matrix}")

            # Store the computed model matrix
            self.model_matrices.append(model_matrix)

            # Update the model's matrix
            model.update_model_matrix(model_matrix)

        #print(f"Final updated composite model matrices:\n{self.model_matrices}")

    def update_flat_model_matrix(self, parent_matrix=glm.mat4(1.0)):
        self.model_matrices.clear()
        for model, rel_pos, rel_rot in self.models:
            translation_matrix = glm.translate(glm.mat4(1.0), rel_pos)
            rotation_x = glm.rotate(glm.mat4(1.0), glm.radians(rel_rot.x), glm.vec3(1.0, 0.0, 0.0))
            rotation_y = glm.rotate(rotation_x, glm.radians(rel_rot.y), glm.vec3(0.0, 1.0, 0.0))
            rotation_z = glm.rotate(rotation_y, glm.radians(rel_rot.z), glm.vec3(0.0, 0.0, 1.0))
            rotation_matrix = rotation_z
            scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(model.scale, model.scale, model.scale))
            model_matrix = parent_matrix * translation_matrix * rotation_matrix * scale_matrix
            self.model_matrices.append(model_matrix)
            model.update_model_matrix(model_matrix)

    def add_comp_model(self, model, relative_position=glm.vec3(0.0, 0.0, 0.0),
                       relative_rotation=glm.vec3(0.0, 0.0, 0.0)):
        self.models.append((model, relative_position, relative_rotation))
        self.set_relative_transform(model, relative_position, relative_rotation)
        self.update_composite_model_matrix(model.model_matrix)

    def set_relative_transform(self, model, relative_position: glm.vec3, relative_rotation: glm.vec3):
        # Calculate the new position of the model relative to the parent model
        new_model_position = self.composite_position + relative_position

        # Calculate the new rotation of the model relative to the parent model
        new_model_rotation = self.composite_rotation + relative_rotation

        # Update the model's position, orientation and matrix
        model.set_position(new_model_position)
        model.set_orientation(new_model_rotation)
        model.update_model_matrix()

    def calculate_composite_centroid(self):
        # Initialize lists to store the coordinates of all centroids
        centroids_x = []
        centroids_y = []
        centroids_z = []

        # Calculate centroid for each model and accumulate their coordinates
        for model, relative_translation, relative_rotation in self.models:
            centroid = model.calculate_centroid()
            centroids_x.append(centroid.x)
            centroids_y.append(centroid.y)
            centroids_z.append(centroid.z)

        # Calculate the mean of each coordinate to find the composite centroid
        composite_centroid_x = np.mean(centroids_x)
        composite_centroid_y = np.mean(centroids_y)
        composite_centroid_z = np.mean(centroids_z)

        # Return the composite centroid as a glm.vec3 object
        return glm.vec3(composite_centroid_x, composite_centroid_y, composite_centroid_z)

    def set_composite_position(self, position):
        for model in self.get_objects():
            #print("set composite position for ", model.name, " to ", model.position)
            model.set_position(position)

    def set_composite_rotation(self, rotation):
        for model in self.get_objects():
            #print("set composite rotation for ", model.name + " to ", model.orientation)
            model.set_orientation(rotation)
