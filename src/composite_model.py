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
                 shift_to_centroid=False
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
        self.accumulator = 0

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

        if parent_matrix is None:
            parent_matrix = glm.mat4(1.0)

        if not self.models:
            print("Warning: no self.models found!")
            return

        root_model = self.models[0][0]
        root_model.update_model_matrix(parent_matrix, debug=False)
        self.model_matrices.append(root_model.model_matrix)

        current_root_model_matrix = root_model.model_matrix

        for model, relative_pos, relative_rot in self.models[1:]:
            translation_matrix = glm.translate(glm.mat4(1.0), relative_pos)
            rotation_x = glm.rotate(glm.mat4(1.0), glm.radians(relative_rot.x), glm.vec3(1.0, 0.0, 0.0))
            rotation_y = glm.rotate(glm.mat4(1.0), glm.radians(relative_rot.y), glm.vec3(0.0, 1.0, 0.0))
            rotation_z = glm.rotate(glm.mat4(1.0), glm.radians(relative_rot.z), glm.vec3(0.0, 0.0, 1.0))
            rotation_matrix = rotation_z * rotation_y * rotation_x
            relative_transform = translation_matrix * rotation_matrix

            model.model_matrix = current_root_model_matrix * relative_transform

            self.model_matrices.append(model.model_matrix)
            model.position = self.position + relative_pos
            model.set_orientation(self.orientation + relative_rot)

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
        model.position=relative_position
        model.orientation=relative_rotation
        self.set_relative_transform(model, relative_position, relative_rotation)
        self.models.append((model, relative_position, relative_rotation))
        self.update_composite_model_matrix(model.model_matrix)

    def set_relative_transform(self, model, relative_position: glm.vec3, relative_rotation: glm.vec3):
        """Sets the pose of the specified model relative to the parent."""

        # Calculate the new position of the model relative to the parent model
        new_model_position = self.position + relative_position

        # Calculate the new rotation of the model relative to the parent model
        new_model_rotation = self.orientation + relative_rotation

        # Update the model's position and orientation
        model.set_position(new_model_position)
        model.set_orientation(new_model_rotation)

        # Update self.models with the new relative position and rotation
        for i, (m, _, _) in enumerate(self.models):
            if m == model:
                self.models[i] = (m, relative_position, relative_rotation)
                break

        # Recalculate the model matrix
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
            # print("set composite rotation for ", model.name + " to ", model.orientation)
            model.set_orientation(rotation)

    def define_joint_locations(self, child_index, joint_position, joint_axis):
        """
        Define the joint locations for child models relative to the parent model.

        :param child_index: Index of the child model in the self.models list.
        :param joint_position: Position of the joint relative to the parent model.
        :param joint_axis: Axis of rotation of the joint.
        """
        if child_index <= 0 or child_index >= len(self.models):
            raise ValueError("Child index must be greater than 0 and less than the number of models")

        parent_model, _, _ = self.models[0]
        child_model, _, _ = self.models[child_index]

        # Define the position and axis of the joint relative to the parent model
        child_model.position = joint_position
        child_model.rotation_axis = joint_axis

        print(f"Joint for child model '{child_model.name}' defined at position {joint_position} "
              f"with rotation axis {joint_axis} relative to parent model '{parent_model.name}'.")

    def rotate_child_model(self, child_index, rotation_angle):
        """
        Rotate the child model around its joint axis by the given rotation angle.

        :param child_index: Index of the child model in the self.models list.
        :param rotation_angle: Rotation angle in radians.
        """
        if child_index <= 0 or child_index >= len(self.models):
            raise ValueError("Child index must be greater than 0 and less than the number of models")

        child_model, _, _ = self.models[child_index]
        print(rotation_angle, child_model.rotation_axis)
        child_model.update_transformation_matrix(rotation_angle, child_model.rotation_axis)

        print(f"Child model '{child_model.name}' rotated by {rotation_angle} radians around its joint axis.")

