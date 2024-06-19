import glm
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
        self.position = translation
        self.rotation = rotation_angles
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
        self.models.append((self, self.position, self.rotation))

    def add_world_model(self, model, scale, relative_position=glm.vec3(0.0, 0.0, 0.0),
                        relative_rotation=glm.vec3(0.0, 0.0, 0.0)):
        self.models.append((model, relative_position, relative_rotation))
        model.set_scale(scale)
        model.set_position(relative_position)
        model.set_orientation(relative_rotation)
        model.init_model_matrix(relative_position, relative_rotation)
        self.update_composite_model_matrix()  # Ensure initial update without explicitly passing identity matrix

    def get_objects(self):
        # Return all models contained in this composite model
        return [model for model, _, _ in self.models]

    def update_composite_model_matrix(self, parent_matrix=glm.mat4(1.0)):
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

    # def __getattr__(self, name):
    #     return getattr(self._model, name)

    def add_comp_model(self, model, scale, relative_position=glm.vec3(0.0, 0.0, 0.0),
                       relative_rotation=glm.vec3(0.0, 0.0, 0.0)):
        self.models.append((model, relative_position, relative_rotation))
        model.set_scale(scale)
        self.set_relative_transform(model, relative_position, relative_rotation)
        self.update_composite_model_matrix()

    def set_relative_transform(self, model, relative_position: glm.vec3, relative_rotation: glm.vec3):
        # Calculate the new position of the model relative to the parent model
        new_model_position = self.position + relative_position

        # Calculate the new rotation of the model relative to the parent model
        new_model_rotation = self.rotation + relative_rotation

        # Set the new position and rotation for the model
        model.position = new_model_position
        model.rotation = new_model_rotation

        # Update the model's orientation and matrix
        model.set_orientation(new_model_rotation)
        model.update_model_matrix()

