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
        self.shift_to_centroid = shift_to_centroid
        self.is_collidable = is_collidable
        self.scale = scale
        self.ns_override = ns_override
        self.ks_override = ks_override
        self.kd_override = kd_override
        self.translation = translation
        self.rotation_angles = rotation_angles
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

    def add_model(self, model, scale, relative_position=glm.vec3(0.0, 0.0, 0.0), relative_rotation=glm.vec3(0.0, 0.0, 0.0)):
        model.set_scale(scale)
        self.models.append((model, relative_position, relative_rotation))
        model.init_model_matrix(relative_position, relative_rotation)
        #self.update_composite_model_matrix(glm.mat4(1.0))  # Ensure initial update with identity matrix

    def update_composite_model_matrix(self, parent_matrix=glm.mat4(1.0)):
        #self.model_matrices.clear()
        #self.set_scale(self.scale)
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

    # def draw(self):
    #     for model, _, _ in self.models:
    #         super().draw()
    #         model.draw()

    # def __getattr__(self, name):
    #     return getattr(self._model, name)
