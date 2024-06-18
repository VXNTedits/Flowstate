import glm
import glfw

import composite_model
from composite_model import CompositeModel
from model import Model
from OpenGL.GL import *


class InteractableObject:
    def __init__(self,
                 filepath,
                 mtl_filepath,
                 translation,
                 interactable=True,
                 scale=1,
                 rotation=glm.vec3(0, 0, 0),
                 velocity=glm.vec3(0, 0, 0),
                 is_collidable=False,
                 material_overrides=None,
                 use_composite=False
                 ):
        if use_composite:
            self._model = CompositeModel(filepath,
                                         mtl_filepath,
                                         player=False,
                                         draw_convex_only=False,
                                         rotation_angles=rotation,
                                         translation=translation,
                                         kd_override=None,
                                         ks_override=None,
                                         ns_override=None,
                                         scale=scale,
                                         is_collidable=is_collidable,
                                         shift_to_centroid=True
                                         )
        else:
            self._model = Model(filepath,
                                mtl_filepath,
                                player=False,
                                draw_convex_only=False,
                                rotation_angles=rotation,
                                translation=translation,
                                kd_override=None,
                                ks_override=None,
                                ns_override=None,
                                scale=scale,
                                is_collidable=is_collidable,
                                shift_to_centroid=True
                                )
        self.use_composite = use_composite
        self.interactable = interactable
        self.interaction_threshold = 1000
        self.velocity = velocity
        self.bounce_amplitude = 10
        self.bounce_frequency = 2.0  # in cycles per second
        self.rotation_speed = glm.vec3(0, 45, 0)  # degrees per second
        self.picked_up = False

        # Create a CompositeModel to manage multiple sub-models
        #self.composite_model = CompositeModel()

        if material_overrides:
            self.material_overrides = material_overrides

        print("centroid = ", self.centroid)
        print(f"Interactable {self.name} initialized at {self.position}")

    def __getattr__(self, name):
        try:
            return getattr(self._model, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def add_sub_model(self, sub_model, relative_position, relative_rotation, scale):
        if self.use_composite:
            self.models.append([sub_model, relative_position, relative_rotation])
            self._model.set_scale(scale)
            self._model.set_position(relative_position)
            self._model.set_orientation(relative_rotation)
            self._model.update_model_matrix(parent_matrix=self.model_matrix)
        else:
            self.model.add_model(sub_model, relative_position, relative_rotation)
            self.models.add_model(sub_model, relative_position, relative_rotation)
            self.model.set_scale(scale)
            self.model.set_position(relative_position)
            self.model.set_orientation(relative_rotation)
            self.model.update_player_model_matrix(parent_matrix=self.model_matrix)

    def interact(self, player):
        if self.interactable:
            print(self.orientation)
            self.on_pickup(player)

    def on_pickup(self, player):
        # Define what happens when the player picks up this object
        print(f"{self.name} picked up by {player.name}")
        print("Initial orientation:", self.orientation)
        if self.interactable:
            self._model.position = glm.vec3(-0.4, -0.1, 1)  #player.position
            self._model.orientation = glm.vec3(0, 0, 0)  #glm.vec3(player.camera.pitch,player.camera.yaw,0.0)
            #self._model.set_scale(self.scale)
        print("Reset orientation:", self.orientation)
        player.inventory.append(self)
        self.interactable = False
        self.picked_up = True
        # Update model matrix after changing orientation
        self.update_interactable_model_matrix(player.torso.model_matrix)

    def update_interactable_model_matrix(self, parent_matrix=None):
        self.model_matrices.clear()

        # Create translation matrix for the object's position
        translation_matrix = glm.translate(glm.mat4(1.0), self._model.position)

        # Create rotation matrices for the object's orientation
        rotation_x = glm.rotate(glm.mat4(1.0), glm.radians(self._model.orientation[0]), glm.vec3(1.0, 0.0, 0.0))
        rotation_y = glm.rotate(glm.mat4(1.0), glm.radians(self._model.orientation[1]), glm.vec3(0.0, 1.0, 0.0))
        rotation_z = glm.rotate(glm.mat4(1.0), glm.radians(self._model.orientation[2]), glm.vec3(0.0, 0.0, 1.0))

        # Combine rotations to form the object's rotation matrix
        rotation_matrix = rotation_z * rotation_y * rotation_x

        # Create scaling matrix for the object's scale
        scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(self._model.scale, self._model.scale, self._model.scale))

        # Combine translation, rotation, and scale to form the local model matrix
        local_model_matrix = translation_matrix * rotation_matrix * scale_matrix

        # Handle case where there is no parent matrix
        if parent_matrix is None:
            self.model_matrix = local_model_matrix
        else:
            self.model_matrix = parent_matrix * local_model_matrix

        # Update the composite model's model matrix
        if self.use_composite:
            self._model.update_model_matrix()

    def update(self, player, delta_time):
        if self.interactable:
            self.check_interactions(player, delta_time)
        if self.picked_up:
            self.update_interactable_model_matrix(player.right_arm.model_matrix)
        else:
            self.update_interactable_model_matrix()  # Ensure the model matrix is updated for non-picked objects
        player.interact = False

    def check_interactions(self, player, delta_time):
        if glm.distance(player.position, self.position) < self.interaction_threshold:
            self.highlight(delta_time)
            if player.interact:
                print("INTERACTED!")
                self.interact(player)
                player.pick_up(self)
                self.update_interactable_model_matrix()


    def highlight(self, delta_time):
        # Rotate around the y-axis
        rotation_angle = self.rotation_speed.y * delta_time
        self._model.orientation[1] += rotation_angle
        self._model.orientation[1] %= 360  # Keep the angle within [0, 360)

        # Use the precomputed centroid
        centroid = self.centroid

        # Update the position and orientation based on rotation around centroid
        self.update_position_and_orientation_with_centroid(centroid, glm.vec3(0, rotation_angle, 0), delta_time)

        # Bounce up and down (apply after rotation to avoid interference)
        bounce_offset = self.bounce_amplitude * glm.sin(2.0 * glm.pi() * self.bounce_frequency * glfw.get_time())
        self.position.y += bounce_offset * delta_time

        # Update the model matrix with the new position
        self.update_interactable_model_matrix()

    def update_position_and_orientation_with_centroid(self, centroid, rotation_angle, delta_time):
        # Translate to centroid
        translate_to_centroid = glm.translate(glm.mat4(1.0), -centroid)

        # Create rotation matrix for y-axis
        rotation_y = glm.rotate(glm.mat4(1.0), glm.radians(rotation_angle.y), glm.vec3(0.0, 1.0, 0.0))

        # Translate back from centroid
        translate_back = glm.translate(glm.mat4(1.0), centroid)

        # Combine transformations
        transformation_matrix = translate_back * rotation_y * translate_to_centroid

        # Apply the transformation to the original position (without previous updates)
        original_position = self.position - glm.vec3(0, self.bounce_amplitude * glm.sin(
            2.0 * glm.pi() * self.bounce_frequency * glfw.get_time()) * delta_time, 0)
        new_position = glm.vec3(transformation_matrix * glm.vec4(original_position, 1.0))

        # Update position and orientation
        self.position = new_position
        self.update_interactable_model_matrix()
