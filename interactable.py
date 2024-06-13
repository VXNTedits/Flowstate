import glm
import glfw
from model import Model
from OpenGL.GL import *


class InteractableObject(Model):
    def __init__(self,
                 filepath,
                 mtl_filepath,
                 translation,
                 interactable=True,
                 scale=1,
                 rotation=glm.vec3(0, 0, 0),
                 velocity=glm.vec3(0, 0, 0),
                 is_collidable=False,
                 material_overrides=None
                 ):

        self.position = translation
        self.orientation = rotation
        self.update_model_matrix()
        self.interactable = interactable
        self.interaction_threshold = 200
        self.name = filepath
        self.scale = scale
        self.velocity = velocity
        self.bounce_amplitude = 5.5
        self.bounce_frequency = 2.0  # in cycles per second
        self.rotation_speed = glm.vec3(0, 45, 0)  # degrees per second
        self.picked_up = False
        # self.centroid = self.calculate_centroid()

        super().__init__(
            filepath,
            mtl_filepath,
            scale=scale,
            translation=translation,
            rotation_angles=rotation,
            is_collidable=is_collidable,
            shift_to_centroid=True,
        )
        if material_overrides:
            self.material_overrides = material_overrides

        print("centroid = ", self.centroid)
        print(f"Interactable {self.name} initialized at {self.position}")

    def interact(self, player):
        if self.interactable:
            print(self.orientation)
            self.on_pickup(player)

    def on_pickup(self, player):
        # Define what happens when the player picks up this object
        print(f"{self.name} picked up by {player.name}")
        print("Initial orientation:", self.orientation)

        # Set orientation using set_orientation method
        #self.set_orientation([self.orientation.x, -90, self.orientation.z])

        print("Updated orientation:", self.orientation)

        player.inventory.append(self)
        self.interactable = False
        self.picked_up = True

        # Update model matrix after changing orientation
        self.update_model_matrix()

    def set_orientation(self, rotation_angles):
        # Apply rotations around x, y, z axes respectively
        rotation_matrix = glm.mat4(1.0)
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[0]), glm.vec3(1.0, 0.0, 0.0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[1]), glm.vec3(0.0, 1.0, 0.0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[2]), glm.vec3(0.0, 0.0, 1.0))

        self.orientation = glm.vec3(rotation_angles)
        self.model_matrix = rotation_matrix * self.model_matrix

    def update_model_matrix(self, parent_matrix: glm.mat4 = None,
                            translation_offset: glm.vec3 = glm.vec3(0.0, 0.0, 0.0),
                            rotation_offset: glm.vec3 = glm.vec3(0.0, 0.0, 0.0),
                            player=None):

        if parent_matrix is not None:
            hand_offset = player.right_hand
            translate_to_center = glm.translate(glm.mat4(1.0), -hand_offset)
            body_rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(player.yaw), player.position)
            translate_back = glm.translate(glm.mat4(1.0), hand_offset)
            hand_model_matrix = translate_back * body_rotation_matrix * translate_to_center
            self.model_matrix = hand_model_matrix
            # # Create translation matrix for the translation offset
            # translation_offset_matrix = glm.translate(glm.mat4(1.0), translation_offset)
            #
            # # Create rotation matrices for the rotation offset
            # rotation_offset_x = glm.rotate(glm.mat4(1.0), glm.radians(rotation_offset.x), glm.vec3(1.0, 0.0, 0.0))
            # rotation_offset_y = glm.rotate(glm.mat4(1.0), glm.radians(rotation_offset.y), glm.vec3(0.0, 1.0, 0.0))
            # rotation_offset_z = glm.rotate(glm.mat4(1.0), glm.radians(rotation_offset.z), glm.vec3(0.0, 0.0, 1.0))
            #
            # # Combine rotation offsets to form the rotation offset matrix
            # rotation_offset_matrix = rotation_offset_z * rotation_offset_y * rotation_offset_x
            #
            # # Combine parent matrix with translation and rotation offsets
            # # The order of multiplication here is important
            # self.model_matrix = parent_matrix
            # self.model_matrix = rotation_offset_matrix * translation_offset_matrix * self.model_matrix
            # # Ensure local_model_matrix is properly initialized and contains the object's local transformations
        else:
            # Create translation matrix for the object's position
            translation_matrix = glm.translate(glm.mat4(1.0), self.position)

            # Create rotation matrices for the object's orientation
            rotation_x = glm.rotate(glm.mat4(1.0), glm.radians(self.orientation.x), glm.vec3(1.0, 0.0, 0.0))
            rotation_y = glm.rotate(glm.mat4(1.0), glm.radians(self.orientation.y), glm.vec3(0.0, 1.0, 0.0))
            rotation_z = glm.rotate(glm.mat4(1.0), glm.radians(self.orientation.z), glm.vec3(0.0, 0.0, 1.0))

            # Combine rotations to form the object's rotation matrix
            rotation_matrix = rotation_z * rotation_y * rotation_x

            # Combine translation and rotation to form the local model matrix
            local_model_matrix = translation_matrix * rotation_matrix
            # Handle case where there is no parent matrix
            self.model_matrix = local_model_matrix

    def check_interactions(self, player, delta_time):
        if glm.distance(player.position, self.position) < self.interaction_threshold:
            #print("Highlight should take place now")
            self.highlight(delta_time)
            if player.interact:
                #print("interaction should take place now")
                self.interact(player)
                player.pick_up(self)
                self.update_model_matrix()

    def update(self, player, delta_time):
        if self.interactable:
            self.check_interactions(player, delta_time)
        if self.picked_up:
            self.model_matrix = self.position_relative_to(player.right_arm.model_matrix, self.model_matrix, glm.vec3(-0.4, -0.2, 1.0))#player.right_arm.model_matrix
            #self.update_model_matrix()

    def position_relative_to(self,first_model_matrix, second_model_matrix, relative_position):
        """
        Positions the second object relative to the first object.

        :param first_model_matrix: glm.mat4, the model matrix of the first object
        :param second_model_matrix: glm.mat4, the initial model matrix of the second object
        :param relative_position: glm.vec3, the relative position of the second object with respect to the first
        :return: glm.mat4, the new model matrix of the second object
        """

        # Create the translation matrix using the relative position
        translation_matrix = glm.translate(glm.mat4(1.0), relative_position)

        # Apply the translation relative to the first object's position and orientation
        new_model_matrix = first_model_matrix * translation_matrix #* second_model_matrix

        return new_model_matrix


    def highlight(self, delta_time):
        # Rotate around the y-axis
        rotation_angle = self.rotation_speed.y * delta_time
        self.orientation.y += rotation_angle
        self.orientation.y %= 360  # Keep the angle within [0, 360)

        # Use the precomputed centroid
        centroid = self.centroid

        # Update the position and orientation based on rotation around centroid
        self.update_position_and_orientation_with_centroid(centroid, glm.vec3(0, rotation_angle, 0), delta_time)

        # Bounce up and down (apply after rotation to avoid interference)
        bounce_offset = self.bounce_amplitude * glm.sin(2.0 * glm.pi() * self.bounce_frequency * glfw.get_time())
        self.position.y += bounce_offset * delta_time

        # Update the model matrix with the new position
        self.update_model_matrix()

    def rotate_about_centroid(self, player, rotation_angles: glm.vec3):
        # Translate to centroid
        translate_to_centroid = glm.translate(glm.mat4(1.0), -player.torso.calculate_centroid())

        # Create rotation matrices
        rotation_x = glm.rotate(glm.mat4(1.0), glm.radians(rotation_angles.x), glm.vec3(1.0, 0.0, 0.0))
        rotation_y = glm.rotate(glm.mat4(1.0), glm.radians(rotation_angles.y), glm.vec3(0.0, 1.0, 0.0))
        rotation_z = glm.rotate(glm.mat4(1.0), glm.radians(rotation_angles.z), glm.vec3(0.0, 0.0, 1.0))

        # Translate back from centroid
        translate_back = glm.translate(glm.mat4(1.0), player.torso.calculate_centroid())

        # Combine transformations
        transformation_matrix = translate_back * rotation_x * rotation_y * rotation_z * translate_to_centroid

        # Apply the transformation to the original position (without previous updates)
        original_position = self.position
        new_position = glm.vec3(transformation_matrix * glm.vec4(original_position, 1.0))

        # Update position and orientation
        self.position = new_position
        self.update_model_matrix()

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
        self.update_model_matrix()
