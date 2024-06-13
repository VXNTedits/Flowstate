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
                 is_collidable=False
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
            is_collidable=is_collidable
        )

        print("centroid = ", self.centroid)
        print(f"Interactable {self.name} initialized at {self.position}")

    def update_model_matrix(self):
        translation_matrix = glm.translate(glm.mat4(1.0), self.position)
        rotation_x = glm.rotate(glm.mat4(1.0), glm.radians(self.orientation.x), glm.vec3(1.0, 0.0, 0.0))
        rotation_y = glm.rotate(glm.mat4(1.0), glm.radians(self.orientation.y), glm.vec3(0.0, 1.0, 0.0))
        rotation_z = glm.rotate(glm.mat4(1.0), glm.radians(self.orientation.z), glm.vec3(0.0, 0.0, 1.0))
        rotation_matrix = rotation_z * rotation_y * rotation_x
        self.model_matrix = translation_matrix * rotation_matrix

    def interact(self, player):
        if self.interactable:
            self.on_pickup(player)

    def on_pickup(self, player):
        # Define what happens when the player picks up this object
        print(f"{self.name} picked up by {player.name}")
        # You can add code to add this object to the player's inventory, etc.
        player.inventory.append(self)
        self.interactable = False
        self.picked_up = True

    def check_interactions(self, player, delta_time):
        if glm.distance(player.position, self.position) < self.interaction_threshold:
            #print("Highlight should take place now")
            self.highlight(delta_time)
            if player.interact:
                #print("interaction should take place now")
                self.interact(player)
                player.pick_up(self)

    def update(self, player, delta_time):
        if self.interactable:
            self.check_interactions(player, delta_time)
        if self.picked_up:
            self.orientation = glm.vec3(player.camera.pitch, -player.yaw, 0)
            self.position = player.right_hand
            #print(f"pitch = {self.orientation.x} yaw = {self.orientation.y}, roll = {self.orientation.z}")
            self.update_model_matrix()

    def highlight(self, delta_time):
        # Rotate around the y-axis
        rotation_angle = self.rotation_speed.y * delta_time
        self.orientation.y += rotation_angle
        self.orientation.y %= 360  # Keep the angle within [0, 360)

        # Use the precomputed centroid
        centroid = self.centroid

        # Update the position and orientation based on rotation around centroid
        self.update_position_and_orientation_with_centroid(centroid, glm.vec3(0,rotation_angle,0), delta_time)

        # Bounce up and down (apply after rotation to avoid interference)
        bounce_offset = self.bounce_amplitude * glm.sin(2.0 * glm.pi() * self.bounce_frequency * glfw.get_time())
        self.position.y += bounce_offset * delta_time

        # Update the model matrix with the new position
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
