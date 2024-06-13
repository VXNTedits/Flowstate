import sys
import glm
import numpy as np

from camera import Camera

from model import Model
from interactable import InteractableObject

class Player(Model):
    def __init__(self, body_path: str, head_path: str, right_arm_path: str, mtl_path: str, camera, default_material,
                 filepath: str, mtl_filepath: str):
        self.default_material = default_material
        self.camera = camera
        self.torso = Model(body_path, mtl_path, player=True, translation=(0, 1, 0))
        self.head = Model(head_path, mtl_path, player=True, translation=(0, 1, 0))
        self.right_arm = Model(right_arm_path, mtl_path, player=True, translation=(0, 1, 0))
        self.right_hand_model_matrix = self.set_hand_position()
        self.position = glm.vec3(10.0, 10.2, -10.0)
        self.previous_position = glm.vec3(10.0, 10.2, -10.0)
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.accelerator = 10
        self.jump_force = 5
        self.max_speed = 10.0
        self.thrust = glm.vec3(0.0, 0.0, 0.0)
        self.velocity = glm.vec3(0, 0, 0)
        self.yaw = camera.yaw
        self.pitch = camera.pitch
        self.rotation = glm.vec3(camera.pitch, camera.yaw, 0)
        self.vertices, self.indices = Model.load_obj(self, body_path)
        self.model_matrix = glm.rotate(glm.mat4(1.0), glm.radians(-90), glm.vec3(1.0, 0.0, 0.0))
        self.update_player_model_matrix()
        self.is_grounded = False
        self.is_jumping = False
        self.displacement = self.position - self.previous_position
        self.inventory = []
        self.interact = False
        self.name = 'ego'

    def update(self, delta_time: float):
        if delta_time <= 0:
            print("Delta time is zero or negative!")
            return
        self.apply_forces(delta_time)
        self.previous_position = glm.vec3(self.position)  # Update previous position before changing current position
        self.position += self.velocity * delta_time
        self.update_camera_position()
        self.right_hand = self.set_hand_position()
        self.update_player_model_matrix()
        self.displacement = self.position - self.previous_position
        # Reset the is_jumping flag if the player has landed
        if self.is_grounded:
            self.is_jumping = False

    def apply_forces(self, delta_time: float):
        # Apply vertical thrust if jumping
        if self.thrust.y > 0 and self.is_grounded:
            self.velocity.y = self.thrust.y
            self.is_grounded = False
            self.is_jumping = True

        # Toggle off the is_jumping flag if the player is falling downwards
        if self.velocity.y < 0.0:
            self.is_jumping = False

        # Apply lateral thrust for movement
        if self.thrust.x != 0.0 or self.thrust.z != 0.0:
            # Calculate desired lateral velocity
            desired_velocity = glm.normalize(glm.vec3(self.thrust.x, 0.0, self.thrust.z)) * self.max_speed
            # Use a lerp function to smoothly interpolate towards the target velocity
            lerp_factor = 1 - np.exp(-self.accelerator * delta_time)
            self.velocity.x = self.velocity.x * (1 - lerp_factor) + desired_velocity.x * lerp_factor
            self.velocity.z = self.velocity.z * (1 - lerp_factor) + desired_velocity.z * lerp_factor
        else:
            # Apply braking force to lateral movement
            deceleration = np.exp(-self.accelerator * delta_time)
            self.velocity.x *= deceleration
            self.velocity.z *= deceleration
            # Ensure that the lateral velocity does not invert due to overshooting the deceleration
            if glm.length(glm.vec3(self.velocity.x, 0.0, self.velocity.z)) ** 2 < 0.01:
                self.velocity.x = 0.0
                self.velocity.z = 0.0

        # Ensure vertical velocity does not invert due to overshooting the deceleration
        if abs(self.velocity.y) < 0.01:
            self.velocity.y = 0.0

        #print(f"Updated Velocity: {self.velocity}")

    def reset_thrust(self):
        self.thrust = glm.vec3(0.0, 0.0, 0.0)

    def update_position(self, direction: str, delta_time: float):
        front = glm.vec3(glm.cos(glm.radians(self.yaw)), 0, glm.sin(glm.radians(self.yaw)))
        right = glm.normalize(glm.cross(front, self.up))
        up = self.up

        if direction == 'FORWARD':
            self.thrust.x += front.x * self.accelerator
            self.thrust.z += front.z * self.accelerator
        if direction == 'BACKWARD':
            self.thrust.x -= front.x * self.accelerator
            self.thrust.z -= front.z * self.accelerator
        if direction == 'LEFT':
            self.thrust.x -= right.x * self.accelerator
            self.thrust.z -= right.z * self.accelerator
        if direction == 'RIGHT':
            self.thrust.x += right.x * self.accelerator
            self.thrust.z += right.z * self.accelerator
        if direction == 'JUMP' and self.is_grounded:
            self.thrust.y += up.y * self.jump_force

    def update_player_model_matrix(self):
        # Rotate around the yaw axis (Y-axis)
        model_rotation = glm.rotate(glm.mat4(1.0), glm.radians(-self.yaw), glm.vec3(0.0, 1.0, 0.0))

        # Additional rotations if necessary
        model_rotation *= glm.rotate(glm.mat4(1.0), glm.radians(90), glm.vec3(0.0, 1.0, 0.0))
        model_rotation *= glm.rotate(glm.mat4(1.0), glm.radians(-90), glm.vec3(1.0, 0.0, 0.0))

        # Apply translation after rotations
        translation = glm.translate(glm.mat4(1.0), self.position)
        final_model_matrix = translation * model_rotation

        # Update model matrices for torso and right_arm
        # Assuming torso and right_arm are at specific offsets from the main model
        self.torso.model_matrix = final_model_matrix
        self.right_arm.model_matrix = final_model_matrix  # Modify if right_arm has an offset
        self.head.model_matrix = final_model_matrix
        # Update the main model matrix
        self.rotation = model_rotation
        self.model_matrix = self.torso.model_matrix

    def get_rotation_matrix(self):
        return glm.rotate(glm.mat4(1.0), glm.radians(self.yaw), glm.vec3(0.0, 1.0, 0.0))

    def set_position(self, position):
        self.position = position
        self.update_player_model_matrix()

    def draw(self, camera: Camera):
        if not camera.first_person:
            self.head.draw()
            self.torso.draw()
            self.right_arm.draw()
        else:
            self.torso.draw()
            self.right_arm.draw()

    def process_mouse_movement(self, xoffset, yoffset):
        self.camera.process_mouse_movement(xoffset, yoffset)
        self.yaw = self.camera.yaw
        self.pitch = self.camera.pitch
        self.update_camera_position()

    def update_camera_position(self):
        self.camera.yaw = self.yaw
        self.camera.pitch = self.pitch
        self.camera.update_camera_vectors()
        if self.camera.first_person:
            self.camera.set_first_person(self.position, self.get_rotation_matrix())
        else:
            self.camera.set_third_person(self.position, self.get_rotation_matrix())

    def pick_up(self, interactable_object):
        if isinstance(interactable_object, InteractableObject):
            interactable_object.interact(self)

    def set_hand_position(self):
        # Define the local position of the hand relative to the right arm
        local_hand_position = glm.vec3(0, -0.5,1)#glm.vec3(0, -0.5, 1)  # Example position at the end of the arm
        # Transform the local hand position to world coordinates using the right arm's model matrix
        world_hand_position = glm.vec3(self.right_arm.model_matrix * glm.vec4(-local_hand_position.x, local_hand_position.y, local_hand_position.z, 1.0))

