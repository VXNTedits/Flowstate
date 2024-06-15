import sys
import glm
import numpy as np

from camera import Camera

from model import Model
from interactable import InteractableObject


class Player(Model):
    def __init__(self,
                 body_path: str,
                 head_path: str,
                 right_arm_path:
                 str, mtl_path: str,
                 camera,
                 default_material,
                 filepath: str,
                 mtl_filepath: str):
        self.is_player = True
        self.player_height = 2
        self.player_width = 1
        self.default_material = default_material
        self.camera = camera
        self.torso = Model(body_path, mtl_path, player=True, translation=(0, 1, 0))
        self.head = Model(head_path, mtl_path, player=True, translation=(0, 1, 0))
        self.right_arm = Model(right_arm_path, mtl_path, player=True, translation=(0, 1, 0))
        self.set_hand_position()
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
        self.trajectory = self.position - self.previous_position
        self.inventory = []
        self.interact = False
        self.name = 'ego'
        self.jump_cooldown_limit = 1
        self.jump_cooldown = self.jump_cooldown_limit
        self.proposed_thrust = self.thrust
        self.bounding_box = self.calculate_player_bounding_box(self.previous_position, self.position)

    def update_player(self, delta_time: float):
        #self.apply_forces(delta_time)
        self.previous_position = glm.vec3(self.position)  # Update previous position before changing current position
        self.position += self.velocity * delta_time
        self.update_camera_position()
        self.set_hand_position()
        self.update_player_model_matrix()
        self.trajectory = glm.normalize(self.position - self.previous_position)
        if self.velocity.y < 0:
            self.is_jumping = False
        self.calculate_player_bounding_box(self.previous_position, self.position)

    def reset_thrust(self):
        self.thrust = glm.vec3(0.0, 0.0, 0.0)

    def handle_input(self, direction: str, delta_time: float):
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
            self.is_jumping = True
            self.velocity.y = self.jump_force
            self.is_grounded = False
            print('jump: updated velocity to', self.velocity)

    def propose_updated_thrust(self, direction: str, delta_time: float):
        front = glm.vec3(glm.cos(glm.radians(self.yaw)), 0, glm.sin(glm.radians(self.yaw)))
        right = glm.normalize(glm.cross(front, self.up))
        proposed_thrust = glm.vec3(0, 0, 0)

        if direction == 'FORWARD':
            proposed_thrust += front * self.accelerator
        if direction == 'BACKWARD':
            proposed_thrust -= front * self.accelerator
        if direction == 'LEFT':
            proposed_thrust -= right * self.accelerator
        if direction == 'RIGHT':
            proposed_thrust += right * self.accelerator
        if direction == 'JUMP' and self.is_grounded:
            self.is_jumping = True
            self.velocity.y = self.jump_force
            self.is_grounded = False
            print('jump: updated velocity to', self.velocity)

        # Propose the new position based on thrust
        self.proposed_thrust = proposed_thrust# * delta_time


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
        local_hand_position = glm.vec3(0, -0.5, 1)  #glm.vec3(0, -0.5, 1)  # Example position at the end of the arm
        # Transform the local hand position to world coordinates using the right arm's model matrix
        world_hand_position = glm.vec3(
            self.right_arm.model_matrix * glm.vec4(-local_hand_position.x, local_hand_position.y, local_hand_position.z,
                                                   1.0))

    def calculate_player_bounding_box(self, start_pos: glm.vec3, end_pos: glm.vec3, bounding_margin=0.1):
        min_x = min(start_pos.x, end_pos.x) - self.player_width / 2 - bounding_margin
        max_x = max(start_pos.x, end_pos.x) + self.player_width / 2 + bounding_margin
        min_y = min(start_pos.y, end_pos.y) - bounding_margin
        max_y = max(start_pos.y, end_pos.y) + self.player_height + bounding_margin
        min_z = min(start_pos.z, end_pos.z) - self.player_width / 2 - bounding_margin
        max_z = max(start_pos.z, end_pos.z) + self.player_width / 2 + bounding_margin
        #print('calculated player bounding box = ', [(min_x, min_y, min_z), (max_x, max_y, max_z)])
        return [(min_x, min_y, min_z), (max_x, max_y, max_z)]
