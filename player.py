import sys

import glm
import numpy as np

from model import Model

import glm
from model import Model

import glm
from model import Model

class Player(Model):
    def __init__(self, model_path: str, mtl_path: str, camera, default_material):
        self.default_material = default_material
        self.camera = camera
        self.model = Model(model_path, mtl_path, player=True)
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
        self.rotation = glm.vec3(camera.pitch, camera.yaw, 0)
        self.vertices, self.indices = Model.load_obj(self, model_path)
        self.model_matrix = glm.rotate(glm.mat4(1.0), glm.radians(-90), glm.vec3(1.0, 0.0, 0.0))
        self.update_model_matrix()
        self.is_grounded = False
        self.is_jumping = False
        self.displacement = self.position - self.previous_position

    def set_origin(self, new_origin):
        self.model.translate(new_origin)
        self.position = new_origin
        self.update_model_matrix()

    def update(self, delta_time: float):
        self.apply_forces(delta_time)
        self.position += self.velocity * delta_time
        self.update_camera_position()
        self.update_model_matrix()
        self.displacement = self.position - self.previous_position
        sys.stdout.write(f"\rCALL: displacement update: {self.displacement}")
        sys.stdout.flush()

        # Reset the is_jumping flag if the player has landed
        if self.is_grounded:
            self.is_jumping = False

    def apply_forces(self, delta_time: float):
        # Apply vertical thrust if jumping
        if self.thrust.y > 1.0 and self.is_grounded:
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
        self.previous_position = self.position


    def update_model_matrix(self):
        model_rotation = (
            glm.rotate(glm.mat4(1.0), glm.radians(-self.yaw), glm.vec3(0.0, 1.0, 0.0)) *
            glm.rotate(glm.mat4(1.0), glm.radians(90), glm.vec3(0.0, 1.0, 0.0)) *
            glm.rotate(glm.mat4(1.0), glm.radians(-90), glm.vec3(1.0, 0.0, 0.0))
        )
        translation = glm.translate(glm.mat4(1.0), self.position)
        self.model.model_matrix = translation * model_rotation
        self.model_matrix = self.model.model_matrix

    def get_rotation_matrix(self):
        return glm.rotate(glm.mat4(1.0), glm.radians(self.yaw), glm.vec3(0.0, 1.0, 0.0))

    def set_position(self, position):
        self.position = position
        self.update_model_matrix()

    def draw(self):
        self.model.draw()

    def process_mouse_movement(self, xoffset, yoffset):
        self.camera.process_mouse_movement(xoffset, yoffset)
        self.yaw = self.camera.yaw
        self.update_camera_position()

    def update_camera_position(self):
        self.camera.yaw = self.yaw
        self.camera.update_camera_vectors()
        if self.camera.first_person:
            self.camera.set_first_person(self.position, self.get_rotation_matrix())
        else:
            self.camera.set_third_person(self.position, self.get_rotation_matrix())
