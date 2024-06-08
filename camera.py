import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm
import os

import glm


import glm

class Camera:
    def __init__(self, position: glm.vec3, up: glm.vec3, yaw: float = -90.0, pitch: float = 0.0):
        self.position = position
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = up
        self.right = glm.vec3()
        self.world_up = up
        self.yaw = yaw
        self.pitch = pitch
        self.mode = 'FIRST_PERSON'
        self.offset = glm.vec3(0.0, 2.0, 5.0)  # Offset for third-person view
        self.sensitivity = 0.1  # Mouse sensitivity
        self.speed = 2.5  # Camera movement speed
        self.update_camera_vectors()

    def get_view_matrix(self, player_position: glm.vec3) -> glm.mat4:
        if self.mode == 'FIRST_PERSON':
            self.position = player_position
            return glm.lookAt(self.position, self.position + self.front, self.up)
        elif self.mode == 'THIRD_PERSON':
            self.position = player_position - self.front * self.offset.z + glm.vec3(0.0, self.offset.y, 0.0)
            return glm.lookAt(self.position, player_position, self.up)

    def process_keyboard(self, direction: str, delta_time: float):
        velocity = self.speed * delta_time
        if direction == 'FORWARD':
            self.position += self.front * velocity
        if direction == 'BACKWARD':
            self.position -= self.front * velocity
        if direction == 'LEFT':
            self.position -= glm.normalize(glm.cross(self.front, self.up)) * velocity
        if direction == 'RIGHT':
            self.position += glm.normalize(glm.cross(self.front, self.up)) * velocity

    def process_mouse_movement(self, xoffset: float, yoffset: float, constrain_pitch: bool = True):
        xoffset *= self.sensitivity
        yoffset *= self.sensitivity
        self.yaw += xoffset
        self.pitch += yoffset
        if constrain_pitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0
        self.update_camera_vectors()

    def update_camera_vectors(self):
        front = glm.vec3()
        front.x = np.cos(glm.radians(self.yaw)) * np.cos(glm.radians(self.pitch))
        front.y = np.sin(glm.radians(self.pitch))
        front.z = np.sin(glm.radians(self.yaw)) * np.cos(glm.radians(self.pitch))
        self.front = glm.normalize(front)
        self.right = glm.normalize(glm.cross(self.front, self.world_up))
        self.up = glm.normalize(glm.cross(self.right, self.front))

    def toggle_view(self):
        if self.mode == 'FIRST_PERSON':
            self.mode = 'THIRD_PERSON'
        else:
            self.mode = 'FIRST_PERSON'
