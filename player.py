from camera import Camera
from model import Model
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm
import os

import glm

import glm
from model import Model


import glm
from model import Model

class Player:
    def __init__(self, model_path: str, camera):
        self.model = Model(model_path)
        self.position = glm.vec3(10.0, 10.0, -10.0)  # Initialize at the specified position
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.speed = 2.5
        self.camera = camera
        self.velocity = glm.vec3(0.0, 0.0, 0.0)

    def update_position(self, direction: str, delta_time: float):
        self.velocity = glm.vec3(0.0, 0.0, 0.0)
        velocity = self.speed * delta_time
        front = glm.vec3(glm.cos(glm.radians(self.camera.yaw)), 0, glm.sin(glm.radians(self.camera.yaw)))
        right = glm.normalize(glm.cross(front, self.up))

        if direction == 'FORWARD':
            self.velocity += front * velocity
        if direction == 'BACKWARD':
            self.velocity -= front * velocity
        if direction == 'LEFT':
            self.velocity -= right * velocity
        if direction == 'RIGHT':
            self.velocity += right * velocity

        self.position += self.velocity
        self.camera.position = self.position  # Ensure camera follows player
        self.update_model_matrix()
        print(f"Player position: {self.position} Camera position: {self.camera.position}")

    def update_model_matrix(self):
        self.model.model_matrix = glm.translate(glm.mat4(1.0), self.position)

    def set_position(self, position):
        self.position = position
        self.update_model_matrix()

    def draw(self):
        self.model.draw()

