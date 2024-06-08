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


class Player:
    def __init__(self, model_path: str, camera: Camera):
        self.model = Model(model_path)
        self.position = glm.vec3(0.0, 10.0, 0.0)  # Initialize 10 units above the ground
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.speed = 2.5
        self.camera = camera

    def update_position(self, direction: str, delta_time: float):
        velocity = self.speed * delta_time
        front = glm.vec3(np.cos(glm.radians(self.camera.yaw)), 0, np.sin(glm.radians(self.camera.yaw)))
        right = glm.normalize(glm.cross(front, self.up))

        if direction == 'FORWARD':
            self.position += front * velocity
        if direction == 'BACKWARD':
            self.position -= front * velocity
        if direction == 'LEFT':
            self.position -= right * velocity
        if direction == 'RIGHT':
            self.position += right * velocity

    def draw(self):
        # Update the model matrix before drawing
        model_matrix = glm.translate(glm.mat4(1.0), self.position)
        self.model.model_matrix = model_matrix
        self.model.draw()
