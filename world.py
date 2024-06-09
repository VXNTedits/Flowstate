import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm
import os

import glm
from model import Model

import os
from model import Model

from model import Model

from model import Model
import glm

from model import Model
import glm

class World(Model):
    def __init__(self, filepath: str, rotation_angles=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0)):
        super().__init__(filepath)
        self.set_orientation(rotation_angles)
        self.set_position(translation)

    def set_orientation(self, rotation_angles):
        # Apply rotations around x, y, z axes respectively
        rotation_matrix = glm.mat4(1.0)
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[0]), glm.vec3(1.0, 0.0, 0.0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[1]), glm.vec3(0.0, 1.0, 0.0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[2]), glm.vec3(0.0, 0.0, 1.0))
        self.model_matrix = rotation_matrix * self.model_matrix

    def set_position(self, translation):
        # Apply translation
        translation_matrix = glm.translate(glm.mat4(1.0), glm.vec3(translation[0], translation[1], translation[2]))
        self.model_matrix = translation_matrix * self.model_matrix
