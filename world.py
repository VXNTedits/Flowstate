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

class World(Model):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        # Any additional initialization for the world can go here
