import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm
import os

from camera import Camera
from model import Model
from shader import Shader


import glm
from OpenGL.GL import *
from shader import Shader

class Renderer:
    def __init__(self, shader: Shader, camera):
        self.shader = shader
        self.camera = camera
        glEnable(GL_DEPTH_TEST)

    def render(self, model, player_position, rotate: bool = True, is_world: bool = False):
        self.shader.use()
        self.update_uniforms(player_position, rotate, is_world)
        model.draw()

    def update_uniforms(self, player_position, rotate: bool, is_world: bool):
        fov = 90.0
        near_clip = 0.1
        far_clip = 10000.0
        projection = glm.perspective(glm.radians(fov), 800.0 / 600.0, near_clip, far_clip)
        view = self.camera.get_view_matrix(player_position)
        model = glm.mat4(1.0)

        if is_world:
            # Rotate the world to lay flat
            model = glm.rotate(model, glm.radians(-90.0), glm.vec3(1.0, 0.0, 0.0))
        elif rotate:
            # Rotate other models
            model = glm.rotate(model, glm.radians(glfw.get_time() * 50), glm.vec3(0.0, 1.0, 0.0))

        self.shader.set_uniform_matrix4fv("projection", projection)
        self.shader.set_uniform_matrix4fv("view", view)
        self.shader.set_uniform_matrix4fv("model", model)
        self.shader.set_uniform3f("lightPos", glm.vec3(1.2, 1.0, 2.0))
        self.shader.set_uniform3f("viewPos", self.camera.position)
        self.shader.set_uniform3f("lightColor", glm.vec3(1.0, 1.0, 1.0))
        self.shader.set_uniform3f("objectColor", glm.vec3(1.0, 0.5, 0.31))
