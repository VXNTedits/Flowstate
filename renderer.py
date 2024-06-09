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

from OpenGL.GL import *

from OpenGL.GL import *

class Renderer:
    def __init__(self, shader, camera):
        self.shader = shader
        self.camera = camera
        glEnable(GL_DEPTH_TEST)

    def render(self, model, view_matrix, projection_matrix):
        self.shader.use()
        model_matrix = model.model_matrix
        self.update_uniforms(model_matrix, view_matrix, projection_matrix)
        model.draw()

    def update_uniforms(self, model_matrix, view_matrix, projection_matrix):
        self.shader.set_uniform_matrix4fv("model", model_matrix)
        self.shader.set_uniform_matrix4fv("view", view_matrix)
        self.shader.set_uniform_matrix4fv("projection", projection_matrix)
        self.shader.set_uniform3f("lightPos", glm.vec3(1.2, 1.0, 2.0))
        self.shader.set_uniform3f("viewPos", self.camera.position)
        self.shader.set_uniform3f("lightColor", glm.vec3(0.6, 0.2, 1.0))  # Neon purple light
        self.shader.set_uniform3f("objectColor", glm.vec3(0.2, 0.2, 0.5))  # Dark blue object color

    def draw_model_bounding_box(self, model, view_matrix, projection_matrix):
        # Use a simple shader program for drawing the bounding box
        glUseProgram(self.shader.id)
        self.shader.set_uniform_matrix4fv("model", model.model_matrix)
        self.shader.set_uniform_matrix4fv("view", view_matrix)
        self.shader.set_uniform_matrix4fv("projection", projection_matrix)

        # Save current polygon mode and set line width
        glPushAttrib(GL_POLYGON_BIT)
        glLineWidth(3.0)  # Set the line width for the bounding box

        # Draw the wireframe over the model
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glColor3f(1.0, 1.0, 1.0)  # Set the wireframe color to white
        glDisable(GL_DEPTH_TEST)  # Disable depth test to draw on top of everything else

        glBindVertexArray(model.vao)
        glDrawElements(GL_LINES, len(model.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        # Restore previous polygon mode and re-enable depth test
        glEnable(GL_DEPTH_TEST)
        glPopAttrib()

