import time

import glm
from OpenGL.GL import *

from model import Model


class Renderer:
    def __init__(self, shader, camera):
        self.shader = shader
        self.camera = camera
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, 800, 600)  # Set the viewport
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Set clear color (black)

    def render_player(self, player_object, view_matrix, projection_matrix):
        self.shader.use()
        model_matrix = player_object.model_matrix
        self.update_uniforms(model_matrix, view_matrix, projection_matrix, player_object)
        player_object.draw()

    def render_world(self, world, view_matrix, projection_matrix):
        self.shader.use()
        model_matrix = world.model_matrix
        self.update_uniforms(model_matrix, view_matrix, projection_matrix, world)
        world.draw()

    def update_uniforms(self, model_matrix, view_matrix, projection_matrix, model):
        self.shader.set_uniform_matrix4fv("model", model_matrix)
        self.shader.set_uniform_matrix4fv("view", view_matrix)
        self.shader.set_uniform_matrix4fv("projection", projection_matrix)
        self.shader.set_uniform3f("viewPos", self.camera.position)

        # Animate light properties
        current_time = time.time()

        # Use sinusoidal functions to oscillate between 0 and 1
        def oscillate(frequency, phase_shift=0):
            return (glm.sin(current_time * frequency + phase_shift) + 1) / 2

        light_positions = [glm.vec3(100.2, 1.0, 2.0), glm.vec3(-1.2, 2.0, 200.0), glm.vec3(10.0, 3.0, 20.0)]

        # Define base colors
        base_colors = [
            glm.vec3(1.0, 0.08, 0.58),  # Pink
            glm.vec3(0.08, 0.08, 1.0),  # Blue
            glm.vec3(0.08, 1.0, 0.08)   # Green
        ]

        # Animate colors
        animated_colors = [
            glm.vec3(oscillate(1.0), oscillate(0.1, 2), oscillate(0.1, 4)),  # Example frequency for Pink
            glm.vec3(oscillate(0.1, 2), oscillate(0.1, 4), oscillate(0.1, 6)), # Example frequency for Blue
            glm.vec3(oscillate(0.1, 4), oscillate(0.2, 6), oscillate(0.1, 8))  # Example frequency for Green
        ]

        for i, (lightPos, lightColor) in enumerate(zip(light_positions, animated_colors)):
            self.shader.set_uniform3f(f"lights[{i}].position", lightPos)
            self.shader.set_uniform3f(f"lights[{i}].color", lightColor)

        # Set material properties using the first material found
        kd = model.default_material['diffuse']
        ks = model.default_material['specular']
        ns = model.default_material['shininess']

        self.shader.set_uniform3f("objectColor", glm.vec3(*kd))
        self.shader.set_uniform3f("specularColor", glm.vec3(*ks))
        self.shader.set_uniform1f("shininess", ns)
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
        glColor3f(1.0, 1.0, 1.0)  # Set the color for the bounding box (white)
        model.draw()
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glPopAttrib()  # Restore previous polygon mode