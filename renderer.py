import time

import glm
import numpy as np
from OpenGL.GL import *

import model
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
        player_object.draw(self.camera)

    def render_world(self, world, view_matrix, projection_matrix):
        self.shader.use()
        model_matrix = world.model_matrix
        self.update_uniforms(model_matrix, view_matrix, projection_matrix, world)
        world.draw()

    def render_aabb(self, list_of_objects_in_world: list, player_pos: glm.vec3, view_matrix, projection_matrix):
        self.shader.use()
        for obj in list_of_objects_in_world:
            min_corner, max_corner = obj.aabb

            vertices = [
                glm.vec3(min_corner[0], min_corner[1], min_corner[2]),
                glm.vec3(min_corner[0], min_corner[1], max_corner[2]),
                glm.vec3(min_corner[0], max_corner[1], min_corner[2]),
                glm.vec3(min_corner[0], max_corner[1], max_corner[2]),
                glm.vec3(max_corner[0], min_corner[1], min_corner[2]),
                glm.vec3(max_corner[0], min_corner[1], max_corner[2]),
                glm.vec3(max_corner[0], max_corner[1], min_corner[2]),
                glm.vec3(max_corner[0], max_corner[1], max_corner[2]),
            ]

            edges = [
                (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
                (4, 5), (4, 6), (5, 7), (6, 7)
            ]

            for edge in edges:
                self.draw_line(vertices[edge[0]], vertices[edge[1]], view_matrix, projection_matrix)

        self.draw_player_position(player_pos, view_matrix, projection_matrix)

    def draw_line(self, start, end, view_matrix, projection_matrix):
        model_matrix = glm.mat4(1.0)
        self.update_uniforms(model_matrix, view_matrix, projection_matrix, None)

        vertices = [start.x, start.y, start.z, end.x, end.y, end.z]
        vertex_array = np.array(vertices, dtype=np.float32)

        VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)

        glBindVertexArray(VAO)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Set the line width for visibility
        glLineWidth(4.0)

        # Set a solid color for debugging (e.g., red)
        self.shader.set_uniform3f("objectColor", glm.vec3(1.0, 1.0, 1.0))

        glDrawArrays(GL_LINES, 0, 2)

        glBindVertexArray(0)
        glDeleteBuffers(1, [VBO])
        glDeleteVertexArrays(1, [VAO])

    def draw_player_position(self, player_pos, view_matrix, projection_matrix):
        model_matrix = glm.mat4(1.0)
        self.update_uniforms(model_matrix, view_matrix, projection_matrix, None)

        vertices = [player_pos.x, player_pos.y, player_pos.z]
        vertex_array = np.array(vertices, dtype=np.float32)

        VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)

        glBindVertexArray(VAO)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Set the point size for visibility
        glPointSize(10.0)  # Adjust the size as needed

        # Set a solid color for debugging (e.g., blue)
        self.shader.set_uniform3f("objectColor", glm.vec3(0.0, 0.0, 1.0))

        glDrawArrays(GL_POINTS, 0, 1)

        glBindVertexArray(0)
        glDeleteBuffers(1, [VBO])
        glDeleteVertexArrays(1, [VAO])

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
        if model:
            kd = model.default_material['diffuse']
            ks = model.default_material['specular']
            ns = model.default_material['shininess']

            self.shader.set_uniform3f("objectColor", glm.vec3(*kd))
            self.shader.set_uniform3f("specularColor", glm.vec3(*ks))
            self.shader.set_uniform1f("shininess", ns)
