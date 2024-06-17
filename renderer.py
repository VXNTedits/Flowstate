import glm
import numpy as np
from OpenGL.GL import *
import model
from model import Model
from shader import Shader


class Renderer:
    def __init__(self, shader, camera):
        self.shader = shader
        self.camera = camera
        self.shadow_width = 2048
        self.shadow_height = 2048
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, 800, 600)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        self.shadow_shader = Shader('shadow_vertex.glsl', 'shadow_fragment.glsl')
        self.emissive_shader = Shader('emissive_vertex.glsl', 'emissive_fragment.glsl')
        self.depth_map_fbo = glGenFramebuffers(1)
        self.depth_map = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depth_map)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.shadow_width, self.shadow_height, 0, GL_DEPTH_COMPONENT,
                     GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        border_color = [1.0, 1.0, 1.0, 1.0]
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)
        glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth_map, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        self.check_framebuffer_status()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def check_framebuffer_status(self):
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"Framebuffer is not complete: {status}")
        else:
            print("Framebuffer is complete.")

    def calculate_light_space_matrix(self, light_position):
        near_plane = 1.0
        far_plane = 100.0
        light_proj = glm.ortho(-20.0, 20.0, -20.0, 20.0, near_plane, far_plane)
        light_view = glm.lookAt(light_position, glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
        return light_proj * light_view

    def render_scene(self, shader, player_object, world, interactables, light_space_matrix, view_matrix=None,
                     projection_matrix=None):
        shader.use()
        shader.set_uniform_matrix4fv("lightSpaceMatrix", light_space_matrix)

        # Render world objects
        for obj in world.get_objects():
            model_matrix = obj.model_matrix
            if view_matrix and projection_matrix:
                self.update_uniforms(model_matrix, view_matrix, projection_matrix, obj)
            shader.set_uniform_matrix4fv("model", model_matrix)
            obj.draw()

        # Render player
        model_matrix = player_object.model_matrix
        if view_matrix and projection_matrix:
            self.update_uniforms(model_matrix, view_matrix, projection_matrix, player_object)
        shader.set_uniform_matrix4fv("model", model_matrix)
        player_object.draw(self.camera)

        # Render interactables
        for interactable in interactables:
            model_matrix = interactable.model_matrix
            if view_matrix and projection_matrix:
                self.update_uniforms(model_matrix, view_matrix, projection_matrix, interactable)
            shader.set_uniform_matrix4fv("model", model_matrix)
            interactable.draw()
            interactable._model.update_composite_model_matrix(model_matrix)
            for mod, pos, dir in interactable._model.models:
                model_matrix = mod.model_matrix
                if view_matrix and projection_matrix:
                    self.update_uniforms(model_matrix, view_matrix, projection_matrix, mod)
                shader.set_uniform_matrix4fv("model", model_matrix)
                mod.draw()

    def render(self, player_object, world, interactables, view_matrix, projection_matrix):
        light_positions = [
            glm.vec3(50.2, 10.0, 2.0),
            glm.vec3(10.2, 20.0, 2.0),
            glm.vec3(0.0, 20.0, -20.0)
        ]
        light_colors = [
            glm.vec3(1, 0.0, 0),
            glm.vec3(0.0, 1, 0.0),
            glm.vec3(0.0, 0.0, 1)
        ]

        glViewport(0, 0, self.shadow_width, self.shadow_height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fbo)
        glClear(GL_DEPTH_BUFFER_BIT)

        light_space_matrices = []
        for pos in light_positions:
            light_space_matrix = self.calculate_light_space_matrix(pos)
            light_space_matrices.append(light_space_matrix)
            self.shadow_shader.use()
            self.shadow_shader.set_uniform_matrix4fv("lightSpaceMatrix", light_space_matrix)
            self.render_scene(self.shadow_shader, player_object, world, interactables, light_space_matrix)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glViewport(0, 0, 800, 600)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.shader.use()
        self.shader.set_uniform_matrix4fv("view", view_matrix)
        self.shader.set_uniform_matrix4fv("projection", projection_matrix)

        for i, (pos, color) in enumerate(zip(light_positions, light_colors)):
            self.shader.set_uniform_matrix4fv(f"lightSpaceMatrix[{i}]", light_space_matrices[i])
            self.shader.set_uniform3f(f"lights[{i}].position", pos)
            self.shader.set_uniform3f(f"lights[{i}].color", color)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.depth_map)
        self.shader.set_uniform1i("shadowMap", 1)

        self.shader.set_bump_scale(1.0)
        self.shader.set_roughness(0.5)

        self.render_scene(self.shader, player_object, world, interactables, light_space_matrix, view_matrix,
                          projection_matrix)

        self.render_lights(light_positions, light_colors, view_matrix, projection_matrix)

    def render_lights(self, light_positions, light_colors, view_matrix, projection_matrix):
        self.emissive_shader.use()
        self.emissive_shader.set_uniform_matrix4fv("view", view_matrix)
        self.emissive_shader.set_uniform_matrix4fv("projection", projection_matrix)

        for pos, color in zip(light_positions, light_colors):
            model_matrix = glm.mat4(1.0)
            model_matrix = glm.translate(model_matrix, pos)
            self.emissive_shader.set_uniform_matrix4fv("model", model_matrix)
            self.emissive_shader.set_uniform3f("lightColor", color)
            self.render_light_object(size=0.2)

    def render_light_object(self, size):
        half_size = size / 2.0
        vertices = [
            -half_size, -half_size, -half_size,
            half_size, -half_size, -half_size,
            half_size, half_size, -half_size,
            -half_size, half_size, -half_size,
            -half_size, -half_size, half_size,
            half_size, -half_size, half_size,
            half_size, half_size, half_size,
            -half_size, half_size, half_size,
        ]

        indices = [
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            0, 1, 5, 5, 4, 0,
            2, 3, 7, 7, 6, 2,
            0, 3, 7, 7, 4, 0,
            1, 2, 6, 6, 5, 1,
        ]

        VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)
        EBO = glGenBuffers(1)

        glBindVertexArray(VAO)

        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, np.array(vertices, dtype=np.float32), GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.array(indices, dtype=np.uint32), GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)
        glDeleteBuffers(1, [VBO])
        glDeleteVertexArrays(1, [VAO])
        glDeleteBuffers(1, [EBO])

    def update_uniforms(self, model_matrix, view_matrix, projection_matrix, model: Model):
        self.shader.set_uniform_matrix4fv("model", model_matrix)
        self.shader.set_uniform_matrix4fv("view", view_matrix)
        self.shader.set_uniform_matrix4fv("projection", projection_matrix)

        if model:
            kd = model.default_material['diffuse']
            ks = model.default_material['specular']
            ns = model.default_material['shininess']
            roughness = model.default_material.get('roughness', 0.5)
            bump_scale = model.default_material.get('bumpScale', 1.0)

            self.shader.set_uniform3f("objectColor", glm.vec3(*kd))
            self.shader.set_uniform3f("specularColor", glm.vec3(*ks))
            self.shader.set_uniform1f("shininess", ns)
            self.shader.set_roughness(roughness)
            self.shader.set_bump_scale(bump_scale)

