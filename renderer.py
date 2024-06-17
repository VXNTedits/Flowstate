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
        self.shadow_width = 2024
        self.shadow_height = 2024
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, 800, 600)  # Set the viewport
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Set clear color (black)

        # Shadow mapping setup
        self.shadow_shader = Shader('shadow_vertex.glsl', 'shadow_fragment.glsl')
        self.depth_map_fbo = glGenFramebuffers(1)
        self.depth_map = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depth_map)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.shadow_width, self.shadow_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
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

    def render_scene(self, shader):
        # Render your scene here using the given shader
        pass

    def render(self, player_object, world, interactables, view_matrix, projection_matrix):
        # Set light positions and colors
        light_positions = [glm.vec3(1.2, 10.0, 2.0), glm.vec3(-1.2, 20.0, 2.0), glm.vec3(0.0, 20.0, 2.0)]
        light_colors = [glm.vec3(1.0, 1.0, 1.0), glm.vec3(1.0, 1.0, 1.0), glm.vec3(1.0, 1.0, 1.0)]

        light_space_matrix = self.calculate_light_space_matrix(light_positions[0])

        # 1. Render depth of scene to texture (from light's perspective)
        self.shadow_shader.use()
        self.shadow_shader.set_uniform_matrix4fv("lightSpaceMatrix", light_space_matrix)
        glViewport(0, 0, self.shadow_width, self.shadow_height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fbo)
        glClear(GL_DEPTH_BUFFER_BIT)
        self.render_scene(self.shadow_shader)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # 2. Render scene as normal using the generated depth/shadow map
        glViewport(0, 0, 800, 600)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.shader.use()
        self.shader.set_uniform_matrix4fv("view", view_matrix)
        self.shader.set_uniform_matrix4fv("projection", projection_matrix)
        self.shader.set_uniform_matrix4fv("lightSpaceMatrix", light_space_matrix)
        self.shader.set_uniform3f("viewPos", self.camera.position)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.depth_map)
        self.shader.set_uniform1i("shadowMap", 1)

        # Set light positions and colors
        for i, (pos, color) in enumerate(zip(light_positions, light_colors)):
            self.shader.set_uniform3f(f"lights[{i}].position", pos)
            self.shader.set_uniform3f(f"lights[{i}].color", color)

        # Set bump scale and roughness
        self.shader.set_bump_scale(1.0)  # You can adjust the value as needed
        self.shader.set_roughness(0.5)  # You can adjust the value as needed

        # Render your scene objects here
        self.render_world(world, view_matrix, projection_matrix)
        self.render_player(player_object, view_matrix, projection_matrix)
        self.render_interactables(interactables, view_matrix, projection_matrix)

        # Debug: Render depth map to screen for visualization
        self.visualize_depth_map()

    def visualize_depth_map(self):
        glViewport(0, 0, 800, 600)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBindTexture(GL_TEXTURE_2D, self.depth_map)
        # Render a quad that fills the screen with the depth map texture
        quad_vertices = [
            -1.0, 1.0, 0.0, 0.0, 1.0,
            -1.0, -1.0, 0.0, 0.0, 0.0,
            1.0, -1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 1.0,
        ]
        quad_indices = [0, 1, 2, 0, 2, 3]

        VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)
        EBO = glGenBuffers(1)

        glBindVertexArray(VAO)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, np.array(quad_vertices, dtype=np.float32), GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.array(quad_indices, dtype=np.uint32), GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), ctypes.c_void_p(3 * sizeof(GLfloat)))
        glEnableVertexAttribArray(1)

        # Use a simple shader to render the depth map texture
        depth_shader = Shader('depth_vertex.glsl', 'depth_fragment.glsl')
        depth_shader.use()
        depth_shader.set_uniform1i("depthMap", 1)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        glBindVertexArray(0)
        glDeleteBuffers(1, [VBO])
        glDeleteVertexArrays(1, [VAO])
        glDeleteBuffers(1, [EBO])

    def calculate_light_space_matrix(self, light_pos):
        light_projection = glm.ortho(-10.0, 10.0, -10.0, 10.0, 1.0, 7.5)
        light_view = glm.lookAt(light_pos, glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
        return light_projection * light_view

    def render_world(self, world, view_matrix, projection_matrix):
        model_matrix = world.model_matrix
        self.update_uniforms(model_matrix, view_matrix, projection_matrix, world)
        world.draw()

    def render_player(self, player_object, view_matrix, projection_matrix):
        model_matrix = player_object.model_matrix
        self.update_uniforms(model_matrix, view_matrix, projection_matrix, player_object)
        player_object.draw(self.camera)

    def render_interactables(self, interactables: list, view_matrix, projection_matrix):
        for interactable in interactables:
            model_matrix = interactable.model_matrix
            self.update_uniforms(model_matrix, view_matrix, projection_matrix, interactable)
            interactable.draw()
            interactable._model.update_composite_model_matrix(model_matrix)
            for mod, pos, dir in interactable._model.models:
                model_matrix = mod.model_matrix
                self.update_uniforms(model_matrix, view_matrix, projection_matrix, mod)
                mod.draw()

    def update_uniforms(self, model_matrix, view_matrix, projection_matrix, model: Model):
        self.shader.set_uniform_matrix4fv("model", model_matrix)
        self.shader.set_uniform_matrix4fv("view", view_matrix)
        self.shader.set_uniform_matrix4fv("projection", projection_matrix)

        # Set other uniforms like objectColor, specularColor, shininess, roughness, etc.
        if model:
            kd = model.default_material['diffuse']
            ks = model.default_material['specular']
            ns = model.default_material['shininess']
            roughness = model.default_material.get('roughness', 0.5)  # Default to 0.5 if not found
            bump_scale = model.default_material.get('bumpScale', 1.0)  # Default to 1.0 if not found

            self.shader.set_uniform3f("objectColor", glm.vec3(*kd))
            self.shader.set_uniform3f("specularColor", glm.vec3(*ks))
            self.shader.set_uniform1f("shininess", ns)
            self.shader.set_roughness(roughness)
            self.shader.set_bump_scale(bump_scale)