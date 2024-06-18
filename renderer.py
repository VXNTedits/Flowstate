import glm
import numpy as np
import ctypes
from OpenGL.GL import *
import model
from interactable import InteractableObject
from model import Model
from player import Player
from shader import Shader, ShaderManager
from world import World


class Renderer:
    def __init__(self, shader, camera):

        self.light_positions = [
            glm.vec3(50.2, 10.0, 2.0),
            glm.vec3(10.2, 20.0, 2.0),
            glm.vec3(0.0, 20.0, -20.0)
        ]
        self.light_count = len(self.light_positions)

        self.light_colors = [
            glm.vec3(1.0, 0.07, 0.58),  # Neon Pink
            glm.vec3(0.0, 1.0, 0.38),  # Neon Green
            glm.vec3(0.07, 0.55, 0.8)  # Neon Blue
        ]

        self.shader = shader
        self.camera = camera
        self.shadow_width = 2048
        self.shadow_height = 2048

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        glViewport(0, 0, 800, 600)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        # Get shaders from the ShaderManager
        self.depth_shader = ShaderManager.get_shader('shaders/depth_vertex.glsl',
                                                     'shaders/depth_fragment.glsl')
        self.shadow_shader = ShaderManager.get_shader('shaders/shadow_vertex.glsl',
                                                      'shaders/shadow_fragment.glsl')
        self.shader = ShaderManager.get_shader('shaders/vertex_shader.glsl',
                                               'shaders/fragment_shader.glsl')
        self.volumetric_shader = ShaderManager.get_shader('shaders/volumetric_vertex.glsl',
                                                          'shaders/volumetric_fragment.glsl')
        self.emissive_shader = ShaderManager.get_shader('shaders/emissive_vertex.glsl',
                                                        'shaders/emissive_fragment.glsl')
        self.debug_depth_shader = Shader("debug_vertex.glsl", "debug_fragment.glsl")


        # Set up VAO and VBO for the quad
        self.quadVAO = glGenVertexArrays(1)
        self.quadVBO = glGenBuffers(1)
        # Set up VAO and VBO for the quad
        self.quadVAO = glGenVertexArrays(1)
        self.quadVBO = glGenBuffers(1)

        quadVertices = [
            -1.0, 1.0, 0.0, 0.0, 1.0,
            -1.0, -1.0, 0.0, 0.0, 0.0,
            1.0, -1.0, 0.0, 1.0, 0.0,

            -1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, -1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 1.0
        ]
        quadVertices = np.array(quadVertices, dtype=np.float32)

        glBindVertexArray(self.quadVAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.quadVBO)
        glBufferData(GL_ARRAY_BUFFER, quadVertices.nbytes, quadVertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * quadVertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * quadVertices.itemsize,
                              ctypes.c_void_p(3 * quadVertices.itemsize))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        #self.depth_map_fbo = glGenFramebuffers(1)
        #self.depth_map = glGenTextures(1)

        self.setup_depth_framebuffer()
        self.setup_quad()


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
        self.init_volumetric_fbo()

    def check_framebuffer_status(self):
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"Framebuffer is not complete: {status}")
        else:
            print("Framebuffer is complete.")

    def calculate_light_space_matrix(self, light_position):
        near_plane = 0.1
        far_plane = 1000.0
        light_proj = glm.ortho(-20.0, 20.0, -20.0, 20.0, near_plane, far_plane)
        light_view = glm.lookAt(light_position, glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
        return light_proj * light_view

    def render(self, player_object, world, interactables, view_matrix, projection_matrix):
        light_count = len(self.light_positions)

        # 1. Render the depth map
        glViewport(0, 0, self.shadow_width, self.shadow_height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fbo)
        glClear(GL_DEPTH_BUFFER_BIT)

        #print("Using shadow shader")
        self.render_shadow_map(player_object, world, interactables, self.light_positions)

        # 2. Render the scene normally
        glViewport(0, 0, 800, 600)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #print("Using main shader")
        self.shader.use()
        self.shader.set_uniform_matrix4fv("view", view_matrix)
        self.shader.set_uniform_matrix4fv("projection", projection_matrix)

        for i, (pos, color) in enumerate(zip(self.light_positions, self.light_colors)):
            self.shader.set_uniform_matrix4fv(f"lightSpaceMatrix[{i}]", self.calculate_light_space_matrix(pos))
            self.shader.set_uniform3f(f"lights[{i}].position", pos)
            self.shader.set_uniform3f(f"lights[{i}].color", color)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.depth_map)
        self.shader.set_uniform1i("shadowMap", 1)

        self.shader.set_bump_scale(5.0)
        self.shader.set_roughness(0.1)

        self.render_scene(self.shader, player_object, world, interactables, view_matrix, projection_matrix)
        self.render_lights(self.light_positions, self.light_colors, view_matrix, projection_matrix)

        # 3. Render volumetric effects
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        #print("Using volumetric shader")
        self.volumetric_shader.use()
        self.volumetric_shader.set_uniform_matrix4fv("view", view_matrix)
        self.volumetric_shader.set_uniform_matrix4fv("projection", projection_matrix)
        self.volumetric_shader.set_uniform1i("lightCount", light_count)
        self.volumetric_shader.set_uniform1f("lightScatteringCoefficient", 1.0)  # Example value

        for i, (pos, color) in enumerate(zip(self.light_positions, self.light_colors)):
            self.volumetric_shader.set_uniform3f(f"lights[{i}].position", pos)
            self.volumetric_shader.set_uniform3f(f"lights[{i}].color", color)
        self.volumetric_shader.set_uniform3f("cameraPosition", self.camera.position)

        self.render_volumetrics(self.volumetric_shader, player_object, world, interactables, view_matrix,
                                projection_matrix)

        glDisable(GL_BLEND)

        #self.render_depth_map_from_player(world, view_matrix, projection_matrix)

    def render_lights(self, light_positions, light_colors, view_matrix, projection_matrix):
        self.light_positions = light_positions
        self.light_colors = light_colors
        self.emissive_shader.use()
        self.emissive_shader.set_uniform_matrix4fv("view", view_matrix)
        self.emissive_shader.set_uniform_matrix4fv("projection", projection_matrix)

        for pos, color in zip(light_positions, light_colors):
            model_matrix = glm.mat4(1.0)
            model_matrix = glm.translate(model_matrix, pos)
            self.emissive_shader.set_uniform_matrix4fv("model", model_matrix)
            self.emissive_shader.set_uniform3f("lightColor", color)
            self.render_light_object(size=3)

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
        self.shader.use()
        #print(f"Updating uniforms for object: {model}")
        #print(f"Model matrix: {model_matrix}")
        #print(f"View matrix: {view_matrix}")
        #print(f"Projection matrix: {projection_matrix}")
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

    def render_object_with_volumetrics(self, shader, obj, view_matrix, projection_matrix):
        #print()
        #print(obj.name)

        if isinstance(obj, World):
            model_matrix = obj.model_matrix
        elif isinstance(obj, Player):
            model_matrix = obj.model_matrix
        elif isinstance(obj, InteractableObject):
            model_matrix = obj.model_matrix
        elif isinstance(obj, model.Model):
            model_matrix = obj.model_matrix
        else:
            raise TypeError(f"Unknown object type: {type(obj)}")

        if view_matrix and projection_matrix:
            self.update_uniforms(model_matrix, view_matrix, projection_matrix, obj)

        shader.set_uniform_matrix4fv("model", model_matrix)

        if isinstance(obj, Player):
            obj.draw(self.camera)
        else:
            obj.draw()

        # Special handling for composite models in interactables
        if isinstance(obj, InteractableObject):
            obj._model.update_composite_model_matrix(model_matrix)
            for mod, pos, dir in obj._model.models:
                model_matrix = mod.model_matrix
                if view_matrix and projection_matrix:
                    self.update_uniforms(model_matrix, view_matrix, projection_matrix, mod)
                shader.set_uniform_matrix4fv("model", model_matrix)
                mod.draw()

    def render_volumetrics(self, volumetric_shader, player_object, world, interactables, view_matrix,
                           projection_matrix):
        glBindFramebuffer(GL_FRAMEBUFFER, self.volumetric_fbo)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        volumetric_shader.use()
        volumetric_shader.set_uniform_matrix4fv("view", view_matrix)
        volumetric_shader.set_uniform_matrix4fv("projection", projection_matrix)

        for i, (pos, color) in enumerate(zip(self.light_positions, self.light_colors)):
            volumetric_shader.set_uniform3f(f"lights[{i}].position", pos)
            volumetric_shader.set_uniform3f(f"lights[{i}].color", color)

        for obj in world.get_objects():
            self.render_object_with_volumetrics(volumetric_shader, obj, view_matrix, projection_matrix)

        self.render_object_with_volumetrics(volumetric_shader, player_object, view_matrix, projection_matrix)

        for interactable in interactables:
            self.render_object_with_volumetrics(volumetric_shader, interactable, view_matrix, projection_matrix)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def init_volumetric_fbo(self):
        self.volumetric_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.volumetric_fbo)

        self.volumetric_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.volumetric_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 800, 600, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.volumetric_texture, 0)

        self.volumetric_depth_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.volumetric_depth_buffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 800, 600)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.volumetric_depth_buffer)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Framebuffer is not complete!")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def render_quad(self):
        if not hasattr(self, 'quadVAO'):
            quadVertices = [
                # Positions        # Texture Coords
                -1.0,  1.0, 0.0,   0.0, 1.0,
                -1.0, -1.0, 0.0,   0.0, 0.0,
                 1.0, -1.0, 0.0,   1.0, 0.0,
                 1.0,  1.0, 0.0,   1.0, 1.0,
            ]
            quadIndices = [
                0, 1, 2,
                0, 2, 3
            ]
            quadVertices = np.array(quadVertices, dtype=np.float32)
            quadIndices = np.array(quadIndices, dtype=np.uint32)

            self.quadVAO = glGenVertexArrays(1)
            self.quadVBO = glGenBuffers(1)
            self.quadEBO = glGenBuffers(1)

            glBindVertexArray(self.quadVAO)
            glBindBuffer(GL_ARRAY_BUFFER, self.quadVBO)
            glBufferData(GL_ARRAY_BUFFER, quadVertices.nbytes, quadVertices, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.quadEBO)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, quadIndices.nbytes, quadIndices, GL_STATIC_DRAW)

            # Position attribute
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * quadVertices.itemsize, ctypes.c_void_p(0))
            # TexCoord attribute
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * quadVertices.itemsize, ctypes.c_void_p(3 * quadVertices.itemsize))

            glBindVertexArray(0)

        glBindVertexArray(self.quadVAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def render_depth_map(self, player, world, interactables):
        # Bind the depth map framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fbo)
        glClear(GL_DEPTH_BUFFER_BIT)

        # Use the shadow shader
        self.shadow_shader.use()

        # Assuming you have a method to get light positions, and you need to calculate the light space matrix
        light_positions = [
            glm.vec3(50.2, 10.0, 2.0),
            glm.vec3(10.2, 20.0, 2.0),
            glm.vec3(0.0, 20.0, -20.0)
        ]

        for pos in light_positions:
            light_space_matrix = self.calculate_light_space_matrix(pos)
            self.shadow_shader.set_uniform_matrix4fv("lightSpaceMatrix", light_space_matrix)
            self.render_scene(self.shadow_shader, player, world,
                              interactables, light_space_matrix)

        # Unbind the framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Now bind the depth map for visualization
        glViewport(0, 0, 800, 600)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.depth_shader.use()  # Use the simple shader to visualize depth
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.depth_map)
        self.render_quad()  # Render a quad to visualize the depth map

    def render_shadow_map(self, player_object, world, interactables, light_positions):
        glViewport(0, 0, self.shadow_width, self.shadow_height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fbo)
        glClear(GL_DEPTH_BUFFER_BIT)

        self.shadow_shader.use()
        for pos in light_positions:
            light_space_matrix = self.calculate_light_space_matrix(pos)
            self.shadow_shader.set_uniform_matrix4fv("lightSpaceMatrix", light_space_matrix)
            self.render_world(self.shadow_shader, player_object, world, interactables, light_space_matrix)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def setup_depth_framebuffer(self):
        # Generate texture
        self.depth_map = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.depth_map)

        # Define texture storage and parameters
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 800, 600, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        border_color = [1.0, 1.0, 1.0, 1.0]
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)

        # Create and bind the framebuffer
        self.depth_map_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth_map, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        # Check framebuffer completeness
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("Framebuffer not complete")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def glm_to_numpy(self, mat):
        return np.array(mat, dtype=np.float32).flatten()

    def glm_to_ctypes(self, mat):
        return (ctypes.c_float * 16)(*mat.flatten())

    def render_depth_map_from_player(self, world, view_matrix, projection_matrix):
        # Bind the depth map framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fbo)
        glClear(GL_DEPTH_BUFFER_BIT)

        # Use the shadow shader for rendering the depth map
        self.shadow_shader.use()

        # Convert matrices to NumPy arrays
        view_matrix_np = self.glm_to_numpy(view_matrix)
        projection_matrix_np = self.glm_to_numpy(projection_matrix)

        # Set view and projection matrices
        view_location = glGetUniformLocation(self.shadow_shader.program, "view")
        projection_location = glGetUniformLocation(self.shadow_shader.program, "projection")
        glUniformMatrix4fv(view_location, 1, GL_FALSE, view_matrix_np)
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, projection_matrix_np)

        # Render objects to the depth map
        for obj in world.get_objects():
            model_matrix_np = self.glm_to_numpy(obj.model_matrix)
            model_location = glGetUniformLocation(self.shadow_shader.program, "model")
            glUniformMatrix4fv(model_location, 1, GL_FALSE, model_matrix_np)
            self.render_object(self.shadow_shader, obj, view_matrix, projection_matrix)

        # Unbind the framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Debugging: Render the depth map directly to see if it contains data
        glViewport(0, 0, 800, 600)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.depth_shader.use()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.depth_map)
        self.render_quad()

        # Optional: Check for OpenGL errors
        error = glGetError()
        if error != GL_NO_ERROR:
            print(f"OpenGL error: {error}")

        # Optional: Render depth values to the screen for debugging
        self.render_debug_depth_texture()

    def render_debug_depth_texture(self):
        self.debug_depth_shader.use()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.depth_map)
        self.render_quad()

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

    def render_world(self, shader, player_object, world, interactables, light_space_matrix, view_matrix=None,
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

    def render_object(self, shader, obj, view_matrix, projection_matrix):
        model_matrix = obj.model_matrix
        if view_matrix and projection_matrix:
            self.update_uniforms(model_matrix, view_matrix, projection_matrix, obj)
        shader.set_uniform_matrix4fv("model", model_matrix)
        obj.draw()

    def setup_quad(self):
        # Define the quad vertices and indices
        quadVertices = np.array([
            # Positions        # Texture Coords
            -1.0,  1.0, 0.0,   0.0, 1.0,
            -1.0, -1.0, 0.0,   0.0, 0.0,
             1.0, -1.0, 0.0,   1.0, 0.0,
             1.0,  1.0, 0.0,   1.0, 1.0,
        ], dtype=np.float32)
        quadIndices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        # Generate and bind VAO
        self.quadVAO = glGenVertexArrays(1)
        self.quadVBO = glGenBuffers(1)
        self.quadEBO = glGenBuffers(1)

        glBindVertexArray(self.quadVAO)

        # Bind and set VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.quadVBO)
        glBufferData(GL_ARRAY_BUFFER, quadVertices.nbytes, quadVertices, GL_STATIC_DRAW)

        # Bind and set EBO
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.quadEBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, quadIndices.nbytes, quadIndices, GL_STATIC_DRAW)

        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * quadVertices.itemsize, ctypes.c_void_p(0))

        # TexCoord attribute
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * quadVertices.itemsize, ctypes.c_void_p(3 * quadVertices.itemsize))

        glBindVertexArray(0)

