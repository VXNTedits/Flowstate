import glm
import numpy as np
import ctypes
from OpenGL.GL import *
from OpenGL.raw.GL.NVX.gpu_memory_info import GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, \
    GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX
from OpenGL.raw.GLU import gluErrorString

import model
from interactable import InteractableObject
from src.model import Model
from player import Player
from src.shader import Shader, ShaderManager
from world import World
import noise
from opensimplex import OpenSimplex
import ctypes
from OpenGL.GL import *


class Renderer:
    def __init__(self, shader, camera, weapons):
        # Initialization of light properties
        # These lights will be used for various shading and rendering techniques
        print("Initialization of light properties...")
        self.light_intensity = 0.1

        self.light_positions = [
            glm.vec3(50.2, 10.0, 2.0),
            glm.vec3(10.2, 20.0, 2.0),
            glm.vec3(0.0, 20.0, -20.0)
        ]
        self.light_count = len(self.light_positions)

        self.light_colors = [
            glm.vec3(1.0, 0.07, 0.58) * self.light_intensity,  # Neon Pink
            glm.vec3(0.0, 1.0, 0.38) * self.light_intensity,  # Neon Green
            glm.vec3(0.07, 0.45, 0.9) * self.light_intensity  # Neon Blue
        ]

        # Volume bounds for volumetric rendering
        # These define the boundaries of the volume in the scene
        print("Volume bounds for volumetric rendering...")
        self.volume_min = glm.vec3(-1000, -1000, -1000)
        self.volume_max = glm.vec3(1000, 1000, 1000)

        # Store the provided shader and camera references
        self.shader = shader
        self.camera = camera

        # Shadow map resolution
        self.shadow_width = 2048
        self.shadow_height = 2048

        # OpenGL state setup
        # Enable depth testing and back face culling for correct rendering of 3D objects
        print("OpenGL state setup...")
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # Set viewport dimensions and clear color
        glViewport(0, 0, 800, 600)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        # Load and compile shaders from ShaderManager
        print("Load and compile shaders from ShaderManager...")
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
        self.debug_depth_shader = ShaderManager.get_shader("shaders/debug_vertex.glsl",
                                                           "shaders/debug_fragment.glsl")
        self.screen_shader = ShaderManager.get_shader("shaders/screen_vertex.glsl",
                                                      "shaders/screen_fragment.glsl")
        self.composite_shader = ShaderManager.get_shader("shaders/composite_vertex.glsl",
                                                         "shaders/composite_fragment.glsl")
        self.procedural_shader = ShaderManager.get_shader("shaders/procedural_vertex.glsl",
                                                          "shaders/procedural_fragment.glsl")
        self.quad_shader = ShaderManager.get_shader("shaders/quad_vertex.glsl",
                                                    "shaders/quad_fragment.glsl")
        self.tracer_shader = ShaderManager.get_shader("shaders/tracer_vertex.glsl",
                                                      "shaders/tracer_fragment.glsl", use_glsl_430=False)

        # Set up the offscreen buffer for shader compositing
        print("Set up the offscreen buffer for shader compositing...")
        self.setup_offscreen_framebuffer()

        # Generate 3D noise texture for volumetric effects
        # This texture will be used in the volumetric rendering process
        #print("Generate 3D noise texture for volumetric effects...")
        self.noise_size = 512  # Size of the 3D noise texture
        self.simplex = OpenSimplex(seed=np.random.randint(0, 10000))
        self.time = 0.0
        self.setup_volume_texture()

        # Setup depth framebuffer for shadow mapping
        print("Setup depth framebuffer for shadow mapping")
        self.setup_depth_framebuffer()

        # Configure the depth map texture used for shadow mapping
        print("Configure the depth map texture used for shadow mapping")
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

        # Initialize framebuffer for volumetric rendering
        print("Initialize framebuffer for volumetric rendering...")
        self.init_volumetric_fbo()

        # Set up the vertex array and buffer objects for rendering a quad
        self.setup_quad()

        print("Initialize projectile buffers...")
        # Initialize projectile buffers
        self.projectile_vao = glGenVertexArrays(1)
        self.projectile_vbo = glGenBuffers(1)
        self.init_projectile_buffers()
        self.weapons = weapons

    def setup_volume_texture(self):
        # Create an empty 3D texture
        self.volume_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.volume_texture)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, self.noise_size, self.noise_size, self.noise_size, 0, GL_RED, GL_FLOAT,
                     None)

        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_3D, 0)

    def update_noise(self, delta_time):
        self.time += delta_time

    def generate_noise_texture(self, size, time):
        noise_texture = np.zeros((size, size, size), dtype=np.float32)
        scale = 1024.0

        for x in range(size):
            for y in range(size):
                for z in range(size):
                    nx, ny, nz = x / size - 0.5, y / size - 0.5, z / size - 0.5
                    noise_texture[x, y, z] = self.simplex.noise4(nx * scale, ny * scale, nz * scale, time)

        # Normalize the noise texture to be in the range [0, 1]
        normalized_texture = self.normalize_noise_texture(noise_texture)
        return normalized_texture

    def normalize_noise_texture(self, noise_texture):
        min_val = np.min(noise_texture)
        max_val = np.max(noise_texture)
        normalized_texture = (noise_texture - min_val) / (max_val - min_val)
        return normalized_texture

    def setup_depth_framebuffer(self):
        self.depth_map_fbo = glGenFramebuffers(1)
        self.depth_map = glGenTextures(1)

    def setup_quad(self):
        #
        self.quadVertices = [
            # Positions        # Texture Coords
            -1.0, 1.0, 0.0, 0.0, 1.0,
            -1.0, -1.0, 0.0, 0.0, 0.0,
            1.0, -1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 1.0,
        ]
        self.quadIndices = [
            0, 1, 2,
            0, 2, 3
        ]

        # Generate and bind VAO
        self.quadVAO = glGenVertexArrays(1)
        glBindVertexArray(self.quadVAO)

        # Generate and bind VBO
        self.quadVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quadVBO)
        glBufferData(GL_ARRAY_BUFFER, np.array(self.quadVertices, dtype=np.float32), GL_STATIC_DRAW)

        # Generate and bind EBO
        self.quadEBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.quadEBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.array(self.quadIndices, dtype=np.uint32), GL_STATIC_DRAW)

        # Define vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        # Unbind VAO
        glBindVertexArray(0)

    def check_framebuffer_status(self):
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"Framebuffer is not complete: {status}")

    def calculate_light_space_matrix(self, light_position):
        near_plane = 0.1
        far_plane = 1000.0
        light_proj = glm.ortho(-20.0, 20.0, -20.0, 20.0, near_plane, far_plane)
        light_view = glm.lookAt(light_position, glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
        self.light_space_matrix = light_proj * light_view
        return self.light_space_matrix

    def render(self, player_object, world, interactables, world_objects, view_matrix, projection_matrix, delta_time):
        """Main rendering pipeline"""

        # 1. Render the depth map
        self.shadow_shader.use()
        glViewport(0, 0, self.shadow_width, self.shadow_height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fbo)
        glClear(GL_DEPTH_BUFFER_BIT)
        self.render_shadow_map(player_object, world, interactables, self.light_positions)
        self.check_opengl_error()

        assert not glIsEnabled(GL_BLEND), "GL_BLEND should be disabled after depth map rendering"
        assert glIsEnabled(GL_DEPTH_TEST), "GL_DEPTH_TEST should be enabled after depth map rendering"

        # 2. Render the scene to the framebuffer
        self.render_scene_to_fbo(shader=self.shader,
                                 player_object=player_object,
                                 world=world,
                                 interactables=interactables,
                                 world_objects=world_objects,
                                 light_space_matrix=self.light_space_matrix,
                                 view_matrix=view_matrix,
                                 projection_matrix=projection_matrix,
                                 enable_bump_mapping=False,
                                 bump_scale=5)

        # Ensure depth test is enabled for future operations
        glEnable(GL_DEPTH_TEST)
        assert glIsEnabled(GL_DEPTH_TEST), "GL_DEPTH_TEST should be enabled for future operations"

        # 3. Render volumetric effects to the framebuffer
        self.render_volumetric_effects_to_fbo(view_matrix,
                                              projection_matrix,
                                              glow_intensity=100.1,
                                              scattering_factor=0.8,
                                              glow_falloff=10,
                                              god_ray_intensity=100000,
                                              god_ray_decay=0.001,
                                              god_ray_sharpness=100000)

        # 5. Composite the scene and volumetric effects
        self.update_noise(delta_time)
        self.composite_scene_and_volumetrics()

        # Final check for depth and blend states
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        self.log_memory_usage()

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

    def render_scene(self, shader, player_object, world, world_objects, interactables, light_space_matrix, view_matrix,
                     projection_matrix, enable_bump_mapping=False, bump_scale=0.0):
        shader.use()
        shader.set_uniform_matrix4fv("lightSpaceMatrix", light_space_matrix)
        shader.set_uniform_bool("enableBumpMapping", enable_bump_mapping)

        # Render interactables
        for interactable in interactables:
            for mod, pos, dir in interactable.models:
                model_matrix = mod.model_matrix
                self.update_uniforms(model_matrix, view_matrix, projection_matrix, mod)

                # Set the model matrix uniform
                shader.set_uniform_matrix4fv("model", model_matrix)

                # Draw the model
                mod.draw()

        # Render player
        model_matrix = player_object.model_matrix
        if view_matrix and projection_matrix:
            self.update_uniforms(model_matrix, view_matrix, projection_matrix, player_object)
            shader.set_uniform_matrix4fv("model", model_matrix)
            player_object.draw(self.camera)

        # Render world
        # print("Render world...")
        # print("view_matrix = \n", view_matrix, "\nprojection_matrix = \n", projection_matrix)
        model_loc = glGetUniformLocation(shader.program, "model")
        view_loc = glGetUniformLocation(shader.program, "view")
        projection_loc = glGetUniformLocation(shader.program, "projection")
        impact_point_loc = glGetUniformLocation(shader.program, "impactPoint")
        crater_radius_loc = glGetUniformLocation(shader.program, "craterRadius")
        crater_depth_loc = glGetUniformLocation(shader.program, "craterDepth")

        # Set the view and projection matrices once per frame
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view_matrix))
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection_matrix))

        shader.set_bump_scale(bump_scale)
        for obj in world.get_world_objects():
            model_matrix = obj.model_matrix
            if view_matrix and projection_matrix:
                self.update_uniforms(model_matrix, view_matrix, projection_matrix, obj)
                glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model_matrix))

                # If an impact has occurred, update the impact-related uniforms
                if hasattr(obj, 'impact') and obj.impact:
                    impact_point = obj.impact_point  # Assuming this attribute is set when an impact occurs
                    crater_radius = obj.crater_radius  # Example value, set appropriately
                    crater_depth = obj.crater_depth  # Example value, set appropriately

                    glUniform3fv(impact_point_loc, 1, glm.value_ptr(impact_point))
                    glUniform1f(crater_radius_loc, crater_radius)
                    glUniform1f(crater_depth_loc, crater_depth)

                obj.draw()

        # Render world objects
        for wobj in world_objects:
            self.update_uniforms(model_matrix, view_matrix, projection_matrix, wobj)
            shader.set_uniform_matrix4fv("model", model_matrix)
            wobj.draw()



    def render_world(self, shader, player_object, world, interactables, light_space_matrix, view_matrix=None,
                     projection_matrix=None):
        shader.use()
        shader.set_uniform_matrix4fv("lightSpaceMatrix", light_space_matrix)

        # Render world objects
        for obj in world.get_world_objects():
            model_matrix = obj.model_matrix
            if view_matrix and projection_matrix:
                self.update_uniforms(model_matrix, view_matrix, projection_matrix, obj)
            shader.set_uniform_matrix4fv("model", model_matrix)
            obj.draw()

    def init_light_object_buffers(self):
        self.light_vao = glGenVertexArrays(1)
        self.light_vbo = glGenBuffers(1)
        self.light_ebo = glGenBuffers(1)
        self.light_initialized = False

    def render_light_object(self, size):
        if not hasattr(self, 'light_vao'):
            self.init_light_object_buffers()

        if not self.light_initialized:
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

            glBindVertexArray(self.light_vao)

            glBindBuffer(GL_ARRAY_BUFFER, self.light_vbo)
            glBufferData(GL_ARRAY_BUFFER, np.array(vertices, dtype=np.float32), GL_STATIC_DRAW)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.light_ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.array(indices, dtype=np.uint32), GL_STATIC_DRAW)

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)

            glBindVertexArray(0)

            self.light_initialized = True

        # Render the light object
        glBindVertexArray(self.light_vao)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def delete_light_object_buffers(self):
        if hasattr(self, 'light_vao'):
            glDeleteVertexArrays(1, [self.light_vao])
        if hasattr(self, 'light_vbo'):
            glDeleteBuffers(1, [self.light_vbo])
        if hasattr(self, 'light_ebo'):
            glDeleteBuffers(1, [self.light_ebo])

    def update_uniforms(self, model_matrix, view_matrix, projection_matrix, model: Model = None):
        self.shader.use()
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

    def init_volumetric_fbo(self):
        """Sets up a framebuffer specifically for volumetric effects,
        with both a color texture attachment and a depth renderbuffer."""
        # Generate and bind the framebuffer
        self.volumetric_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.volumetric_fbo)

        # Generate and bind the texture for color attachment
        self.volumetric_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.volumetric_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 800, 600, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.volumetric_texture, 0)

        # Generate and bind the renderbuffer for depth attachment
        self.volumetric_depth_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.volumetric_depth_buffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 800, 600)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.volumetric_depth_buffer)

        # Check if the framebuffer is complete
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            raise Exception("Framebuffer is not complete!")

        # Unbind the framebuffer to avoid accidental modification
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def render_quad(self):
        if not hasattr(self, 'quadVAO'):
            self.quadVertices = [
                # Positions        # Texture Coords
                -1.0, 1.0, 0.0, 0.0, 1.0,
                -1.0, -1.0, 0.0, 0.0, 0.0,
                1.0, -1.0, 0.0, 1.0, 0.0,
                1.0, 1.0, 0.0, 1.0, 1.0,
            ]
            self.quadIndices = [
                0, 1, 2,
                0, 2, 3
            ]

            quadVertices = np.array(self.quadVertices, dtype=np.float32)
            quadIndices = np.array(self.quadIndices, dtype=np.uint32)

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
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * quadVertices.itemsize,
                                  ctypes.c_void_p(3 * quadVertices.itemsize))

            glBindVertexArray(0)

        # Use the quad shader
        self.quad_shader.use()
        self.quad_shader.set_uniform3f("quadColor", glm.vec3(0.0, 1.0, 0.0))  # Set color to green for testing

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
        """Renders the shadow map to a depth framebuffer."""
        glViewport(0, 0, self.shadow_width, self.shadow_height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fbo)
        glClear(GL_DEPTH_BUFFER_BIT)

        self.shadow_shader.use()
        for pos in light_positions:
            light_space_matrix = self.calculate_light_space_matrix(pos)
            self.shadow_shader.set_uniform_matrix4fv("lightSpaceMatrix", light_space_matrix)

            # Set additional uniforms if needed
            self.shadow_shader.set_uniform_matrix4fv("view", self.camera.get_view_matrix())
            self.shadow_shader.set_uniform_matrix4fv("projection", self.camera.get_projection_matrix())

            # Render the world with the shadow shader
            self.render_world(self.shadow_shader, player_object, world, interactables, light_space_matrix)

        self.check_framebuffer_status()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # def setup_depth_framebuffer(self):
    #     # Generate texture
    #     self.depth_map = glGenTextures(1)
    #     glBindTexture(GL_TEXTURE_2D, self.depth_map)
    #
    #     # Define texture storage and parameters
    #     glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 800, 600, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    #     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    #     border_color = [1.0, 1.0, 1.0, 1.0]
    #     glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)
    #
    #     # Create and bind the framebuffer
    #     self.depth_map_fbo = glGenFramebuffers(1)
    #     glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fbo)
    #     glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth_map, 0)
    #     glDrawBuffer(GL_NONE)
    #     glReadBuffer(GL_NONE)
    #
    #     # Check framebuffer completeness
    #     if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
    #         print("Framebuffer not complete")
    #     glBindFramebuffer(GL_FRAMEBUFFER, 0)

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
        for obj in world.get_world_objects():
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

    def render_object(self, shader, obj, view_matrix, projection_matrix):
        model_matrix = obj.model_matrix
        if view_matrix and projection_matrix:
            self.update_uniforms(model_matrix, view_matrix, projection_matrix, obj)
        shader.set_uniform_matrix4fv("model", model_matrix)
        obj.draw()

    def render_volume_bounds(self):
        # Set line width and color for the bounding box
        glLineWidth(10.0)  # Set the line width to 2 pixels
        glColor3f(1.0, 1.0, 0.0)  # Set the color to bright yellow

        glBegin(GL_LINES)
        # Bottom face
        glVertex3f(self.volume_min.x, self.volume_min.y, self.volume_min.z)
        glVertex3f(self.volume_max.x, self.volume_min.y, self.volume_min.z)

        glVertex3f(self.volume_max.x, self.volume_min.y, self.volume_min.z)
        glVertex3f(self.volume_max.x, self.volume_min.y, self.volume_max.z)

        glVertex3f(self.volume_max.x, self.volume_min.y, self.volume_max.z)
        glVertex3f(self.volume_min.x, self.volume_min.y, self.volume_max.z)

        glVertex3f(self.volume_min.x, self.volume_min.y, self.volume_max.z)
        glVertex3f(self.volume_min.x, self.volume_min.y, self.volume_min.z)

        # Top face
        glVertex3f(self.volume_min.x, self.volume_max.y, self.volume_min.z)
        glVertex3f(self.volume_max.x, self.volume_max.y, self.volume_min.z)

        glVertex3f(self.volume_max.x, self.volume_max.y, self.volume_min.z)
        glVertex3f(self.volume_max.x, self.volume_max.y, self.volume_max.z)

        glVertex3f(self.volume_max.x, self.volume_max.y, self.volume_max.z)
        glVertex3f(self.volume_min.x, self.volume_max.y, self.volume_max.z)

        glVertex3f(self.volume_min.x, self.volume_max.y, self.volume_max.z)
        glVertex3f(self.volume_min.x, self.volume_max.y, self.volume_min.z)

        # Vertical edges
        glVertex3f(self.volume_min.x, self.volume_min.y, self.volume_min.z)
        glVertex3f(self.volume_min.x, self.volume_max.y, self.volume_min.z)

        glVertex3f(self.volume_max.x, self.volume_min.y, self.volume_min.z)
        glVertex3f(self.volume_max.x, self.volume_max.y, self.volume_min.z)

        glVertex3f(self.volume_max.x, self.volume_min.y, self.volume_max.z)
        glVertex3f(self.volume_max.x, self.volume_max.y, self.volume_max.z)

        glVertex3f(self.volume_min.x, self.volume_min.y, self.volume_max.z)
        glVertex3f(self.volume_min.x, self.volume_max.y, self.volume_max.z)

        glEnd()

    def check_opengl_error(self):
        err = glGetError()
        if err != GL_NO_ERROR:
            error_str = ""
            if err == GL_INVALID_ENUM:
                error_str = "GL_INVALID_ENUM"
            elif err == GL_INVALID_VALUE:
                error_str = "GL_INVALID_VALUE"
            elif err == GL_INVALID_OPERATION:
                error_str = "GL_INVALID_OPERATION"
            elif err == GL_STACK_OVERFLOW:
                error_str = "GL_STACK_OVERFLOW"
            elif err == GL_STACK_UNDERFLOW:
                error_str = "GL_STACK_UNDERFLOW"
            elif err == GL_OUT_OF_MEMORY:
                error_str = "GL_OUT_OF_MEMORY"
            elif err == GL_INVALID_FRAMEBUFFER_OPERATION:
                error_str = "GL_INVALID_FRAMEBUFFER_OPERATION"
            else:
                error_str = f"Unknown error code: {err}"

            print(f"OpenGL error: {error_str} (0x{err:X})")

    def setup_offscreen_framebuffer(self):
        # Setup framebuffer for the scene
        self.scene_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.scene_fbo)

        self.scene_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.scene_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 800, 600, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.scene_texture, 0)

        self.scene_rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.scene_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 800, 600)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.scene_rbo)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Scene framebuffer is not complete!")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Setup framebuffer for volumetric effects
        self.volumetric_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.volumetric_fbo)

        self.volumetric_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.volumetric_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 800, 600, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.volumetric_texture, 0)

        self.volumetric_rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.volumetric_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 800, 600)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.volumetric_rbo)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Volumetric framebuffer is not complete!")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def draw_offscreen_texture(self):
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        assert not glIsEnabled(GL_DEPTH_TEST), "GL_DEPTH_TEST should be disabled when drawing offscreen texture"
        assert not glIsEnabled(GL_BLEND), "GL_BLEND should be disabled when drawing offscreen texture"

        self.screen_shader.use()
        glBindVertexArray(self.quadVAO)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.volume_texture)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        assert glIsEnabled(GL_DEPTH_TEST), "GL_DEPTH_TEST should be re-enabled after drawing offscreen texture"
        assert glIsEnabled(GL_BLEND), "GL_BLEND should be re-enabled after drawing offscreen texture"
        print("draw_offscreen_texture complete.")
        self.check_gl_state()

    def render_scene_to_fbo(self, shader, player_object, world, world_objects, interactables, light_space_matrix,
                            view_matrix, projection_matrix, enable_bump_mapping=False, bump_scale=0.0):
        glBindFramebuffer(GL_FRAMEBUFFER, self.scene_fbo)
        glViewport(0, 0, 800, 600)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

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
        self.check_opengl_error()

        self.shader.set_roughness(0.1)

        self.render_lights(self.light_positions, self.light_colors, view_matrix, projection_matrix)
        self.render_scene(shader, player_object, world, world_objects, interactables, light_space_matrix,
                          view_matrix, projection_matrix, enable_bump_mapping, bump_scale)

        for weapon in self.weapons:
            self.draw_tracers(tracers=weapon.tracers, view_matrix=view_matrix, projection_matrix=projection_matrix)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)



    def render_volumetric_effects_to_fbo(self, view_matrix, projection_matrix, glow_intensity, scattering_factor,
                                         glow_falloff, god_ray_intensity, god_ray_decay, god_ray_sharpness):
        glBindFramebuffer(GL_FRAMEBUFFER, self.volumetric_fbo)
        glViewport(0, 0, 800, 600)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.volumetric_shader.use()

        # Correctly calculate the inverse view-projection matrix
        view_proj_matrix = projection_matrix * view_matrix
        inv_view_proj_matrix = glm.inverse(view_proj_matrix)

        self.volumetric_shader.set_uniform_matrix4fv("invViewProjMatrix", inv_view_proj_matrix)
        self.volumetric_shader.set_uniform3fv("volumeMin", self.volume_min)
        self.volumetric_shader.set_uniform3fv("volumeMax", self.volume_max)
        self.volumetric_shader.set_uniform1f("glowIntensity", glow_intensity)
        self.volumetric_shader.set_uniform1f("scatteringFactor", scattering_factor)
        self.volumetric_shader.set_uniform1f("glowFalloff", glow_falloff)
        self.volumetric_shader.set_uniform1f("godRayIntensity", god_ray_intensity)
        self.volumetric_shader.set_uniform1f("godRayDecay", god_ray_decay)
        self.volumetric_shader.set_uniform1f("godRaySharpness", god_ray_sharpness)
        self.volumetric_shader.set_uniform1f("time", self.time)

        num_lights = len(self.light_positions)
        self.volumetric_shader.set_uniform1i("numLights", num_lights)
        for i, (pos, color) in enumerate(zip(self.light_positions, self.light_colors)):
            self.volumetric_shader.set_uniform3fv(f"lightPositions[{i}]", pos)
            self.volumetric_shader.set_uniform3fv(f"lightColors[{i}]", color)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.volume_texture)
        self.check_opengl_error()

        glBindVertexArray(self.quadVAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        self.check_opengl_error()

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def composite_scene_and_volumetrics(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, 800, 600)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.composite_shader.use()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.scene_texture)
        self.composite_shader.set_uniform1i("sceneTexture", 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.volumetric_texture)
        self.composite_shader.set_uniform1i("volumetricTexture", 1)

        glBindVertexArray(self.quadVAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def check_gl_state(self):
        depth_test_enabled = glIsEnabled(GL_DEPTH_TEST)
        blend_enabled = glIsEnabled(GL_BLEND)
        depth_func = glGetIntegerv(GL_DEPTH_FUNC)
        blend_src = glGetIntegerv(GL_BLEND_SRC)
        blend_dst = glGetIntegerv(GL_BLEND_DST)

        print(f"GL_DEPTH_TEST: {'ENABLED' if depth_test_enabled else 'DISABLED'}")
        print(f"GL_BLEND: {'ENABLED' if blend_enabled else 'DISABLED'}")
        print(f"GL_DEPTH_FUNC: {depth_func}")
        print(f"GL_BLEND_SRC: {blend_src}")
        print(f"GL_BLEND_DST: {blend_dst}")

    def delete_buffers(self):
        if hasattr(self, 'scene_fbo'):
            glDeleteFramebuffers(1, [self.scene_fbo])
        if hasattr(self, 'scene_texture'):
            glDeleteTextures(1, [self.scene_texture])
        if hasattr(self, 'scene_rbo'):
            glDeleteRenderbuffers(1, [self.scene_rbo])
        if hasattr(self, 'volumetric_fbo'):
            glDeleteFramebuffers(1, [self.volumetric_fbo])
        if hasattr(self, 'volumetric_texture'):
            glDeleteTextures(1, [self.volumetric_texture])
        if hasattr(self, 'volumetric_rbo'):
            glDeleteRenderbuffers(1, [self.volumetric_rbo])
        if hasattr(self, 'quadVAO'):
            glDeleteVertexArrays(1, [self.quadVAO])
        if hasattr(self, 'quadVBO'):
            glDeleteBuffers(1, [self.quadVBO])
        # Add any other buffers or textures that need to be deleted

    def cleanup(self):
        self.delete_buffers()
        # Add other cleanup tasks

    def log_memory_usage(self):
        total_memory = glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX)
        current_memory = glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX)
        #print(f"Total GPU memory: {total_memory} KB")
        #print(f"Current available GPU memory: {current_memory} KB")
        if current_memory - total_memory > 1000000:
            print("GPU memory warning:", current_memory - total_memory, "KB used")

    def log_buffer_allocation(self, buffer_name):
        current_memory = glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX)
        print(f"Allocated {buffer_name}, current available GPU memory: {current_memory} KB")

    def render_procedural_volumetrics(self, view_matrix, projection_matrix):
        glBindFramebuffer(GL_FRAMEBUFFER, self.volumetric_fbo)
        glViewport(0, 0, 800, 600)
        glClear(GL_COLOR_BUFFER_BIT)

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.procedural_shader.use()

        # Get uniform locations
        cam_pos_location = glGetUniformLocation(self.procedural_shader.program, "camPos")
        volume_min_location = glGetUniformLocation(self.procedural_shader.program, "volumeMin")
        volume_max_location = glGetUniformLocation(self.procedural_shader.program, "volumeMax")
        num_lights_location = glGetUniformLocation(self.procedural_shader.program, "numLights")
        light_positions_location = [glGetUniformLocation(self.procedural_shader.program, f"lightPositions[{i}]") for i
                                    in
                                    range(len(self.light_positions))]
        lightColors_location = [glGetUniformLocation(self.procedural_shader.program, f"lightColors[{i}]") for i in
                                range(len(self.light_colors))]
        density_location = glGetUniformLocation(self.procedural_shader.program, "density")

        # Set uniform values
        cam_pos = glm.vec3(view_matrix[3][0], view_matrix[3][1], view_matrix[3][2])
        glUniform3fv(cam_pos_location, 1, glm.value_ptr(cam_pos))
        glUniform3fv(volume_min_location, 1, glm.value_ptr(self.volume_min))
        glUniform3fv(volume_max_location, 1, glm.value_ptr(self.volume_max))
        glUniform1i(num_lights_location, len(self.light_positions))

        print(f"Camera Position: {cam_pos}")
        print(f"Volume Min: {self.volume_min}, Volume Max: {self.volume_max}")
        print(f"Number of Lights: {len(self.light_positions)}")

        for i, pos in enumerate(self.light_positions):
            glUniform3fv(light_positions_location[i], 1, glm.value_ptr(pos))
            print(f"Light {i} Position: {pos}")
        for i, color in enumerate(self.light_colors):
            glUniform3fv(lightColors_location[i], 1, glm.value_ptr(color))
            print(f"Light {i} Color: {color}")

        glUniform1f(density_location, 0.1)  # Example density value
        print(f"Density: 0.1")

        glBindVertexArray(self.quadVAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def init_projectile_buffers(self):
        self.projectile_vao = glGenVertexArrays(1)
        self.projectile_vbo = glGenBuffers(1)

        glBindVertexArray(self.projectile_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.projectile_vbo)

        # Allocate buffer memory, 10,000 points, 3 floats per point
        glBufferData(GL_ARRAY_BUFFER, 30000 * ctypes.sizeof(ctypes.c_float), None, GL_DYNAMIC_DRAW)

        # Define the vertex data layout
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # Check for OpenGL errors during initialization
        error = glGetError()
        if error != GL_NO_ERROR:
            raise RuntimeError(f"OpenGL error during buffer initialization: {gluErrorString(error).decode()}")

    def draw_tracers(self, tracers, view_matrix, projection_matrix):
        if len(tracers) > 0:
            self.tracer_shader.use()

            # Set the tracers and trajectories using SSBOs
            #self.tracer_shader.set_tracers_uniform(tracers)

            glDisable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            #self.tracer_shader.set_uniform_matrix4fv("view", view_matrix)
            #self.tracer_shader.set_uniform_matrix4fv("projection", projection_matrix)
            #self.tracer_shader.set_uniform3fv("viewPos", self.camera.position)
            #self.tracer_shader.set_uniform3fvec("tracerColor", [1.0, 1.0, 1.0])

            num_tracers = len(tracers)
            #print("num_tracers = ", num_tracers)
            #self.tracer_shader.set_uniform1i("numTracers", num_tracers)

            # Calculate the total number of vertices
            total_vertices = sum(len(tracer) * 2 for tracer in tracers)
            #print("total vertices = ", total_vertices)
            # Bind the VAO and VBO for drawing
            glBindVertexArray(self.projectile_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.projectile_vbo)
            glLineWidth(5.0)

            # Perform the draw call
            glDrawArrays(GL_LINE_STRIP, 0, total_vertices)

            # Unbind the VAO and VBO
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            glEnable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)
            glLineWidth(1.0)

            error = glGetError()
            if error != GL_NO_ERROR:
                print(f"OpenGL error after drawing tracers: {gluErrorString(error).decode()}")

    def debug_render(self):
        # Create and bind the Vertex Array Object (VAO)
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        # Create and bind the Vertex Buffer Object (VBO)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)

        # Define the vertex data for a triangle
        vertices = np.array([
            -0.5, -0.5, 0.0,  # Vertex 1: x, y, z
            0.5, -0.5, 0.0,  # Vertex 2: x, y, z
            0.0, 0.5, 0.0  # Vertex 3: x, y, z
        ], dtype=np.float32)

        # Upload the vertex data to the GPU
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Unbind the VBO and VAO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # Set up the Shader Storage Buffer Object (SSBO) with colors for each vertex
        colors_data = np.array([
            [1.0, 0.0, 0.0, 1.0],  # Color for Vertex 1: Red
            [0.0, 1.0, 0.0, 1.0],  # Color for Vertex 2: Green
            [0.0, 0.0, 1.0, 1.0],  # Color for Vertex 3: Blue
        ], dtype=np.float32)
        self.ssbo = self.set_ssbo(0, colors_data)

        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Use the shader program
        self.tracer_shader.use()

        # Bind the SSBO
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo)

        # Bind the VAO
        glBindVertexArray(vao)

        # Draw the triangle
        total_vertices = 3  # Number of vertices to draw
        glDrawArrays(GL_TRIANGLES, 0, total_vertices)

        # Unbind the VAO
        glBindVertexArray(0)

        # Unbind the SSBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Check for and print any OpenGL errors
        error = glGetError()
        if error != GL_NO_ERROR:
            print(f"OpenGL error after drawing: {gluErrorString(error).decode()}")

    def set_ssbo(self, binding, data, usage=GL_DYNAMIC_DRAW):
        ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, usage)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, ssbo)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        print(f"SSBO Data (binding {binding}): {data.tolist()}")
        return ssbo
