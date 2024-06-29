import glm
import numpy as np
import ctypes
from OpenGL.GL import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.raw.GL.NVX.gpu_memory_info import GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, \
    GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX
from OpenGL.raw.GLU import gluErrorString
from PIL import Image
import model
from interactable import InteractableObject
from src.model import Model
from player import Player
from src.shader import Shader, ShaderManager
from src.utils.file_utils import get_relative_path
from world import World
#import noise
from opensimplex import OpenSimplex
import ctypes
from OpenGL.GL import *


class Renderer:
    """
    GL_TEXTURE0:  self.volume_texture (Empty volume texture)
    GL_TEXTURE1:  self.depth_map_fpv_texture (From the camera's perspective)
    GL_TEXTURE2:  self.volumetric_texture
    GL_TEXTURE3:  self.scene_texture
    GL_TEXTURE4:  self.tracer_texture
    GL_TEXTURE5:  self.blur_texture1
    GL_TEXTURE6:  self.blur_texture2
    GL_TEXTURE7:  self.hud_texture
    GL_TEXTURE8:  self.fog_texture_color_buffer
    GL_TEXTURE9:  self.fog_texture_depth_buffer
    GL_TEXTURE10: self.atmosphere_texture
    GL_TEXTURE11: self.atmos_color_texture
    GL_TEXTURE12: self.atmos_depth_texture

    GL_TEXTURE20
    ...
    GL_TEXTURE40: (Reserved for shadow maps)

    1. Generate a texture:               self.texture0 = glGenTextures(1)
    2. Activate a texture unit:          glActiveTexture(GL_TEXTURE0)
    3. Bind a texture to an active unit: glBindTexture(GL_TEXTURE0, self.texture0)
    """

    def __init__(self, shader, camera, physics, weapons):
        # Store the provided references
        self.physics = physics
        self.main_shader = shader
        self.camera = camera
        self.weapons = weapons

        # Initialization of light properties
        print("Initialization of light properties...")
        self.light_intensity = 1.0

        self.light_positions = [
            glm.vec3(50.2, 10.0, 2.0),
            glm.vec3(10.2, 20.0, 2.0),
            glm.vec3(0.0, 20.0, -20.0)
        ]
        # self.light_positions = [
        #     glm.vec3(50.2, -10.0, 2.0),
        #     glm.vec3(10.2, -20.0, 2.0),
        #     glm.vec3(0.0, -20.0, -20.0)
        # ]
        self.light_count = len(self.light_positions)

        self.light_colors = [
            glm.vec3(1.0, 0.07, 0.58) * self.light_intensity,  # Neon Pink
            glm.vec3(0.0, 1.0, 0.38) * self.light_intensity,  # Neon Green
            glm.vec3(0.07, 0.45, 0.9) * self.light_intensity  # Neon Blue
        ]

        self.calculate_light_space_matrices(light_positions=self.light_positions)

        # Volume bounds for volumetric rendering
        # These define the boundaries of the volume in the scene
        print("Volume bounds for volumetric rendering...")
        self.volume_min = glm.vec3(-1000, -1000, -1000)
        self.volume_max = glm.vec3(1000, 1000, 1000)

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
        self.main_shader = ShaderManager.get_shader('shaders/main_vertex.glsl',
                                                    'shaders/main_fragment.glsl')
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
                                                      "shaders/tracer_fragment.glsl")
        self.blur_shader = ShaderManager.get_shader("shaders/blur_vertex.glsl",
                                                    "shaders/blur_fragment.glsl")
        self.hud_shader = ShaderManager.get_shader("shaders/hud_vertex_shader.glsl",
                                                   "shaders/hud_fragment_shader.glsl")
        self.fog_shader = ShaderManager.get_shader("shaders/fog_vertex.glsl",
                                                   "shaders/fog_fragment.glsl")
        self.atmos_shader = ShaderManager.get_shader("shaders/atmos_vertex.glsl",
                                                     "shaders/atmos_fragment.glsl")

        # Set up the offscreen buffer for shader compositing
        print("Set up the offscreen buffer for shader compositing...")
        self.setup_offscreen_framebuffer()

        # Generate 3D noise texture for volumetric effects
        # This texture will be used in the volumetric rendering process
        print("Generate 3D noise texture for volumetric effects...")
        self.noise_size = 512  # Size of the 3D noise texture
        self.simplex = OpenSimplex(seed=np.random.randint(0, 10000))
        self.time = 0.0
        self.setup_volume_texture()

        # Setup and configure the depth framebuffer for shadow mapping
        print("Setup depth framebuffer for shadow mapping")
        self.setup_depth_framebuffer(num_lights=self.light_count)

        # Setup and configure the depth map from the camera's perspective
        self.initialize_fpv_depth_map()

        # Initialize framebuffer for volumetric rendering
        print("Initialize framebuffer for volumetric rendering...")
        self.init_volumetric_fbo()

        # Set up the vertex array and buffer objects for rendering a quad
        self.setup_quad()

        print(f"OpenGL version: {glGetString(GL_VERSION).decode()}")

        # Initialize tracer rendering system
        self.initialize_tracer_renderer(800, 600)

        # Initialize HUD
        self.hud_texture = self.load_hud(get_relative_path("res/hud.png"))
        self.hud_vao, self.hud_vbo = self.create_hud_buffers()

        # Initialize fog renderer
        self.init_atmosphere()
        self.initialize_fog()
        self.create_fog_quad()

    def setup_volume_texture(self):
        """ Create an empty 3D texture """
        self.volume_texture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
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

    def setup_depth_framebuffer(self, num_lights):
        print("Initializing depth maps for shadow rendering")
        self.depth_map_fbos = []
        self.shadow_maps = []

        texture_unit = 20
        for n in range(num_lights):
            # Create framebuffer
            fbo = glGenFramebuffers(1)
            self.depth_map_fbos.append(fbo)

            # Create depth texture
            shadow_map = glGenTextures(1)
            glActiveTexture(GL_TEXTURE0 + texture_unit)
            glBindTexture(GL_TEXTURE_2D, shadow_map)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.shadow_width, self.shadow_height, 0,
                         GL_DEPTH_COMPONENT, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
            border_color = [1.0, 1.0, 1.0, 1.0]
            glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)

            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow_map, 0)
            glDrawBuffer(GL_NONE)
            glReadBuffer(GL_NONE)
            self.check_framebuffer_status()
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            self.shadow_maps.append(shadow_map)

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

    def create_fog_quad(self):
        quad_vertices = np.array([
            # Positions    # TexCoords
            -1.0, 1.0, 0.0, 1.0,
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,

            -1.0, 1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
        ], dtype=np.float32)

        VAO = glGenVertexArrays(1)
        VBO = glGenBuffers(1)

        glBindVertexArray(VAO)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * quad_vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * quad_vertices.itemsize,
                              ctypes.c_void_p(2 * quad_vertices.itemsize))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.fog_quad_vao = VAO

    def check_framebuffer_status(self):
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"Framebuffer is not complete: {status}")

    def calculate_light_space_matrices(self, light_positions):
        near_plane = 0.1
        far_plane = 1000.0
        light_space_matrices = {}

        for i, light_position in enumerate(light_positions):
            light_proj = glm.ortho(-20.0, 20.0, -20.0, 20.0, near_plane, far_plane)
            light_view = glm.lookAt(light_position, glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
            light_space_matrix = light_proj * light_view
            light_space_matrices[i] = light_space_matrix

        self.light_space_matrices = light_space_matrices
        return self.light_space_matrices

    def render(self, player_object, world, interactables, world_objects, view_matrix, projection_matrix, delta_time):
        """Main rendering pipeline"""

        # 1. Render the shadow map to its own shadow (depth) map
        self.shadow_shader.use()
        glViewport(0, 0, self.shadow_width, self.shadow_height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.scene_fbo)
        glClear(GL_DEPTH_BUFFER_BIT)
        self.render_shadow_maps(player_object, world, interactables, self.light_positions)
        self.check_opengl_error()

        # 2. Render the scene to the self.scene_fbo framebuffer
        self.render_scene_to_fbo(shader=self.main_shader,
                                 player_object=None,
                                 world=world,
                                 interactables=interactables,
                                 world_objects=world_objects,
                                 view_matrix=view_matrix,
                                 projection_matrix=projection_matrix,
                                 enable_bump_mapping=False,
                                 bump_scale=5,
                                 weapons=self.weapons)

        # Render the player-perspective depth map to its own fbo
        self.calculate_fpv_depth_map(world, view_matrix, projection_matrix)

        # 3. Render tracers to the framebuffer
        for weapon in self.weapons:
            tracer_pos, tracer_lifetime = weapon.get_tracer_positions()
            if tracer_pos.any():
                self.draw_tracers(tracer_pos, view_matrix, projection_matrix, player_object, tracer_lifetime)
        self.check_opengl_errors()

        # 4. Render volumetric effects to the framebuffer
        self.render_volumetric_effects_to_fbo(view_matrix,
                                              projection_matrix,
                                              glow_intensity=100.0,
                                              scattering_factor=0.8,
                                              glow_falloff=100,
                                              god_ray_intensity=1,
                                              god_ray_decay=0.001,
                                              god_ray_sharpness=1)
        self.check_opengl_errors()

        # 5. Render atmosphere to the framebuffer
        self.render_atmosphere_to_fbo(view_matrix, projection_matrix, player_object.position)

        # 6. Render the player
        self.render_player_to_fbo(player_object, view_matrix, projection_matrix)

        # 6. Composite the scene and volumetric effects
        self.composite_scene_and_volumetrics()
        self.check_opengl_errors()

        # Final check for depth and blend states
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

        self.update_noise(delta_time)
        # TODO: add debug wrapper
        # Not available on all GPUs
        # self.log_memory_usage()

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

    def render_scene(self, shader, player_object, world, world_objects, interactables, view_matrix,
                     projection_matrix, enable_bump_mapping=False, bump_scale=0.0):
        shader.use()

        # Set multiple light space matrices
        for i, light_space_matrix in self.light_space_matrices.items():
            shader.set_uniform_matrix4fv(f"lightSpaceMatrix[{i}]", light_space_matrix)

        shader.set_uniform_bool("enableBumpMapping", enable_bump_mapping)

        shader.set_uniform_bool("enableBumpMapping", enable_bump_mapping)

        # Render interactables
        for interactable in interactables:
            for mod, _, _ in interactable.models:
                model_matrix = mod.model_matrix
                self.update_uniforms(model_matrix, view_matrix, projection_matrix, mod)

                # Set the model matrix uniform
                shader.set_uniform_matrix4fv("model", model_matrix)

                # Draw the model
                mod.draw()

        if player_object is not None:
            # Render player
            for player_model in player_object.get_objects():
                self.update_uniforms(player_model.model_matrix, view_matrix, projection_matrix, player_model)
                shader.set_uniform_matrix4fv("model", player_model.model_matrix)
                player_model.draw(self.camera)

        # Render world
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

    def render_world(self, shader, world, light_space_matrix, view_matrix=None,
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
        self.main_shader.use()
        self.main_shader.set_uniform_matrix4fv("model", model_matrix)
        self.main_shader.set_uniform_matrix4fv("view", view_matrix)
        self.main_shader.set_uniform_matrix4fv("projection", projection_matrix)

        if model:
            kd = model.default_material['diffuse']
            ks = model.default_material['specular']
            ns = model.default_material['shininess']
            roughness = model.default_material.get('roughness', 0.5)
            bump_scale = model.default_material.get('bumpScale', 1.0)

            self.main_shader.set_uniform3f("objectColor", glm.vec3(*kd))
            self.main_shader.set_uniform3f("specularColor", glm.vec3(*ks))
            self.main_shader.set_uniform1f("shininess", ns)
            self.main_shader.set_roughness(roughness)
            self.main_shader.set_bump_scale(bump_scale)

    def init_volumetric_fbo(self):
        """Sets up a framebuffer specifically for volumetric effects,
        with both a color texture attachment and a depth renderbuffer."""
        # Generate and bind the framebuffer
        self.volumetric_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.volumetric_fbo)
        # Generate and bind the texture for color attachment
        self.volumetric_texture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self.volumetric_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 800, 600, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.volumetric_texture, 0)
        self.check_opengl_errors()
        # Generate and bind the renderbuffer for depth attachment
        self.volumetric_depth_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.volumetric_depth_buffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 800, 600)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.volumetric_depth_buffer)
        self.check_opengl_errors()
        # Check if the framebuffer is complete
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            raise Exception("Framebuffer is not complete!")
        self.check_opengl_errors()
        # Unbind the framebuffer to avoid accidental modification
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def render_quad(self):
        if not hasattr(self, 'quadVAO'):
            quad_vertices = np.array([
                # positions   # texCoords
                -1.0, 1.0, 0.0, 1.0,
                -1.0, -1.0, 0.0, 0.0,
                1.0, -1.0, 1.0, 0.0,
                1.0, 1.0, 1.0, 1.0,
            ], dtype=np.float32)

            quad_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

            self.quadVAO = glGenVertexArrays(1)
            quadVBO = glGenBuffers(1)
            quadEBO = glGenBuffers(1)

            glBindVertexArray(self.quadVAO)

            glBindBuffer(GL_ARRAY_BUFFER, quadVBO)
            glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * quad_vertices.itemsize, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * quad_vertices.itemsize,
                                  ctypes.c_void_p(2 * quad_vertices.itemsize))

            glBindVertexArray(0)

        glBindVertexArray(self.quadVAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def init_quad(self):
        quad_vertices = np.array([
            -1.0, 1.0, 0.0, 0.0, 1.0,
            -1.0, -1.0, 0.0, 0.0, 0.0,
            1.0, -1.0, 0.0, 1.0, 0.0,
            1.0, -1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 1.0,
            -1.0, 1.0, 0.0, 0.0, 1.0,
        ], dtype=np.float32)

        self.quad_vao = glGenVertexArrays(1)
        quad_vbo = glGenBuffers(1)

        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * quad_vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * quad_vertices.itemsize, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
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

    def render_shadow_maps(self, player_object, world, interactables, light_positions):
        """Renders shadow maps for each light source."""
        glViewport(0, 0, self.shadow_width, self.shadow_height)

        for i, light_position in enumerate(light_positions):
            glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fbos[i])
            glClear(GL_DEPTH_BUFFER_BIT)

            self.shadow_shader.use()
            light_space_matrix = self.light_space_matrices[i]
            self.shadow_shader.set_uniform_matrix4fv("lightSpaceMatrix", light_space_matrix)

            # Set additional uniforms if needed
            self.shadow_shader.set_uniform_matrix4fv("view", self.camera.get_view_matrix())
            self.shadow_shader.set_uniform_matrix4fv("projection", self.camera.get_projection_matrix())

            # Render the world with the shadow shader
            self.render_world(shader=self.shadow_shader, world=world, light_space_matrix=light_space_matrix,
                              view_matrix=self.camera.get_view_matrix())

            glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def glm_to_numpy(self, mat):
        return np.array(mat, dtype=np.float32).flatten()

    def glm_to_ctypes(self, mat):
        return (ctypes.c_float * 16)(*mat.flatten())

    def calculate_fpv_depth_map(self, world, view_matrix, projection_matrix):

        # Bind the depth map framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fpv_fbo)
        glViewport(0, 0, 800, 600)
        glClear(GL_DEPTH_BUFFER_BIT)

        # Use the depth map shader for rendering the depth map from the camera's perspective
        self.depth_shader.use()

        # Set view and projection matrices
        self.depth_shader.set_uniform_matrix4fv("view", view_matrix)
        self.depth_shader.set_uniform_matrix4fv("projection", projection_matrix)

        # Render objects to the depth map
        for obj in world.get_world_objects():
            self.depth_shader.set_uniform_matrix4fv("model", obj.model_matrix)
            self.render_object(self.depth_shader, obj, view_matrix, projection_matrix)

        # Unbind the framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Reset viewport to original dimensions
        glViewport(0, 0, 800, 600)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def initialize_fpv_depth_map(self):
        self.depth_map_fpv_fbo = glGenFramebuffers(1)
        self.depth_map_fpv_texture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.depth_map_fpv_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 800, 600, 0, GL_DEPTH_COMPONENT,
                     GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)#BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)#BORDER)
        border_color = [1.0, 1.0, 1.0, 1.0]
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)

        glBindFramebuffer(GL_FRAMEBUFFER, self.depth_map_fpv_fbo)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, self.depth_map_fpv_texture, 0)#glFramebufferTexture2D((GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth_map_fpv_texture, 0))
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        self.check_framebuffer_status()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

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
        glActiveTexture(GL_TEXTURE3)
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
        glActiveTexture(GL_TEXTURE2)
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

    def render_scene_to_fbo(self, shader, player_object, world, world_objects, interactables,
                            view_matrix, projection_matrix, enable_bump_mapping=False, bump_scale=0.0, weapons=None):
        glBindFramebuffer(GL_FRAMEBUFFER, self.scene_fbo)

        glViewport(0, 0, 800, 600)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.main_shader.use()
        self.main_shader.set_uniform_matrix4fv("view", view_matrix)
        self.main_shader.set_uniform_matrix4fv("projection", projection_matrix)

        for i, (pos, color) in enumerate(zip(self.light_positions, self.light_colors)):
            self.main_shader.set_uniform_matrix4fv(f"lightSpaceMatrix[{i}]", self.light_space_matrices[i])
            self.main_shader.set_uniform3f(f"lights[{i}].position", pos)
            self.main_shader.set_uniform3f(f"lights[{i}].color", color)

            # glActiveTexture(GL_TEXTURE1)
            # glBindTexture(GL_TEXTURE_2D, self.scene_fbo)
            self.main_shader.set_uniform1i(f"shadowMap[{i}]", 1)
            self.check_opengl_error()

        self.main_shader.set_roughness(0.1)

        # Set tracer lights
        num_tracer_lights = 0
        tracer_light_positions = []
        tracer_light_colors = []
        tracer_light_intensities = []

        for weapon in weapons:
            num_tracer_lights += len(weapon.tracers)
            for tracer in weapon.tracers:
                # Retrieve the latest position for this tracer and add it to tracer_light_positions
                tracer_light_positions.append(tracer['position'])
                # Retrieve the lifetime associated with the latest position
                lifetime = tracer['lifetime']
                tracer_light_colors.append(
                    (1.0,
                     (0.5 * (glm.e() ** (-lifetime))),  #if lifetime != 0 else 0.5, # 0.5
                     (0.2 * (glm.e() ** (-lifetime)))  #if lifetime != 0 else 0.2 )# 0.2
                     ))
                # Associate the intensity of the tracer light to its lifetime
                tracer_light_intensities.append((glm.e() ** (-lifetime)))  #if lifetime != 0 else 1.0)

        shader.set_uniform1i("numTracerLights", num_tracer_lights)
        shader.set_uniform3fvec("tracerLightPositions", tracer_light_positions)
        shader.set_uniform3fvec("tracerLightColors", tracer_light_colors)
        shader.set_uniform1fv("tracerLightIntensities", tracer_light_intensities)

        shader.set_uniform3fv("fogColor", glm.vec3(1, 0.1, 0.1))
        shader.set_uniform1f("fogDensity", 0.5)
        shader.set_uniform1f("fogHeightFalloff", 0.1)

        self.render_lights(self.light_positions, self.light_colors, view_matrix, projection_matrix)
        self.render_scene(shader, player_object, world, world_objects, interactables,
                          view_matrix, projection_matrix, enable_bump_mapping, bump_scale)

        glEnable(GL_DEPTH_TEST)

        self.render_hud()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def render_volumetric_effects_to_fbo(self, view_matrix, projection_matrix, glow_intensity, scattering_factor,
                                         glow_falloff, god_ray_intensity, god_ray_decay, god_ray_sharpness):
        glBindFramebuffer(GL_FRAMEBUFFER, self.volumetric_fbo)
        glViewport(0, 0, 800, 600)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #glDisable(GL_DEPTH_TEST)
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
        # 1. Bind the default framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, 800, 600)  # Set the viewport to the window size
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear buffers
        glEnable(GL_DEPTH_TEST)

        # 2. Use the composite shader program
        self.composite_shader.use()

        # 3. Pass depth information uniforms
        glActiveTexture(GL_TEXTURE12)
        glBindTexture(GL_TEXTURE_2D, self.depth_map_fpv_texture)
        self.composite_shader.set_uniform1i("depthTexture", 12)
        self.composite_shader.set_uniform1f("nearPlane", 0.1)
        self.composite_shader.set_uniform1f("farPlane", 10000)

        # 4. Bind the scene texture
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, self.scene_texture)
        self.composite_shader.set_uniform1i("sceneTexture", 3)
        self.check_opengl_errors()

        # 5. Bind the volumetrics texture
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, self.volumetric_texture)
        self.composite_shader.set_uniform1i("volumetricTexture", 2)
        self.check_opengl_errors()

        # 6. Bind the tracers texture
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D, self.tracer_texture)
        self.composite_shader.set_uniform1i("tracers", 4)
        self.check_opengl_errors()

        # 7. Bind the atmosphere texture
        glActiveTexture(GL_TEXTURE10)
        glBindTexture(GL_TEXTURE_3D, self.atmosphere_texture)
        self.composite_shader.set_uniform1i("atmosphere", 10)
        self.check_opengl_errors()

        # 8. Render the quad for compositing
        self.render_quad()
        self.check_opengl_errors()

        # 9. Unbind all textures
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindTexture(GL_TEXTURE_3D, 0)
        self.check_opengl_errors()

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

    def draw_tracers(self, tracer_positions, view_matrix, projection_matrix, player, lifetimes):
        # Update the VBO with new tracer positions

        glBindBuffer(GL_ARRAY_BUFFER, self.tracer_vbo)
        glBufferData(GL_ARRAY_BUFFER, tracer_positions.nbytes, tracer_positions, GL_DYNAMIC_DRAW)

        # Bind framebuffer and render tracers
        glBindFramebuffer(GL_FRAMEBUFFER, self.tracer_fbo)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.tracer_shader.use()
        self.tracer_shader.set_uniform3f("lightPos", glm.vec3(tracer_positions[-1]))
        self.tracer_shader.set_uniform1f("lightIntensity", 10.0)
        self.tracer_shader.set_uniform_matrix4fv("model", glm.mat4(1))
        self.tracer_shader.set_uniform_matrix4fv("view", view_matrix)
        self.tracer_shader.set_uniform3f("viewPos", glm.vec3(player.pitch, player.yaw, 0.0))
        self.tracer_shader.set_uniform_matrix4fv("projection", projection_matrix)
        self.tracer_shader.set_uniform3f("tracerColor", glm.vec3(
            1.0,
            glm.exp(-lifetimes[-1] * 50),  # Exponential decay TODO: Tweak
            glm.exp(-lifetimes[-1] * 20)  # Exponential decay
        ))
        self.tracer_shader.set_uniform3f("lightColor", glm.vec3(
            1.0,
            glm.exp(-lifetimes[-1] * 2),  # Exponential decay
            glm.exp(-lifetimes[-1] * 5)  # Exponential decay
        ))
        glBindVertexArray(self.tracer_vao)
        glDrawArrays(GL_LINE_STRIP, 0, len(tracer_positions) // 3)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Debugging: Check if tracer framebuffer is complete
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print("Tracer framebuffer not complete:", status)

        # Apply horizontal blur
        glBindFramebuffer(GL_FRAMEBUFFER, self.blur_fbo1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.blur_shader.use()
        self.blur_shader.set_uniform_sampler2D("image", 0)
        self.blur_shader.set_uniform_bool("horizontal", True)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tracer_texture)
        self.render_quad()  # Render quad here for horizontal blur
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Debugging: Check if first blur framebuffer is complete
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print("First blur framebuffer not complete:", status)

        # Apply vertical blur
        glBindFramebuffer(GL_FRAMEBUFFER, self.blur_fbo2)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.blur_shader.set_uniform_bool("horizontal", False)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.blur_texture1)  # Use the result from the first blur pass
        self.render_quad()  # Render quad here for vertical blur
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Debugging: Check if second blur framebuffer is complete
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print("Second blur framebuffer not complete:", status)

        # Optionally: Use the final blurred texture for further rendering or post-processing
        glBindTexture(GL_TEXTURE_2D, self.blur_texture2)
        # Perform any additional rendering or post-processing with the final blurred texture

    def check_opengl_errors(self):
        error = glGetError()
        if error != GL_NO_ERROR:
            print(f"OpenGL error: {gluErrorString(error).decode()}")

    def set_ssbo(self, binding, data, usage=GL_DYNAMIC_DRAW):
        ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, usage)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, ssbo)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        print(f"SSBO Data (binding {binding}): {data.tolist()}")
        self.check_opengl_errors()
        return ssbo

    def initialize_tracer_renderer(self, screen_width, screen_height):
        """Function to initialize the tracer renderer"""
        # Create vbo and vao for tracers
        vbo = glGenBuffers(1)
        vao = glGenVertexArrays(1)

        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)  # Initialize with no data

        # Vertex position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Framebuffer for rendering tracers
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)

        # Texture to store framebuffer data
        self.tracer_texture = glGenTextures(1)
        print("tracer texture", self.tracer_texture)
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D, self.tracer_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, screen_width, screen_height, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.tracer_texture, 0)

        # Check framebuffer completeness
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Framebuffer is not complete!")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self.tracer_vao = vao
        self.tracer_vbo = vbo
        self.tracer_fbo = fbo

        # Create and configure blur FBOs and textures
        self.blur_fbo1 = glGenFramebuffers(1)
        self.blur_texture1 = glGenTextures(1)
        glActiveTexture(GL_TEXTURE5)
        self.configure_blur_fbo(self.blur_fbo1, self.blur_texture1, screen_width, screen_height)

        self.blur_fbo2 = glGenFramebuffers(1)
        self.blur_texture2 = glGenTextures(1)
        glActiveTexture(GL_TEXTURE6)
        self.configure_blur_fbo(self.blur_fbo2, self.blur_texture2, screen_width, screen_height)

        print(
            f"Tracer renderer initialized. \n VAO = {self.tracer_vao} \n VBO = {self.tracer_vbo} \n FBO = {self.tracer_fbo}")

    def configure_blur_fbo(self, fbo, texture, width, height):
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Blur Framebuffer is not complete!")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def load_hud(self, filename):
        image = Image.open(filename).convert("RGBA")
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = image.tobytes()

        self.hud_texture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE7)
        glBindTexture(GL_TEXTURE_2D, self.hud_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glBindTexture(GL_TEXTURE_2D, 0)

        return self.hud_texture

    def render_hud(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.hud_shader.use()

        glActiveTexture(GL_TEXTURE0)  # Ensure the correct texture unit is active
        glBindTexture(GL_TEXTURE_2D, self.hud_texture)  # Bind the HUD texture

        projection = self.np_ortho(0, 800, 0, 600, -1, 1)
        self.hud_shader.set_uniform_matrix4fv('projection', projection)
        self.hud_shader.set_uniform_sampler2D('overlayTexture', 0)  # Set to texture unit 0

        glBindVertexArray(self.hud_vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def np_ortho(self, left, right, bottom, top, near, far):
        return np.array([
            [2.0 / (right - left), 0, 0, 0],
            [0, 2.0 / (top - bottom), 0, 0],
            [0, 0, -2.0 / (far - near), 0],
            [-(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1]
        ], dtype=np.float32)

    def glm_ortho(self, left, right, bottom, top, near, far):
        return glm.mat4(
            [2.0 / (right - left), 0, 0, 0],
            [0, 2.0 / (top - bottom), 0, 0],
            [0, 0, -2.0 / (far - near), 0],
            [-(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1])

    def create_hud_buffers(self):
        vertices = np.array([
            # Positions    # TexCoords
            0.0, 0.0, 0.0, 0.0,
            800.0, 0.0, 1.0, 0.0,
            800.0, 600.0, 1.0, 1.0,
            0.0, 600.0, 0.0, 1.0
        ], dtype=np.float32)

        indices = np.array([
            0, 1, 2,
            2, 3, 0
        ], dtype=np.uint32)

        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)

        glBindVertexArray(vao)

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Position attribute
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # TexCoord attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(2 * vertices.itemsize))
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        return vao, vbo


    def initialize_fog(self):
        # Create a framebuffer and bind it
        self.fog_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fog_fbo)

        # Create a color attachment texture
        self.fog_texture_color_buffer = glGenTextures(1)
        glActiveTexture(GL_TEXTURE8)
        glBindTexture(GL_TEXTURE_2D, self.fog_texture_color_buffer)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 800, 600, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.fog_texture_color_buffer, 0)

        # Create a depth attachment texture
        self.fog_texture_depth_buffer = glGenTextures(1)
        glActiveTexture(GL_TEXTURE9)
        glBindTexture(GL_TEXTURE_2D, self.fog_texture_depth_buffer)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 800, 600, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.fog_texture_depth_buffer, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Framebuffer is not complete!")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def init_atmosphere(self, tex_width=128, tex_height=128, tex_depth=128, density_falloff=5.0):
        """ Generates the 3D volume and initializes the associated VAO, VBO, and EBO used to render the atmosphere """
        print("Initializing atmosphere texture...")

        # Generate a texture ID
        self.atmosphere_texture = glGenTextures(1)
        glActiveTexture(GL_TEXTURE10)
        # Bind the texture
        glBindTexture(GL_TEXTURE_3D, self.atmosphere_texture)

        # Define the starting and ending values
        start_values = np.array([0, 0, 0, 0], dtype=np.float32)
        end_values = np.array([0.05, 0.07, 0.1, 1.0], dtype=np.float32)

        # Calculate the differences
        diff_values = end_values - start_values

        # Calculate the center of the texture volume
        center_x = (tex_width - 1) / 2.0
        center_y = (tex_height - 1) / 2.0
        center_z = (tex_depth - 1) / 2.0

        # Generate grid coordinates in the range [-1, 1]
        x = np.linspace(-center_x, center_x, tex_width) / center_x
        y = np.linspace(-center_y, center_y, tex_height) / center_y
        z = np.linspace(-center_z, center_z, tex_depth) / center_z
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

        # Calculate the distance from the center for each voxel
        distances = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)

        # Normalize the distances to be in the range [0, 1]
        normalized_distances = distances / np.sqrt(3)

        # Apply density falloff to the distances and ensure it forms a sphere within the cube
        atmosphere_data = start_values + (1 - (normalized_distances ** density_falloff))[..., np.newaxis] * diff_values

        # Clamp values to [start_values, end_values] range
        atmosphere_data = np.clip(atmosphere_data, start_values, end_values)

        # Set texture parameters and upload data
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        self.check_opengl_error()

        # Correct the internal format and data type
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, tex_width, tex_height, tex_depth, 0, GL_RGBA, GL_FLOAT,
                     atmosphere_data)
        self.check_gl_state()  # Check for OpenGL errors
        self.check_opengl_error()

        # Generate and bind a dedicated framebuffer and textures
        self.atmos_fbo = glGenFramebuffers(1)
        self.atmos_color_texture = glGenTextures(1)
        self.atmos_depth_texture = glGenTextures(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self.atmos_fbo)

        # Create color texture
        glActiveTexture(GL_TEXTURE11)
        glBindTexture(GL_TEXTURE_2D, self.atmos_color_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_width, tex_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.atmos_color_texture, 0)
        self.check_opengl_error()

        # Set the border color to white
        border_color = [1.0, 0.0, 0.0, 1.0]
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)

        # Note: Using depth map texture provided by self.depth_map_fpv_texture
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth_map_fpv_texture, 0)
        self.check_opengl_error()

        # Check framebuffer completeness
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Framebuffer is not complete")
        # Unbind
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Each group of three lines corresponds to two triangles that form one face of the cube.
        vertices, indices = self.atmos_volume()

        atmos_vao = glGenVertexArrays(1)
        atmos_vbo = glGenBuffers(1)
        atmos_ebo = glGenBuffers(1)

        glBindVertexArray(atmos_vao)

        glBindBuffer(GL_ARRAY_BUFFER, atmos_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, atmos_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Unbind VAO and buffers
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.atmos_vao = atmos_vao
        self.atmos_vbo = atmos_vbo
        self.atmos_ebo = atmos_ebo


    def render_atmosphere_to_fbo(self, view_matrix, projection_matrix, camera_pos):
        # Ensure atmosphere framebuffer is initialized
        if not hasattr(self, 'atmos_fbo'):
            self.init_atmosphere()

        # Bind the atmosphere framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.atmos_fbo)
        glViewport(0, 0, 800, 600)  # Ensure viewport matches the FBO size
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set depth and blend settings
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Use the atmosphere shader program
        self.atmos_shader.use()

        # Bind uniforms: vertex
        self.atmos_shader.set_uniform_matrix4fv("model", glm.mat4(1))
        self.atmos_shader.set_uniform_matrix4fv("view", view_matrix)
        self.atmos_shader.set_uniform_matrix4fv("projection", projection_matrix)

        # Bind the 3D texture for atmospheric effects
        glActiveTexture(GL_TEXTURE0 + 10)
        glBindTexture(GL_TEXTURE_3D, self.atmosphere_texture)
        self.atmos_shader.set_uniform1i("fogTexture", 10)

        # Bind uniforms: fragment
        self.atmos_shader.set_uniform3fv("lightPosition", glm.vec3(1000.0, 0.0, 1000.0))
        self.atmos_shader.set_uniform3fv("lightColor", glm.vec3(1.0, 1.0, 1.0))
        self.atmos_shader.set_uniform1f("lightIntensity", 1.0)
        self.atmos_shader.set_uniform3fv("cameraPosition", camera_pos)
        self.atmos_shader.set_uniform1f("scatteringCoefficient", 0.01)
        # self.atmos_shader.set_uniform_matrix4fv("viewMatrix", view_matrix)
        # self.atmos_shader.set_uniform_matrix4fv("projectionMatrix", projection_matrix)

        # Draw the atmosphere
        glBindVertexArray(self.atmos_vao)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.atmos_ebo)
        glDrawElements(GL_TRIANGLES, self.atmosphere_indices.size, GL_UNSIGNED_INT, None)

        # Unbind resources
        glBindVertexArray(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_3D, 0)

        # Check for OpenGL errors
        self.check_opengl_error()

    def atmos_volume(self, size=512.0, position=(0.0, 0.0, 0.0)):
        half_size = size / 2.0
        px, py, pz = position

        self.atmosphere_vertices = np.array([
            # Front face
            px - half_size, py - half_size, pz + half_size,  # Bottom-left
            px + half_size, py - half_size, pz + half_size,  # Bottom-right
            px + half_size, py + half_size, pz + half_size,  # Top-right
            px - half_size, py - half_size, pz + half_size,  # Bottom-left
            px + half_size, py + half_size, pz + half_size,  # Top-right
            px - half_size, py + half_size, pz + half_size,  # Top-left

            # Back face
            px - half_size, py - half_size, pz - half_size,  # Bottom-left
            px - half_size, py + half_size, pz - half_size,  # Top-left
            px + half_size, py + half_size, pz - half_size,  # Top-right
            px - half_size, py - half_size, pz - half_size,  # Bottom-left
            px + half_size, py + half_size, pz - half_size,  # Top-right
            px + half_size, py - half_size, pz - half_size,  # Bottom-right

            # Top face
            px - half_size, py + half_size, pz - half_size,  # Top-left
            px - half_size, py + half_size, pz + half_size,  # Bottom-left
            px + half_size, py + half_size, pz + half_size,  # Bottom-right
            px - half_size, py + half_size, pz - half_size,  # Top-left
            px + half_size, py + half_size, pz + half_size,  # Bottom-right
            px + half_size, py + half_size, pz - half_size,  # Top-right

            # Bottom face
            px - half_size, py - half_size, pz - half_size,  # Top-left
            px + half_size, py - half_size, pz - half_size,  # Top-right
            px + half_size, py - half_size, pz + half_size,  # Bottom-right
            px - half_size, py - half_size, pz - half_size,  # Top-left
            px + half_size, py - half_size, pz + half_size,  # Bottom-right
            px - half_size, py - half_size, pz + half_size,  # Bottom-left

            # Right face
            px + half_size, py - half_size, pz - half_size,  # Bottom-left
            px + half_size, py + half_size, pz - half_size,  # Top-left
            px + half_size, py + half_size, pz + half_size,  # Top-right
            px + half_size, py - half_size, pz - half_size,  # Bottom-left
            px + half_size, py + half_size, pz + half_size,  # Top-right
            px + half_size, py - half_size, pz + half_size,  # Bottom-right

            # Left face
            px - half_size, py - half_size, pz - half_size,  # Bottom-left
            px - half_size, py - half_size, pz + half_size,  # Bottom-right
            px - half_size, py + half_size, pz + half_size,  # Top-right
            px - half_size, py - half_size, pz - half_size,  # Bottom-left
            px - half_size, py + half_size, pz + half_size,  # Top-right
            px - half_size, py + half_size, pz - half_size  # Top-left
        ], dtype=np.float32).flatten()

        self.atmosphere_indices = np.array([
            0, 1, 2, 2, 3, 0,  # Front face
            4, 5, 6, 6, 7, 4,  # Back face
            8, 9, 10, 10, 11, 8,  # Top face
            12, 13, 14, 14, 15, 12,  # Bottom face
            16, 17, 18, 18, 19, 16,  # Right face
            20, 21, 22, 22, 23, 20  # Left face
        ], dtype=np.uint32)

        return self.atmosphere_vertices, self.atmosphere_indices

    def render_player_to_fbo(self, player_object, view_matrix, projection_matrix):
        glBindFramebuffer(GL_FRAMEBUFFER, self.scene_fbo)
        # glDisable(GL_DEPTH_TEST)
        # Render player
        for player_model in player_object.get_objects():
            self.update_uniforms(player_model.model_matrix, view_matrix, projection_matrix, player_model)
            self.main_shader.set_uniform_matrix4fv("model", player_model.model_matrix)
            player_model.draw(self.camera)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        # glEnable(GL_DEPTH_TEST)
