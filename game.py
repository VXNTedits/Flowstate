import os
import glfw
import glm
from player import Player
from world import World
from model import Model
from renderer import Renderer
from input_handler import InputHandler
from text_renderer import TextRenderer
from physics import Physics
from camera import Camera
from shader import Shader
from OpenGL.GL import *


class Game:
    def __init__(self):
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        self.window = glfw.create_window(800, 600, "3D Game", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")
        glfw.make_context_current(self.window)

        self.camera = Camera(glm.vec3(0.0, 0.0, 3.0), glm.vec3(0.0, 1.0, 0.0))
        self.shader = Shader('vertex_shader.glsl', 'fragment_shader.glsl')

        # Load the world model
        self.world = World(os.path.join('obj', 'world1.obj'))

        # Load the player model
        self.player = Player(os.path.join('obj', 'player1.obj'), self.camera)

        # Load other models
        self.models = [
            #Model(os.path.join('obj', 'cube.obj')),
            #Model(os.path.join('obj', 'another_model.obj'))  # Add more models as needed
        ]

        self.renderer = Renderer(self.shader, self.camera)
        self.input_handler = InputHandler(self.camera, self.player)

        self.text_renderer = TextRenderer(800, 600)  # Initialize TextRenderer with window size

        glfw.set_key_callback(self.window, self.input_handler.key_callback)
        glfw.set_cursor_pos_callback(self.window, self.input_handler.mouse_callback)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    def run(self):
        last_frame = 0.0
        while not glfw.window_should_close(self.window):
            current_frame = glfw.get_time()
            delta_time = current_frame - last_frame
            last_frame = current_frame

            self.input_handler.process_input(delta_time)

            # Clear the screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Render the world without rotation and laying flat
            self.renderer.render(self.world, self.player.position, rotate=False, is_world=True)

            # Render the player model
            self.renderer.render(self.player.model, self.player.position, rotate=False)

            # Render other models with rotation
            for model in self.models:
                self.renderer.render(model, self.player.position, rotate=True)

            # Render player coordinates text overlay
            coordinates_text = f"Player Position: ({self.player.position.x:.2f}, {self.player.position.y:.2f}, {self.player.position.z:.2f})"
            self.text_renderer.render_text(coordinates_text, 10, 10)

            glfw.swap_buffers(self.window)
            glfw.poll_events()
        glfw.terminate()
