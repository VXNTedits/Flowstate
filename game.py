from typing import List

import glm
from OpenGL.GL import *
from window import Window
from components import Components
import glfw
import time


class Game:
    def __init__(self, fullscreen=False):
        self.window = Window(800, 600, "3D Game", fullscreen)
        print('Window initialized')
        self.components = Components(self.window)
        print('Components initialized')
        self.components.set_input_callbacks(self.window)
        print('Inputs initialized')
        self.projection_matrix = glm.perspective(glm.radians(90.0), self.window.width / self.window.height, 0.001,
                                                 10000.0)
        self.tick_rate = 1.0 / 144.0

    def run(self):
        last_frame_time = glfw.get_time()
        accumulator = 0.0

        while not self.window.should_close():
            self.window.poll_events()

            current_frame_time = glfw.get_time()
            delta_time = current_frame_time - last_frame_time
            last_frame_time = current_frame_time
            accumulator += delta_time

            # Update game logic at fixed intervals
            while accumulator >= self.tick_rate:
                self.components.input_handler.process_input(self.components.player, self.tick_rate)
                self.components.player.update_player(self.tick_rate)
                self.components.physics.update_physics(self.tick_rate)
                self.components.update_components(self.tick_rate)
                accumulator -= self.tick_rate

            # Rendering
            view_matrix = self.components.camera.get_view_matrix()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear the screen

            self.components.renderer.render(self.components.player, self.components.world, self.components.interactables, view_matrix, self.projection_matrix, delta_time)

            self.window.swap_buffers()

        glfw.terminate()
