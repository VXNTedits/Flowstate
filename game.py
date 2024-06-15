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
                                                 1000.0)
        self.tick_rate = 1.0 / 100.0

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
                self.components.input_handler.update_inputs(self.tick_rate)
                self.components.player.update_player(self.tick_rate)
                self.components.physics.update_physics(self.tick_rate)
                self.components.update_components(self.tick_rate)
                for item in self.components.interactables:
                    item.update(player=self.components.player, delta_time=self.tick_rate)
                accumulator -= self.tick_rate

            # Rendering
            view_matrix = self.components.camera.get_view_matrix()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear the screen

            # Render the player
            self.components.renderer.render_player(self.components.player, view_matrix, self.projection_matrix)
            # Render the world components
            for model in self.components.world_objects.models:
                self.components.renderer.render_world(model, view_matrix, self.projection_matrix)

            # Render the world
            for model in self.components.world.get_objects():
                self.components.renderer.render_world(model, view_matrix, self.projection_matrix)

            #self.components.renderer.render_aabb(self.components.world.get_objects(), self.components.player.position,
            #                                     view_matrix, self.projection_matrix)
            # Render interactables
            self.components.renderer.render_interactables(self.components.interactables, view_matrix,
                                                          self.projection_matrix)

            self.window.swap_buffers()

        glfw.terminate()
