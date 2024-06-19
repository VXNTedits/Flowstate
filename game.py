import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
import glm
from enum import Enum, auto

from components import Components
from window import Window


class GameState(Enum):
    MAIN_MENU = auto()
    GAMEPLAY = auto()

class Game:
    def __init__(self, fullscreen=False):
        self.window = Window(800, 600, "3D Game", fullscreen)
        print('Window initialized')
        self.components = Components(self.window)
        self.projection_matrix = glm.perspective(glm.radians(90.0), self.window.width / self.window.height, 0.001, 10000.0)
        self.tick_rate = 1.0 / 144.0
        self.state = GameState.MAIN_MENU

        # Initialize ImGui
        imgui.create_context()
        self.impl = GlfwRenderer(self.window.window)

    def run(self):
        last_frame_time = glfw.get_time()
        accumulator = 0.0

        while not self.window.should_close():
            self.window.poll_events()
            self.impl.process_inputs()

            current_frame_time = glfw.get_time()
            delta_time = current_frame_time - last_frame_time
            last_frame_time = current_frame_time
            accumulator += delta_time

            # Render the UI
            imgui.new_frame()

            if self.state == GameState.MAIN_MENU:
                self.render_main_menu()
            elif self.state == GameState.GAMEPLAY:
                self.update_gameplay(accumulator, delta_time)
                accumulator = 0.0  # Reset accumulator after updating gameplay

            imgui.render()
            self.impl.render(imgui.get_draw_data())

            self.window.swap_buffers()

        self.impl.shutdown()
        glfw.terminate()

    def render_main_menu(self):
        imgui.begin("Main Menu")
        if imgui.button("Start Game"):
            self.state = GameState.GAMEPLAY
            self.components.initialize_gameplay_components()
            self.components.set_input_callbacks()
        if imgui.button("Exit"):
            glfw.set_window_should_close(self.window.window, True)
        if imgui.button("Settings"):
            print("TODO: Implement settings")
        imgui.end()

    def update_gameplay(self, accumulator, delta_time):
        # Update game logic at fixed intervals
        while accumulator >= self.tick_rate:
            self.components.input_handler.process_input(self.components.player, self.tick_rate)
            self.components.player.update_player(self.tick_rate)
            self.components.physics.update_physics(self.tick_rate)
            self.components.update_components(self.tick_rate)
            accumulator -= self.tick_rate

        # Rendering
        view_matrix = self.components.camera.get_view_matrix()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  # Clear the screen

        self.components.renderer.render(self.components.player, self.components.world, self.components.interactables, view_matrix, self.projection_matrix, delta_time)
