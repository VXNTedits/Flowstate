import glm

from window import Window
from components import Components
import glfw
class Game:
    def __init__(self, fullscreen=False):
        self.window = Window(800, 600, "3D Game", fullscreen)
        self.components = Components(self.window)
        self.components.set_input_callbacks(self.window)

    def run(self):
        last_frame_time = glfw.get_time()

        while not self.window.should_close():
            self.window.poll_events()

            current_frame_time = glfw.get_time()
            delta_time = current_frame_time - last_frame_time
            last_frame_time = current_frame_time

            # Update and render logic
            self.components.input_handler.update(delta_time)
            self.components.player.update(delta_time)
            #self.components.physics.update(delta_time)

            view_matrix = self.components.camera.get_view_matrix()
            projection_matrix = glm.perspective(glm.radians(45.0), self.window.width / self.window.height, 0.1, 100.0)

            self.components.renderer.render(self.components.world, view_matrix, projection_matrix)
            for model in self.components.models:
                self.components.renderer.render(model, view_matrix, projection_matrix)

            self.components.text_renderer.render_text("FPS: {}".format(int(1.0 / delta_time)), 10, 10)

            self.window.swap_buffers()

        glfw.terminate()

