from OpenGL.GL import *
import glfw
import glm
import sys
from camera import Camera
from shader import Shader
from world import World
from player import Player
from renderer import Renderer
from input_handler import InputHandler
from physics import Physics
from model import Model
from text_renderer import TextRenderer

class Game:
    def __init__(self, fullscreen=False):
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        # Get primary monitor
        primary_monitor = glfw.get_primary_monitor()
        if not primary_monitor:
            raise Exception("Failed to get primary monitor")

        if fullscreen:
            # Get video mode of the primary monitor (native resolution and refresh rate)
            video_mode = glfw.get_video_mode(primary_monitor)
            if not video_mode:
                raise Exception("Failed to get video mode for primary monitor")

            # Set window to fullscreen with native resolution
            self.window = glfw.create_window(video_mode.size.width, video_mode.size.height, "3D Game", primary_monitor, None)
            self.window_width = video_mode.size.width
            self.window_height = video_mode.size.height
        else:
            # Set window to a smaller fixed size
            self.window = glfw.create_window(800, 600, "3D Game", None, None)
            self.window_width = 800
            self.window_height = 600

        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        glfw.make_context_current(self.window)

        self.camera = Camera(glm.vec3(0.0, 0.0, 3.0), glm.vec3(0.0, 1.0, 0.0))
        self.shader = Shader('vertex_shader.glsl', 'fragment_shader.glsl')

        # Load the world model
        self.world = World('obj/world1.obj', rotation_angles=(-90.0, 0.0, 0.0), translation=(-70.0, -2.0, 50.0))

        # Load the player model
        self.player = Player('obj/player1.obj', self.camera)

        # Load other models
        self.models = [
            Model('obj/cube.obj'),
            # Model('obj/another_model.obj')  # Commented out for now
        ]

        self.renderer = Renderer(self.shader, self.camera)
        self.input_handler = InputHandler(self.camera, self.player)

        self.text_renderer = TextRenderer(self.window_width, self.window_height)  # Initialize TextRenderer with window size

        # Initialize physics
        self.physics = Physics(self.world, self.player)

        # Add colliders for the models in the scene
        for model in self.models:
            surfaces = model.get_surfaces()
            for surface in surfaces:
                self.physics.add_collider(surface)

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
            self.physics.apply_gravity(self.player, delta_time)
            # Check for collisions
            self.physics.check_collision(self.player, self.world)

            # Clear the screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Toggle camera view mode
            if self.camera.first_person:
                self.camera.set_first_person(self.player.position)
                self.camera.set_position(self.player.position)  # Update the player's position to match the camera's position in first-person view
                player_visible = False
            else:
                self.camera.set_third_person(self.player.position)
                player_visible = True

            # Compute view and projection matrices
            fov = 90.0
            near_clip = 0.1
            far_clip = 10000.0
            projection = glm.perspective(glm.radians(fov), self.window_width / self.window_height, near_clip, far_clip)
            view = self.camera.get_view_matrix()

            # Render the world without rotation and laying flat
            self.renderer.render(self.world, view, projection)

            # Render the player model only if in third-person view
            if player_visible:
                self.renderer.render(self.player.model, view, projection)

            # Render other models with rotation
            for model in self.models:
                self.renderer.render(model, view, projection)

            # Render bounding boxes for collision detection using the actual geometry of the models
            self.renderer.draw_model_bounding_box(self.world, view, projection)
            if player_visible:
                self.renderer.draw_model_bounding_box(self.player.model, view, projection)
            for model in self.models:
                self.renderer.draw_model_bounding_box(model, view, projection)

            # Render player coordinates text overlay
            coordinates_text = f"Player Position: ({self.player.position.x:.2f}, {self.player.position.y:.2f}, {self.player.position.z:.2f})"
            self.text_renderer.render_text(coordinates_text, 10, 10)

            # Print player and camera positions, and view mode
            view_mode = "First-Person" if self.camera.first_person else "Third-Person"
            sys.stdout.write(f"\rPlayer position: {self.player.position} Camera position: {self.camera.position} View mode: {view_mode}")
            sys.stdout.flush()

            glfw.swap_buffers(self.window)
            glfw.poll_events()
        glfw.terminate()
