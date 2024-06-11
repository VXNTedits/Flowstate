from camera import Camera
from shader import Shader
from world import World
from player import Player
from renderer import Renderer
from input_handler import InputHandler
from physics import Physics
from model import Model
from text_renderer import TextRenderer
import glm

class Components:
    def __init__(self, window):
        self.camera = Camera(glm.vec3(0.0, 0.0, 3.0), glm.vec3(0.0, 1.0, 0.0))
        self.shader = Shader('vertex_shader.glsl', 'fragment_shader.glsl')
        self.world = World('obj/world_test.obj', rotation_angles=(-90.0, 0.0, 0.0), translation=(-70.0, -2.0, 50.0))
        self.player = Player('obj/player1.obj', self.camera)
        self.models = [Model('obj/cube.obj'), self.world, self.player.model]
        self.renderer = Renderer(self.shader, self.camera)
        self.input_handler = InputHandler(self.camera, self.player)
        self.text_renderer = TextRenderer(window.width, window.height)
        self.physics = Physics(self.world, self.player)

    def set_input_callbacks(self, window):
        window.set_callbacks(self.input_handler.key_callback, self.input_handler.mouse_callback)
