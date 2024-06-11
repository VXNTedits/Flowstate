from camera import Camera
from shader import Shader
from player import Player
from renderer import Renderer
from input_handler import InputHandler
from physics import Physics
from text_renderer import TextRenderer
import glm
from world import World

class Components:
    def __init__(self, window):
        self.camera = Camera(glm.vec3(0.0, 0.0, 3.0), glm.vec3(0.0, 1.0, 0.0))
        self.shader = Shader('vertex_shader.glsl', 'fragment_shader.glsl')

        # Define filepaths, rotations, translations, and material overrides for multiple objects
        filepaths =          ['obj/world_test.obj', 'obj/donut.obj', 'obj/cube.obj']
        mtl_filepaths =      ['obj/world_test.mtl', 'obj/donut.mtl', 'obj/cube.mtl']
        rotations =          [(-90.0, 0.0, 0.0)   , (45.0, 45.0, 45.0), (-90.0, 0.0, 0.0)]
        translations =       [(-70.0, -2.0, 50.0) , (-30.0, 0.5, -40.0), (-70.0, -10.0, 50.0)]
        material_overrides = [(None, glm.vec3(1,1,1), 10) , (None, glm.vec3(1,0.1,1), 1000), (None, glm.vec3(1,1,1), 64.0)]

        self.world = World(filepaths, mtl_filepaths, rotations, translations, material_overrides)
        self.player = Player('obj/player1.obj', 'obj/player1.mtl', self.camera)

        self.models = [self.world] + self.world.objects + [self.player.model]
        self.renderer = Renderer(self.shader, self.camera)
        self.input_handler = InputHandler(self.camera, self.player)
        self.text_renderer = TextRenderer(window.width, window.height)
        self.physics = Physics(self.world, self.player)

    def set_input_callbacks(self, window):
        window.set_callbacks(self.input_handler.key_callback, self.input_handler.mouse_callback)
