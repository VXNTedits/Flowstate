# components.py
from camera import Camera
from model import Model
from shader import Shader
from player import Player
from renderer import Renderer
from input_handler import InputHandler
from physics import Physics
from text_renderer import TextRenderer
import glm
from world import World, MaterialOverride


class ObjectAttributes:
    def __init__(self, filepath, mtl_filepath, rotation, translation, material_override, scale=None):
        self.scale = scale
        self.filepath = filepath
        self.mtl_filepath = mtl_filepath
        self.rotation = rotation
        self.translation = translation
        self.material_override = material_override


class Components:
    def __init__(self, window):
        self.camera = Camera(glm.vec3(0.0, 0.0, 3.0), glm.vec3(0.0, 1.0, 0.0))
        self.shader = Shader('vertex_shader.glsl', 'fragment_shader.glsl')

        # Define object attributes for multiple objects
        object_attributes = [
            ObjectAttributes(
                'obj/world_test.obj', 'obj/world_test.mtl',
                (-90.0, 0.0, 0.0), (-70.0, -100.0, 50.0),
                MaterialOverride(None, glm.vec3(0, 1, 0), 1000)
            ),
            ObjectAttributes(
                'obj/donut.obj', 'obj/donut.mtl',
                (45.0, 45.0, 45.0), (-30.0, 0.5, -40.0),
                MaterialOverride(glm.vec3(0.4, 0.1, 0.7), glm.vec3(1, 0.1, 1), 1000)
            ),
            ObjectAttributes(
                'obj/cube.obj', 'obj/cube.mtl',
                (-45.0, 45.0, 45.0), (-70.0, 0.0, 50.0),
                MaterialOverride(None, glm.vec3(1, 1, 0), 1000.0)
            ),
            ObjectAttributes(
                'obj/roof_low_poly.obj', 'obj/roof_low_poly.mtl',
                (-90.0, 0.0, 0.0), (-10.0, -10.0, -10.0),
                MaterialOverride(None, None, None),
                10
            ),
            ObjectAttributes(
                'obj/cube.obj', 'obj/cube.mtl',
                (-45.0, 45.0, 45.0), (20.0, 0.0, -50.0),
                MaterialOverride(None, glm.vec3(0, 1, 0), 1000.0)
            )
        ]

        filepaths = [attr.filepath for attr in object_attributes]
        mtl_filepaths = [attr.mtl_filepath for attr in object_attributes]
        rotations = [attr.rotation for attr in object_attributes]
        translations = [attr.translation for attr in object_attributes]
        material_overrides = [attr.material_override for attr in object_attributes]
        scales = [attr.scale for attr in object_attributes]

        self.world = World(filepaths, mtl_filepaths, rotations, translations, material_overrides, scales)
        print('World initialized')
        self.player = Player('obj/body.obj', 'obj/head.obj', 'obj/arm_right.obj', mtl_path='obj/body.mtl',
                             camera=self.camera, default_material=Model.default_material)
        print('Player initialized')
        self.models = [self.player.torso, self.player.right_arm]
        self.models += self.world.objects  #[self.world] +
        for model in self.models:
            print("models in components.models: ", model.name)
        print('Models initialized')
        self.renderer = Renderer(self.shader, self.camera)
        print('Renderer initialized')
        self.input_handler = InputHandler(self.camera, self.player)
        print('Input handler initialized')
        self.text_renderer = TextRenderer(window.width, window.height)
        self.physics = Physics(self.world, self.player)

    def set_input_callbacks(self, window):
        window.set_callbacks(self.input_handler.key_callback, self.input_handler.mouse_callback)

    def update_components(self, delta_time: float):
        self.camera.update(delta_time)
        self.player.update(delta_time)
        self.world.update(delta_time)
