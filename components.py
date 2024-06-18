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

from world import World
from world_objects import WorldObjects, MaterialOverride
from interactable import InteractableObject


class ObjectAttributes:
    def __init__(self, filepath, mtl_filepath, rotation=[0, 0, 0], translation=[0, 0, 0], material_override=None,
                 scale=1):
        self.scale = scale
        self.filepath = filepath
        self.mtl_filepath = mtl_filepath
        self.rotation = rotation
        self.translation = translation
        self.material_override = material_override


class Components:
    def __init__(self, window):
        self.camera = Camera(glm.vec3(0.0, 0.0, 3.0), glm.vec3(0.0, 1.0, 0.0))

        # Define object attributes for multiple objects
        world_objects = [
            # ObjectAttributes(
            #     'obj/world_test.obj', 'obj/world_test.mtl',
            #     (-90.0, -10.0, 0.0), (-70.0, -50.0, 50.0),
            #     MaterialOverride(None, glm.vec3(0, 1, 0), 1000)
            # ),
            ObjectAttributes(
                'obj/cube.obj', 'obj/cube.mtl',
                (-45.0, 45.0, 45.0), (-70.0, 0.0, 50.0),
                MaterialOverride(None, glm.vec3(1, 1, 0), 1000.0)
            ),
            # ObjectAttributes(
            #     'obj/roof_low_poly.obj', 'obj/roof_low_poly.mtl',
            #     (-90.0, 0.0, 0.0), (-10.0, -10.0, -10.0),
            #     MaterialOverride(None, None, None),
            #     100
            # ),
            ObjectAttributes(
                'obj/cube.obj', 'obj/cube.mtl',
                (-45.0, 45.0, 45.0), (20.0, 0.0, -50.0),
                MaterialOverride(None, glm.vec3(0, 1, 0), 1000.0)
            )
        ]

        filepaths = [attr.filepath for attr in world_objects]
        mtl_filepaths = [attr.mtl_filepath for attr in world_objects]
        rotations = [attr.rotation for attr in world_objects]
        translations = [attr.translation for attr in world_objects]
        material_overrides = [attr.material_override for attr in world_objects]
        scales = [attr.scale for attr in world_objects]

        deagle = InteractableObject(
            filepath='obj/deagle_main.obj',
            mtl_filepath='obj/deagle_main.mtl',
            translation=glm.vec3(0.0, 0.0, 0.0),
            rotation=glm.vec3(90, 0, 0),
            scale=1,
            is_collidable=False,
            material_overrides=MaterialOverride(None, glm.vec3(1, 1, 1), 500),
            use_composite=True
        )
        deagle_slide = Model(
            filepath='obj/deagle_slide.obj',
            mtl_filepath='obj/deagle_slide.mtl',
            shift_to_centroid=True,
            scale=1
        )
        deagle.add_sub_model(sub_model=deagle_slide, relative_position=glm.vec3(0, 2, 0.1),
                             relative_rotation=glm.vec3(0, 0, 0), scale=2)

        self.interactables = [
            deagle
        ]
        self.world = World('world2')
        self.world_objects = WorldObjects(filepaths, mtl_filepaths, rotations, translations, material_overrides, scales)
        print('World initialized')
        self.player = Player('obj/body.obj', 'obj/head.obj', 'obj/arm_right.obj', mtl_path='obj/body.mtl',
                             camera=self.camera, default_material=Model.default_material, filepath='obj/body.obj',
                             mtl_filepath='obj/body.mtl')
        print('Player initialized')

        self.models = [self.player.torso, self.player.right_arm]
        self.models += self.world.get_objects()
        self.models += self.world_objects.objects
        self.models += self.interactables

        for model in self.models:
            print("models in components.models: ", model.name)
        print('Models initialized')

        self.physics = Physics(self.world_objects, self.player, self.interactables, self.world)
        print('Physics initialized')

        self.input_handler = InputHandler(self.camera, self.player, self.physics)
        print('Input handler initialized')

        self.shader = Shader('shaders/vertex_shader.glsl', 'shaders/fragment_shader.glsl')
        print('Shader initialized')

        self.renderer = Renderer(self.shader, self.camera)
        print('Renderer initialized')

    def set_input_callbacks(self, window):
        window.set_callbacks(self.input_handler.key_callback, self.input_handler.mouse_callback)

    def update_components(self, delta_time: float):
        self.world_objects.update(delta_time)
        for interactable in self.interactables:
            interactable.update(self.player, delta_time)

    def add_interactable(self, interactable_object):
        self.interactables.append(interactable_object)
        self.models.append(interactable_object)
