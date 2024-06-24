import os

from src.caliber import Caliber
from src.camera import Camera
from src.model import Model
from src.shader import Shader
from src.player import Player
from src.renderer import Renderer
from src.input_handler import InputHandler
from src.physics import Physics
from src.weapon import Weapon
from text_renderer import TextRenderer
import glm
from src.world import World
from src.world_objects import WorldObjects, MaterialOverride
from src.interactable import InteractableObject
from utils.file_utils import get_relative_path


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
        self.script_dir = os.path.dirname(os.path.dirname(__file__))
        self.window = window
        self.camera = None
        self.world = None
        self.world_objects = None
        self.player = None
        self.models = []
        self.physics = None
        self.input_handler = None
        self.shader = None
        self.renderer = None
        self.interactables = []
        self.weapons = []

    def initialize_gameplay_components(self):

        self.camera = Camera(glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))

        # Define object attributes for multiple objects
        world_objects = [
            ObjectAttributes(get_relative_path("res/donut.obj"), get_relative_path("res/cube.mtl"),
                             (-45.0, 45.0, 45.0), (-50.0, 5.0, 0.0),
                             MaterialOverride(None, glm.vec3(1, 1, 0), 1000.0)),
            ObjectAttributes(get_relative_path("res/cube.obj"), get_relative_path("res/cube.mtl"),
                             (-45.0, 45.0, 45.0), (50.0, 5.0, -5.0),
                             MaterialOverride(None, glm.vec3(0, 1, 0), 1000.0))
        ]

        filepaths = [attr.filepath for attr in world_objects]
        mtl_filepaths = [attr.mtl_filepath for attr in world_objects]
        rotations = [attr.rotation for attr in world_objects]
        translations = [attr.translation for attr in world_objects]
        material_overrides = [attr.material_override for attr in world_objects]
        scales = [attr.scale for attr in world_objects]

        self.world_objects = WorldObjects(filepaths,
                                          mtl_filepaths,
                                          rotations,
                                          translations,
                                          material_overrides,
                                          scales)

        self.world = World("world2",air_density=1.3)
        print("World initialized")

        self.player = Player(get_relative_path("res/body.obj"),
                             get_relative_path("res/head.obj"),
                             get_relative_path("res/arm_right.obj"),
                             mtl_path=get_relative_path("res/body.mtl"),
                             camera=self.camera,
                             default_material=Model.default_material,
                             filepath=get_relative_path("res/body.obj"),
                             mtl_filepath=get_relative_path("res/body.mtl"))
        print("Player initialized")

        self.models = [self.player.torso, self.player.right_arm]
        self.models += self.world.get_world_objects()
        self.models += self.world_objects.objects
        self.models += self.interactables
        print("Components created")

        for model in self.models:
            print(model)
            print("model in components.models: ", model.name)
        print("Models initialized")

        self.physics = Physics(self.world_objects, self.player, self.interactables, self.world)
        print("Physics initialized")


        fifty_ae = Caliber(initial_velocity=470,mass=0.02,drag_coefficient=0.5, bullet_area=0.000127)

        deagle = Weapon(
            fire_rate=1,
            bullet_velocity_modifier=1,
            caliber=fifty_ae,
            filepath=get_relative_path("res/deagle_main.obj"),
            mtl_filepath=get_relative_path("res/deagle_main.mtl"),
            translation=glm.vec3(0, 0, 0),  # (3.0, 2.0, -3.0),
            rotation=glm.vec3(0, 0, 0),
            scale=1,  # TODO: Scaling doesn't work as expected between root and child
            is_collidable=False,
            material_overrides=MaterialOverride(None, glm.vec3(1, 1, 1), 500),
            use_composite=True,
            shift_to_centroid=False,
            physics=self.physics
        )

        deagle_slide = Model(filepath=get_relative_path("res/deagle_slide.obj"),
                             mtl_filepath=get_relative_path("res/deagle_slide.mtl"),
                             shift_to_centroid=False,
                             scale=1)
        deagle.add_comp_model(model=deagle_slide,
                              relative_position=glm.vec3(0.0, 0.0, 0.0),
                              relative_rotation=glm.vec3(0.0, 0.0, 0.0))

        test_cube_interactable = InteractableObject(filepath=get_relative_path("res/10cube.obj"),
                                                    mtl_filepath=get_relative_path("res/10cube.mtl"),
                                                    use_composite=False,
                                                    shift_to_centroid=False,
                                                    translation=glm.vec3(20,5,-20))

         # TODO: Events are only polled for the first interactable in the list ???
        self.add_interactable(deagle)
        self.add_interactable(test_cube_interactable)

        self.input_handler = InputHandler(self.camera, self.player, self.physics)
        print("Input handler initialized")
        self.shader = Shader(get_relative_path("shaders/vertex_shader.glsl"),
                             get_relative_path("shaders/fragment_shader.glsl"))
        print("Shader manager initialized")

        self.renderer = Renderer(self.shader, self.camera, self.physics, self.weapons)
        print("Renderer initialized")

        # TESTING
        #print("vertices before add_crater:\n", test_cube_interactable.vertices)
        #test_cube_interactable.add_crater(glm.vec3(1.5,1.5,1.5),0.1,0.1)
        #print("vertices after add_crater:\n", test_cube_interactable.vertices)
        # --//--

    def set_input_callbacks(self):
        self.window.set_callbacks(self.input_handler.key_callback,
                                  self.input_handler.mouse_callback,
                                  self.input_handler.mouse_button_callback)

    def update_components(self, delta_time: float):
        self.world_objects.update(delta_time)
        for interactable in self.interactables:
            interactable.update_interactable(self.player, delta_time)

    def add_interactable(self, interactable_object):
        self.interactables.append(interactable_object)
        self.models.append(interactable_object)

        if isinstance(interactable_object, Weapon):
            self.add_weapon(interactable_object)


    def add_weapon(self, weapon):
        self.weapons.append(weapon)
