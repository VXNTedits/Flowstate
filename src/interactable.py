import glm
import glfw

from src.composite_model import CompositeModel
from src.model import MaterialOverride


class InteractableObject(CompositeModel):
    def __init__(self,
                 filepath,
                 mtl_filepath,
                 translation=glm.vec3(0, 0, 0),
                 interactable=True,
                 scale=1,
                 rotation=glm.vec3(0, 0, 0),
                 velocity=glm.vec3(0, 0, 0),
                 is_collidable=False,
                 material_overrides=MaterialOverride(None, None, None),
                 use_composite=False,
                 shift_to_centroid=False,
                 is_player=False
                 ):
        super().__init__(
            filepath=filepath,
            mtl_filepath=mtl_filepath,
            player=is_player,
            draw_convex_only=False,
            rotation_angles=rotation,
            translation=translation,
            kd_override=material_overrides.kd_override if material_overrides else None,
            ks_override=material_overrides.ks_override if material_overrides else None,
            ns_override=material_overrides.ns_override if material_overrides else None,
            scale=scale,
            is_collidable=is_collidable,
            shift_to_centroid=shift_to_centroid
        )

        self.model = CompositeModel(filepath,
                                    mtl_filepath,
                                    player=False,
                                    draw_convex_only=False,
                                    rotation_angles=rotation,
                                    translation=translation,
                                    kd_override=material_overrides.kd_override if material_overrides else None,
                                    ks_override=material_overrides.ks_override if material_overrides else None,
                                    ns_override=material_overrides.ns_override if material_overrides else None,
                                    scale=scale,
                                    is_collidable=is_collidable,
                                    shift_to_centroid=shift_to_centroid)

        self.use_composite = use_composite
        self.interactable = interactable
        self.interaction_threshold = 10
        self.bounce_amplitude = 1
        self.bounce_frequency = 1.5  # in cycles per second
        self.rotation_speed = glm.vec3(0, 45, 0)  # degrees per second
        self.picked_up = False
        self.position = self.model.composite_position
        self.rotation = self.model.composite_rotation
        print("centroid = ", self.model.composite_centroid)
        self.rotation_angle = 0.0

        if material_overrides:
            self.material_overrides = material_overrides
        print(f"Interactable {self.name} initialized at {self.position}")

    def interact(self, player):
        if self.interactable:
            print(self.orientation)
            self.on_pickup(player)

    def on_pickup(self, player):
        # Define what happens when the player picks up this object
        if self.interactable:
            self.set_composite_position(glm.vec3(0, 0, 0))
            self.set_composite_rotation(glm.vec3(90, 0, -90))
            print(f"{self.name} picked up by {player.name}.")
            player.inventory.append(self)
            self.interactable = False
            self.picked_up = True
            self.owner = player

    def update_interactable(self, player, delta_time):
        if self.interactable:
            self.check_interactions(player, delta_time)
        if self.picked_up:
            # Do any other stuff upon pickup
            self.update_composite_model_matrix(player.right_hand_model_matrix)
        else:
            self.update_composite_model_matrix()  # Ensure the model matrix is updated for non-picked objects
        player.interact = False

    def check_interactions(self, player, delta_time):
        if glm.distance(player.position, self.position) < self.interaction_threshold:
            self.highlight(delta_time)
            if player.interact:
                print("Player interacted.")
                self.interact(player)
                player.pick_up(self)
                self.update_composite_model_matrix()

    def highlight(self, delta_time):
        # Rotate around the y-axis
        self.rotation_angle += self.rotation_speed.y * delta_time

        centroid = self.model.composite_centroid

        self.rotate_about_centroid(centroid, self.rotation_angle, delta_time)

        # Keep the angle within [0, 360)
        self.rotation_angle %= 360

        # Bounce up and down (apply after rotation to avoid interference)
        bounce_offset = self.bounce_amplitude * glm.sin(2.0 * glm.pi() * self.bounce_frequency * glfw.get_time())
        self.position.y += bounce_offset * delta_time
        self.set_composite_position(self.position)

        # Ensure the model matrix is updated after position and orientation changes
        self.update_composite_model_matrix()

    def rotate_about_centroid(self, centroid, rotation_angle, delta_time):
        initial_position = self.composite_position
        # 1. Translate to origin
        self.set_composite_position(-centroid - initial_position)
        self.update_composite_model_matrix()
        # 2. Rotate
        self.set_composite_rotation((0, rotation_angle, 0))
        self.update_composite_model_matrix()
        # 3. Translate back
        self.set_composite_position(centroid + initial_position)
        self.update_composite_model_matrix()
