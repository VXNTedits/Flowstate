import time

import glm

from src.camera import Camera
from src.composite_model import CompositeModel

from src.model import Model
from src.interactable import InteractableObject
from src.weapon import Weapon
from src.world import World


class Player(CompositeModel):
    def __init__(self, body_path: str, head_path: str, right_arm_path: str, mtl_path: str, camera, default_material,
                 filepath: str, mtl_filepath: str, *args, **kwargs):
        # Player attributes
        self.MAX_RECOIL_ANGLE = 5
        self.ANIMATION_SPEED = 2.5
        self.RECOIL_DURATION = 1.0
        self.is_player = True
        self.player_height = 2
        self.player_width = 1
        self.default_material = default_material
        self.name = 'ego'

        # Camera attributes
        self.camera = camera
        self.yaw = camera.yaw
        self.pitch = camera.pitch

        # Player position and movement
        self.position = glm.vec3(0.0, 0.0, 0.0)
        self.previous_position = self.position
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.velocity = glm.vec3(0, 0, 0)
        self.accelerator = 10
        self.jump_force = glm.vec3(0, 5, 0)
        self.max_speed = 10.0
        self.thrust = glm.vec3(0.0, 0.0, 0.0)
        self.proposed_thrust = self.thrust
        self.is_grounded = False
        self.is_jumping = False
        self.trajectory = self.position - self.previous_position
        self.jump_cooldown_limit = 1
        self.jump_cooldown = self.jump_cooldown_limit

        # Interaction
        self.inventory = []
        self.interact = False

        # Shooting
        self.is_shooting = False
        self.last_shot_time = 0
        self.mouse_buttons = [False, False]
        self.animation_accumulator = 0.0

        # Models and transformations      # +left +up  +fwd
        self.local_hand_position = glm.vec3(-0.25, -0.3, 0.6)
        self.right_arm_offset = glm.vec3(0.0, 0.0, 0.0)  # In case right arm needs an offset (+x: +left)

        # Bounding box
        self.bounding_box = self.calculate_player_bounding_box(self.previous_position, self.position)

        # Composite model setup. The head is the composite's root. It _is_ the instance CompositeModel.
        super().__init__(filepath=head_path, mtl_filepath=mtl_path, translation=glm.vec3(0, 0, 1.7),
                         rotation_angles=glm.vec3(-90, 180, 0), player=True,
                         *args, **kwargs)
        self.translate_vertices(glm.vec3(0, 0, -1.7))
        self.add_comp_model(
            Model(body_path, mtl_path, player=True, translation=(0, 0, 0)))
        self.add_comp_model(
            Model(right_arm_path, mtl_path, player=True), relative_position=glm.vec3(0, 1.4, 0),
            relative_rotation=glm.vec3(90, 0, 0))
        self.models[2][0].translate_vertices(glm.vec3(0, -1.4, 0))

    def update_player(self, delta_time: float, mouse_buttons: list, world: World):
        # Updates camera position
        self.update_camera_position()

        # Updates previous position before changing current position
        self.previous_position = self.position

        # Updates position based on velocity and delta_time. The head _is_ the player's position.
        self.position += self.velocity * delta_time
        self.update_composite_player_model()

        # Calculates the new trajectory based on the updated position
        self.trajectory = glm.normalize(self.position - self.previous_position)

        # Updates jumping status
        if self.velocity.y <= -0.01:
            self.is_jumping = False

        # Calculates the player's bounding box based on the new position
        self.calculate_player_bounding_box(self.previous_position, self.position)

        # Updates combat logic (shooting, animations, etc)
        self.update_combat(delta_time, mouse_buttons, world)

    def shoot(self, weapon: Weapon, delta_time, world):
        self.is_shooting = True
        current_time = time.time()
        fire_rate = weapon.fire_rate
        time_between_shots = 1.0 / fire_rate if fire_rate > 0 else float('inf')

        if fire_rate == -1 and self.last_shot_time != 0:
            # Single-fire mode: ensure only one bullet is fired
            return

        if current_time - self.last_shot_time >= time_between_shots:
            print("Player shot.")
            weapon.animate_shoot(delta_time)
            weapon.initialize_trajectory(
                initial_position=self.right_hand_position,
                player_pitch=self.pitch,
                player_yaw=self.yaw,
                delta_time=delta_time
            )
            self.last_shot_time = current_time

    def reset_thrust(self):
        self.thrust = glm.vec3(0.0, 0.0, 0.0)
        self.proposed_thrust = glm.vec3(0.0, 0.0, 0.0)

    def propose_updated_thrust(self, directions: list, delta_time: float):
        front = glm.vec3(glm.cos(glm.radians(self.yaw)), 0, glm.sin(glm.radians(self.yaw)))
        right = glm.normalize(glm.cross(front, self.up))
        proposed_thrust = glm.vec3(0, 0, 0)

        for direction in directions:
            if direction == 'FORWARD':
                proposed_thrust += front  # * self.accelerator
            if direction == 'BACKWARD':
                proposed_thrust += -front  # * self.accelerator
            if direction == 'LEFT':
                proposed_thrust += -right  # * self.accelerator
            if direction == 'RIGHT':
                proposed_thrust += right  # * self.accelerator
            if direction == 'JUMP' and self.is_grounded:
                self.is_jumping = True
                proposed_thrust += self.jump_force
                self.is_grounded = False
                print('jump: updated proposed_thrust to', proposed_thrust, "is jumping=", self.is_jumping)
            if direction == 'INTERACT':
                self.interact = True
                print("Player interacted")

        self.proposed_thrust = proposed_thrust

    def update_right_hand_model_matrix(self, ads=False):
        local_hand_vec4 = glm.vec4(self.local_hand_position, 1.0)
        transformed_hand_position = self.models[2][0].model_matrix * local_hand_vec4

        self.right_hand_model_matrix = glm.mat4(self.models[2][0].model_matrix)

        self.right_hand_model_matrix[3] = glm.vec4(transformed_hand_position.x,
                                                   transformed_hand_position.y,
                                                   transformed_hand_position.z, 1.0)

        self.right_hand_position = glm.vec3(transformed_hand_position.x,
                                            transformed_hand_position.y,
                                            transformed_hand_position.z)

        if ads:
            # TODO: Compensate for arm's rotation to align the hand model matrix with the view
            self.right_hand_orientation = glm.vec3(self.models[0][0].orientation) + glm.vec3(0, 90, 0)

        else:
            self.right_hand_orientation = glm.vec3(self.models[2][0].orientation) + glm.vec3(0, 90, 0)

    def animate_hipfire_recoil(self, delta_time):
        models = self.get_objects()
        arm_right = models[2]

        if self.animation_accumulator <= self.RECOIL_DURATION:
            recoil_angle = -self.MAX_RECOIL_ANGLE * (
                    self.RECOIL_DURATION - self.animation_accumulator) / self.RECOIL_DURATION
            recoil_rotation = glm.vec3(recoil_angle, 0, 0)
            #self.set_relative_transform(arm_right, self.right_arm_offset, recoil_rotation)
            print(recoil_rotation)
            arm_right.set_orientation(arm_right.orientation + recoil_rotation)
            self.animation_accumulator += delta_time * self.ANIMATION_SPEED
        else:
            # arm_right.set_orientation(arm_right.orientation)
            self.animation_accumulator = 0.0
            self.is_shooting = False

    # def ads(self, delta_time):
    #     """Rotates the right arm such that the right hand is positioned at the view center"""
    #     self.camera.zoom = 1.1
    #
    #     # The parameter to be adjusted
    #     right_arm_model = self.models[2][0]
    #
    #     # Calculate the view direction vector
    #     view_direction = glm.vec3(
    #         glm.cos(glm.radians(-self.yaw-90)) * glm.cos(glm.radians(-self.pitch)),
    #         glm.sin(glm.radians(-self.pitch)),
    #         glm.sin(glm.radians(-self.yaw-90)) * glm.cos(glm.radians(-self.pitch))
    #     )
    #
    #     # Calculate the desired position of the right hand in world space
    #     shoulder_position = right_arm_model.position
    #     desired_hand_position = shoulder_position + view_direction * 0.795
    #
    #     # Use IK to calculate the necessary rotation for the right arm
    #     current_hand_position = self.right_hand_position
    #     direction_to_target = glm.normalize(desired_hand_position - current_hand_position)
    #     current_direction = glm.vec3(1, 0, 0)  # Assuming the arm points along the X axis
    #
    #     # Calculate the quaternion rotation required to align the arm with the target direction
    #     rotation_quaternion = glm.quatLookAt(direction_to_target, glm.vec3(0, 1, 0))  # Up vector is Y
    #
    #     # Convert quaternion to Euler angles for the arm model
    #     rotation_euler = glm.eulerAngles(rotation_quaternion) + glm.vec3(90, 0, 0)
    #
    #     # Apply the rotation to the right arm model
    #     self.set_relative_transform(right_arm_model, glm.vec3(0, 0, 0), rotation_euler)
    #
    #     # Rotates the right hand model matrix to align its orientation with the view direction
    #     self.update_right_hand_model_matrix(ads=True)

    # def ads(self, delta_time):
    #     """Rotates the right arm such that the right hand is positioned at the view center"""
    #     self.camera.zoom = 1.1
    #
    #     # The parameter to be adjusted
    #     right_arm_model = self.models[2][0]  # Can also be directly written to right_arm_model.model_matrix
    #
    #     # Calculate the rotation correction required based on player's pitch and yaw.
    #     # When the player pitches, the end of the arm does not remain in the center of the view
    #     # because the shoulder and the camera are offset from each other.
    #     pitch_correction_function = -1 * glm.sin(glm.radians(self.pitch))
    #     yaw_correction_function = 25 * glm.sin(glm.radians(self.yaw))
    #     roll_corrector_function = 1 * glm.sin(glm.radians(self.pitch))
    #
    #     correction_vector = glm.vec3(pitch_correction_function, yaw_correction_function, roll_corrector_function)
    #
    #     # At zero pitch, the values glm.vec3(90, 25, 0) apply the correct rotation.
    #     base_rotation = glm.vec3(90, 25, 0)
    #     rotation = correction_vector + base_rotation
    #
    #     # Applies a relative rotation to the right arm w.r.t. the camera
    #     self.set_relative_transform(right_arm_model, glm.vec3(0, 0, 0), rotation)
    #
    #     # Rotates the right hand model matrix to align its orientation with the view direction
    #     self.update_right_hand_model_matrix(ads=True)

    def ads(self, delta_time):
        """Rotates the right arm such that the right hand is positioned at the view center"""
        self.camera.zoom = 1.1

        right_arm_model = self.models[2][0]

        pitch_angle = -glm.degrees(glm.tan(0.3 / 0.795)) + (-self.pitch)
        yaw_angle = glm.degrees(glm.sin(0.25/0.795)) + (-self.yaw+90)

        right_arm_model.set_orientation(glm.vec3(pitch_angle, yaw_angle, 0))
        # Rotates the right hand model matrix to align its orientation with the view direction
        self.update_right_hand_model_matrix(ads=True)

    def get_rotation_matrix(self):
        return glm.rotate(glm.mat4(1.0), glm.radians(self.yaw), glm.vec3(0.0, 1.0, 0.0))

    def set_player_position(self, position):
        self.position = position

    def draw_player(self, camera: Camera):
        if not camera.first_person:
            for model in self.get_objects():
                model.draw()
        else:
            self.get_objects()[1].draw()
            self.get_objects()[2].draw()

    def process_mouse_movement(self, xoffset, yoffset):
        self.camera.process_mouse_movement(xoffset, yoffset)
        self.yaw = self.camera.yaw
        self.pitch = self.camera.pitch
        self.update_camera_position()

    def update_camera_position(self):
        self.camera.yaw = self.yaw
        self.camera.pitch = self.pitch
        self.camera.update_camera_vectors()
        if self.camera.first_person:
            self.camera.set_first_person(self.get_objects()[0].position)
        else:
            self.camera.set_third_person(self.get_objects()[0].position)

    def pick_up(self, interactable_object):
        if isinstance(interactable_object, InteractableObject):
            interactable_object.interact(self)

    def calculate_player_bounding_box(self, start_pos, end_pos, bounding_margin=0.1):
        min_x = min(start_pos.x, end_pos.x) - self.player_width / 2 - bounding_margin
        max_x = max(start_pos.x, end_pos.x) + self.player_width / 2 + bounding_margin
        min_y = min(start_pos.y, end_pos.y) - bounding_margin
        max_y = max(start_pos.y, end_pos.y) + self.player_height + bounding_margin
        min_z = min(start_pos.z, end_pos.z) - self.player_width / 2 - bounding_margin
        max_z = max(start_pos.z, end_pos.z) + self.player_width / 2 + bounding_margin

        bounding_box = [(min_x, min_y, min_z), (max_x, max_y, max_z)]
        return bounding_box

    def update_combat(self, delta_time, mouse_buttons: list, world: World):
        if self.inventory:
            if mouse_buttons[0]:
                self.shoot(self.inventory[0], delta_time, world)
            self.inventory[0].update_weapon(delta_time)
            if self.is_shooting:
                self.animate_hipfire_recoil(delta_time)
            if mouse_buttons[1]:
                self.ads(delta_time)
            elif not mouse_buttons[1]:
                self.camera.zoom = 1.0

    def handle_left_click(self, is_left_mouse_button_pressed):
        if is_left_mouse_button_pressed:
            print("LMB pressed.")
            self.mouse_buttons[0] = True
        elif not is_left_mouse_button_pressed:
            print("LMB not pressed.")
            print()
            self.mouse_buttons[0] = False

    def handle_right_click(self, is_right_mouse_button_pressed):
        if is_right_mouse_button_pressed:
            print("RMB pressed.")
            self.mouse_buttons[1] = True
        elif not is_right_mouse_button_pressed:
            print("RMB not pressed.")
            print()
            self.mouse_buttons[1] = False

    def update_composite_player_model(self, neck_pos=(0, 1.5, 0), shoulder_pos=(0.5, 1.4, 0)):
        """ Responsible for animating and updating individual body parts """
        # First, update all positions.
        initial_position = self.position
        for m, _, _ in self.models:
            m.set_position(initial_position)

        # 1. The head is the root model in self.models[0][0]
        #    It must follow the camera's pitch and yaw (self.pitch, self.yaw),
        #    rotating about the local point neck_pos=(0, 1.5, 0)
        head_model = self.models[0][0]
        head_rotation = (-self.pitch - 90, -self.yaw + 90, 0)
        head_model.set_orientation(head_rotation, pivot_point=initial_position + neck_pos)
        head_model.set_position(initial_position)

        # 2. The chest is the first child model (self.models[1][0])
        #    Using a lerp function, it must follow the camera's yaw
        #    and its position must be updated according to the head
        chest_model = self.models[1][0]
        chest_rotation = (-90, -self.yaw + 90, 0)
        chest_model.set_orientation(glm.mix(chest_model.orientation, chest_rotation, 0.05))

        # Interpolating position for chest model based on head position
        chest_pos = glm.vec3(head_model.position) + glm.vec3(0, -1.7, 0)  # Adjust position relative to the head
        chest_model.set_position(chest_pos)

        # 3. The right arm is the second child model (self.models[2][0])
        #    It must lerp the camera's pitch and yaw, pivoting about the shoulder,
        #    and its position must be updated according to the head
        right_arm_model = self.models[2][0]
        arm_rotation = (-self.pitch, -self.yaw + 90, 0)
        right_arm_model.set_orientation(glm.mix(right_arm_model.orientation, arm_rotation, 0.1),
                                        pivot_point=initial_position + shoulder_pos)

        # Interpolating position for right arm model based on chest position
        arm_pos = glm.vec3(chest_model.position) + glm.vec3(0, 1.4, 0)  # Adjust position relative to the chest
        right_arm_model.set_position(arm_pos)
        self.update_right_hand_model_matrix()
