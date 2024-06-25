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
                 filepath: str, mtl_filepath: str):
        # Player attributes
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

        # Models and transformations
        self.local_hand_position = glm.vec3(-0.35, -0.8, 1.0)
        self.torso = CompositeModel(body_path, mtl_path, player=True, translation=(0, 0, 0))
        self.head = Model(head_path, mtl_path, player=True, translation=(0, 0, 0))
        self.right_arm = Model(right_arm_path, mtl_path, player=True, translation=(0, 0, 0))
        self.right_arm_offset = glm.vec3(0.0, 0.0, 0.0)  # In case right arm needs an offset (+x: +left)
        self.vertices, self.indices = Model.load_obj(self, body_path)

        # Initial model matrix
        self.model_matrix = glm.rotate(glm.mat4(1.0), glm.radians(-90), glm.vec3(1.0, 0.0, 0.0))
        self.rotation = glm.vec3(camera.pitch, camera.yaw, 0)
        self.initialize_player_model_matrix()
        self.set_right_hand_model_matrix()

        # Bounding box
        self.bounding_box = self.calculate_player_bounding_box(self.previous_position, self.position)

        # Composite model setup
        # super().__init__(filepath, mtl_filepath)
        self.torso.add_comp_model(self.head)
        self.torso.add_comp_model(self.right_arm)

    def update_player(self, delta_time: float, mouse_buttons: list, world: World):
        # Update previous position before changing current position
        self.previous_position = self.position

        # Update position based on velocity and delta_time
        self.position += self.velocity * delta_time

        # Update camera position
        self.update_camera_position()

        # Update the player's model matrix (torso, head, right arm) based on the new position
        self.initialize_player_model_matrix()

        # Update composite model matrices for all parts relative to the torso
        self.torso.update_composite_model_matrix(self.torso.model_matrix)

        # Calculate the new trajectory based on the updated position
        self.trajectory = glm.normalize(self.position - self.previous_position)

        # Update jumping status
        if self.velocity.y <= -0.01:
            self.is_jumping = False

        # Calculate the player's bounding box based on the new position
        self.calculate_player_bounding_box(self.previous_position, self.position)

        # Update combat logic (e.g., handling shooting)
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

    def initialize_player_model_matrix(self):
        model_rotation = glm.rotate(glm.mat4(1.0), glm.radians(-self.yaw), glm.vec3(0.0, 1.0, 0.0))
        model_rotation *= glm.rotate(glm.mat4(1.0), glm.radians(90), glm.vec3(0.0, 1.0, 0.0))
        model_rotation *= glm.rotate(glm.mat4(1.0), glm.radians(-90), glm.vec3(1.0, 0.0, 0.0))

        translation = glm.translate(glm.mat4(1.0), self.position)
        final_model_matrix = translation * model_rotation

        self.torso.model_matrix = final_model_matrix
        right_arm_translation = glm.translate(glm.mat4(1.0), self.right_arm_offset)
        self.right_arm.model_matrix = final_model_matrix * right_arm_translation
        self.head.model_matrix = final_model_matrix

        self.rotation = model_rotation
        self.model_matrix = self.torso.model_matrix
        self.set_right_hand_model_matrix()


    def set_right_hand_model_matrix(self):
        local_hand_vec4 = glm.vec4(self.local_hand_position, 1.0)
        transformed_hand_position = self.right_arm.model_matrix * local_hand_vec4
        self.right_hand_model_matrix = glm.mat4(self.right_arm.model_matrix)
        self.right_hand_model_matrix[3] = glm.vec4(transformed_hand_position.x, transformed_hand_position.y,
                                                   transformed_hand_position.z, 1.0)
        self.right_hand_position = glm.vec3(transformed_hand_position.x, transformed_hand_position.y,
                                            transformed_hand_position.z)

    def animate_hipfire_recoil(self, delta_time):
        models = self.torso.get_objects()
        self.torso = models[0]
        arm_right = models[2]
        if self.animation_accumulator <= 45:
            # print("self.is_shooting: ", self.is_shooting)
            recoil_rotation = glm.vec3(0, 0, 45-self.animation_accumulator)
            self.torso.set_relative_transform(arm_right, self.right_arm_offset, recoil_rotation)
            self.animation_accumulator += delta_time * 20
            #root.update_composite_model_matrix()
        else:
            self.torso.set_relative_transform(arm_right, self.right_arm_offset, glm.vec3(0,0,0))
            self.animation_accumulator = 0.0
            self.is_shooting = False

    def get_rotation_matrix(self):
        return glm.rotate(glm.mat4(1.0), glm.radians(self.yaw), glm.vec3(0.0, 1.0, 0.0))

    def set_position(self, position):
        self.position = position
        self.initialize_player_model_matrix()

    def draw(self, camera: Camera):
        if not camera.first_person:
            self.head.draw()
            self.torso.draw()
            self.right_arm.draw()
        else:
            self.torso.draw()
            self.right_arm.draw()

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
            self.camera.set_first_person(self.position, self.get_rotation_matrix())
        else:
            self.camera.set_third_person(self.position, self.get_rotation_matrix())

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

    def handle_left_click(self, is_left_mouse_button_pressed):
        if is_left_mouse_button_pressed:
            print("LMB pressed.")
            self.mouse_buttons[0] = True
        elif not is_left_mouse_button_pressed:
            print("LMB not pressed.")
            print()
            self.mouse_buttons[0] = False
