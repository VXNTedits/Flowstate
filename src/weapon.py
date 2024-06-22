from src.caliber import Caliber
from src.interactable import InteractableObject
import glm

from src.physics import Physics


class Weapon(InteractableObject):
    def __init__(self,
                 fire_rate,
                 bullet_velocity_modifier,
                 caliber: Caliber,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.active_trajectories = []
        self.projectile_positions = []
        self.fire_rate = fire_rate
        self.bullet_velocity_modifier = bullet_velocity_modifier
        self.caliber = caliber
        self.instantaneous_bullet_velocity = glm.vec3(0, 0, 0)
        self.instantaneous_bullet_position = glm.vec3(0, 0, 0)
        self.shoot = False

    def update_weapon(self, delta_time):
        pass

    def initialize_trajectory(self, initial_position, player_pitch, player_yaw):
        pitch = glm.radians(player_pitch)
        yaw = glm.radians(player_yaw)
        direction = glm.vec3(
            glm.cos(yaw) * glm.cos(pitch),
            glm.sin(pitch),
            glm.sin(yaw) * glm.cos(pitch)
        )
        instantaneous_bullet_velocity = glm.normalize(direction) * self.caliber.initial_velocity
        trajectory = {
            'position': initial_position,
            'velocity': instantaneous_bullet_velocity,
            'positions': [initial_position],
            'elapsed_time': 0.0,
            'dirty': True  # Mark trajectory as dirty for initial buffer update
        }
        self.active_trajectories.append(trajectory)