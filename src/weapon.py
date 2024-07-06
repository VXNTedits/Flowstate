import numpy as np

from src.caliber import Caliber
from src.interactable import InteractableObject
import glm

from src.physics import Physics

class Weapon(InteractableObject):
    def __init__(self,
                 fire_rate,
                 bullet_velocity_modifier,
                 caliber: Caliber,
                 physics: Physics,
                 name: str,
                 player_owner,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.animation_accumulator = 0.0
        self.active_trajectories = []
        self.projectile_positions = []
        self.fire_rate = fire_rate
        self.bullet_velocity_modifier = bullet_velocity_modifier
        self.caliber = caliber
        self.instantaneous_bullet_velocity = glm.vec3(0, 0, 0)
        self.instantaneous_bullet_position = glm.vec3(0, 0, 0)
        self.shoot = False
        self.physics = physics
        self.initial_position = glm.vec3(0, 0, 0)
        self.tracer_lifetime = 0.5
        self.tracers = []
        self.name = name
        self.player_owner = player_owner

    def update_weapon(self, delta_time):
        if self.shoot:
            self.animate_shoot(delta_time)

    def initialize_trajectory(self, initial_position, player_pitch, player_yaw, delta_time):
        self.shoot = True
        print("Initializing trajectory...")
        if not self.physics:
            print("Physics context is not available. Skipping trajectory initialization.")
            return  # Skip the trajectory computation if physics is None

        pitch = glm.radians(player_pitch)
        yaw = glm.radians(player_yaw)
        direction = glm.vec3(
            glm.cos(yaw) * glm.cos(pitch),  # X component affected by both pitch and yaw
            glm.sin(pitch),  # Y component affected by pitch only
            glm.sin(yaw) * glm.cos(pitch)  # Z component affected by both pitch and yaw
        )
        velocity = glm.normalize(direction) * self.caliber.initial_velocity
        trajectory = {
            'position': glm.vec3(initial_position),  # Ensure a copy is made if glm.vec3 isn't automatically one
            'velocity': velocity,
            'positions': [{'position': glm.vec3(initial_position), 'lifetime': 0.0}],  # Initial position with lifetime
            'elapsed_time': 0.0,
            'dirty': True  # Mark trajectory as dirty for initial buffer update
        }
        self.active_trajectories.append(trajectory)
        self.initial_position = glm.vec3(initial_position)  # Copy to ensure it remains unchanged
        self.instantaneous_bullet_position = glm.vec3(initial_position)  # Make a copy here
        self.instantaneous_bullet_velocity = velocity

    def get_tracer_positions(self):
        positions = []
        lifetimes = []
        for tracer in self.tracers:
            positions.append(tracer['position'])
            lifetimes.append(tracer['lifetime'])
        return np.array(positions, dtype=np.float32), np.array(lifetimes, dtype=np.float32)

    def animate_shoot(self, delta_time):
        if self.name == 'deagle':
            self.shoot_deagle(delta_time)

    def shoot_deagle(self, delta_time):
        models = super().get_objects()
        root = models[0]
        child = models[1]

        if self.animation_accumulator <= 0.2:
            root.set_relative_transform(child,
                                        glm.vec3(-0.2 + self.animation_accumulator, 0, 0),
                                        glm.vec3(0, 0, 0))
            self.animation_accumulator += delta_time * 1.0
        else:
            root.set_relative_transform(child,
                                        glm.vec3(0, 0, 0),
                                        glm.vec3(0, 0, 0))
            self.shoot = False
            self.animation_accumulator = 0.0

