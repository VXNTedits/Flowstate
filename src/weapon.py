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
        self.physics = physics
        self.initial_position = glm.vec3(0, 0, 0)
        self.tracer_lifetime = 1.0
        self.tracers = []

    def update_weapon(self, delta_time):
        pass

    def initialize_trajectory(self, initial_position, player_pitch, player_yaw, delta_time):
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
            'positions': [glm.vec3(initial_position)],  # Copy here as well
            'elapsed_time': 0.0,
            'dirty': True  # Mark trajectory as dirty for initial buffer update
        }
        self.active_trajectories.append(trajectory)
        self.initial_position = glm.vec3(initial_position)  # Copy to ensure it remains unchanged
        # print("Trajectory origin = ", self.initial_position)
        self.instantaneous_bullet_position = glm.vec3(initial_position)  # Make a copy here
        self.instantaneous_bullet_velocity = velocity

        while not self.physics.is_out_of_bounds(self.instantaneous_bullet_position):
            # print(
            #    f"Trajectory origin = {self.initial_position}, Current position = {self.instantaneous_bullet_position}")
            drag_force = self.physics.calculate_drag_force(self.instantaneous_bullet_velocity, self.caliber)
            acceleration = drag_force / self.caliber.mass - self.physics.gravity
            self.instantaneous_bullet_velocity += acceleration * delta_time
            self.instantaneous_bullet_position += self.instantaneous_bullet_velocity * delta_time
            trajectory['positions'].append(
                glm.vec3(self.instantaneous_bullet_position))  # Copy the position to the list
            if self.physics.check_projectile_collision(trajectory['positions']):
                print("Projectile collision detected!")
                break
