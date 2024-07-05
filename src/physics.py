import sys
from collections import defaultdict
from typing import Tuple, List, Set
import glfw
import glm
import numpy as np

from pid import PIDController
from model import Model
from OpenGL.GL import *


class Physics:
    EPSILON = 1e-6

    def __init__(self, player, interactables: list, world):
        self.max_velocity = 1
        self.world = world
        self.player = player
        self.gravity = glm.vec3(0, -10, 0)
        self.interactables = interactables
        self.pid_controller = PIDController(kp=0.8, ki=0.0, kd=0.0)
        self.offset = 0.1
        self.set_gravity = True
        self.toggle_gravity = False
        self.active_trajectories = []

    def apply_gravity(self, delta_time: float):
        if self.toggle_gravity:
            if not self.player.is_grounded:
                self.player.velocity += self.gravity * delta_time
                self.player.position += self.player.velocity * delta_time + 0.5 * self.gravity * delta_time ** 2
                self.set_gravity = True
            elif self.player.is_grounded:
                self.set_gravity = False

    def check_projectile_collision(self, projectile_positions):
        """
        Checks if the projectile intersects with any object in the world.
        Returns the exact collision point if a collision is detected, otherwise None.
        """
        world_objects = self.world.get_world_objects()

        for i in range(len(projectile_positions) - 1):
            start_pos = projectile_positions[i]
            end_pos = projectile_positions[i + 1]
            for obj in world_objects:
                aabb = obj.aabb
                is_intersecting, nearest_face_name, nearest_face_vectors = self.is_bounding_box_intersecting_aabb(
                    (start_pos, end_pos), aabb)

                if is_intersecting:
                    collision_point = self.calculate_collision_point(start_pos, end_pos, nearest_face_vectors)
                    print("Projectile collision detected at: ", collision_point)
                    return collision_point

        return None

    def check_linear_collision(self):
        """
        Checks if the player's movement intersects with any object in the world.
        Returns all intersecting faces and their vectors.
        """
        start_pos = self.player.previous_position
        end_pos = self.player.position
        player_bb = self.player.calculate_player_bounding_box(start_pos, end_pos)

        collisions = []
        for obj in self.world.get_world_objects():
            aabb = obj.aabb
            is_intersecting, nearest_face_name, nearest_face_vectors = (
                self.is_bounding_box_intersecting_aabb(player_bb, aabb))

            if is_intersecting:
                collisions.append((nearest_face_name, nearest_face_vectors))
        return collisions

    def handle_collisions(self, player_thrust, delta_time):
        #self.adjust_velocity_based_on_wall_collision(player_thrust,delta_time)

        collisions = self.check_linear_collision()

        if collisions:
            # Resolve primary collision
            nearest_face_name, nearest_face_vectors = collisions[0]
            self.simple_resolve_collision(nearest_face_vectors, player_thrust, delta_time, nearest_face_name)

            # Check and resolve secondary collisions
            for face_name, face_vectors in collisions[1:]:
                if self.is_secondary_collision(face_vectors):
                    print("Secondary collision detected!")
                    self.simple_resolve_collision(face_vectors, player_thrust, delta_time, face_name)

    def is_secondary_collision(self, face_vectors):
        """
        Check if there is still a collision after resolving the primary collision.
        """
        p0, p1, p2 = face_vectors

        p0 = glm.vec3(p0)
        p1 = glm.vec3(p1)
        p2 = glm.vec3(p2)

        # Use p0 as the reference point on the plane
        P = p0
        N = glm.cross(p1 - p0, p2 - p0)

        # Normalize the normal vector
        N_normalized = glm.normalize(N)

        V = self.player.position

        # Compute the vector from the player's position to the reference point on the plane
        W = V - P

        # Compute the dot product of W and the normalized normal vector
        dot_product = glm.dot(W, N_normalized)

        # If the dot product is close to zero, there's no collision
        return abs(dot_product) > 1e-5

    def is_bounding_box_intersecting_aabb(self, line_segment, aabb):
        (start_pos, end_pos) = line_segment
        (min_x, min_y, min_z), (max_x, max_y, max_z) = aabb

        # Quick broad phase check
        if not (min_x <= end_pos[0] <= max_x or min_x <= start_pos[0] <= max_x):
            return False, None, None
        if not (min_y <= end_pos[1] <= max_y or min_y <= start_pos[1] <= max_y):
            return False, None, None
        if not (min_z <= end_pos[2] <= max_z or min_z <= start_pos[2] <= max_z):
            return False, None, None

        collision_faces_world = {
            "left": (glm.vec3(min_x, min_y, min_z), glm.vec3(min_x, max_y, min_z), glm.vec3(min_x, min_y, max_z)),
            "right": (glm.vec3(max_x, min_y, min_z), glm.vec3(max_x, max_y, min_z), glm.vec3(max_x, min_y, max_z)),
            "bottom": (glm.vec3(min_x, min_y, min_z), glm.vec3(max_x, min_y, min_z), glm.vec3(min_x, min_y, max_z)),
            "top": (glm.vec3(min_x, max_y, min_z), glm.vec3(max_x, max_y, min_z), glm.vec3(min_x, max_y, max_z)),
            "front": (glm.vec3(min_x, min_y, min_z), glm.vec3(max_x, min_y, min_z), glm.vec3(min_x, max_y, min_z)),
            "back": (glm.vec3(min_x, min_y, max_z), glm.vec3(max_x, min_y, max_z), glm.vec3(min_x, max_y, max_z))
        }

        for face, vertices in collision_faces_world.items():
            if self.line_intersects_plane(glm.vec3(start_pos), glm.vec3(end_pos), vertices):
                return True, face, vertices

        return False, None, None

    def line_intersects_plane(self, start, end, plane_vertices):
        p0, p1, p2 = plane_vertices
        plane_normal = glm.normalize(glm.cross(p1 - p0, p2 - p0))
        plane_d = -glm.dot(plane_normal, p0)
        line_dir = end - start
        line_length = glm.length(line_dir)
        line_dir = glm.normalize(line_dir)

        denom = glm.dot(plane_normal, line_dir)
        if abs(denom) < 1e-6:
            return False  # No intersection, line is parallel to plane

        t = -(glm.dot(plane_normal, start) + plane_d) / denom
        if t < 0 or t > line_length:
            return False  # No intersection within the segment

        intersection_point = start + line_dir * t
        if self.point_in_plane_bounds(intersection_point, plane_vertices):
            return True

        return False

    def point_in_plane_bounds(self, point, plane_vertices):
        p0, p1, p2 = plane_vertices
        u = p1 - p0
        v = p2 - p0
        w = point - p0

        uu = glm.dot(u, u)
        uv = glm.dot(u, v)
        vv = glm.dot(v, v)
        wu = glm.dot(w, u)
        wv = glm.dot(w, v)

        denom = uv * uv - uu * vv
        if abs(denom) < 1e-6:
            return False  # Degenerate plane

        s = (uv * wv - vv * wu) / denom
        t = (uv * wu - uu * wv) / denom

        return (s >= 0) and (t >= 0) and (s + t <= 1)

    def calculate_collision_point(self, start_pos, end_pos, plane_vertices):
        """
        Calculate the exact collision point using plane-line intersection.
        """
        p0, p1, p2 = plane_vertices
        plane_normal = glm.normalize(glm.cross(p1 - p0, p2 - p0))
        plane_d = -glm.dot(plane_normal, p0)
        line_dir = end_pos - start_pos
        line_length = glm.length(line_dir)
        line_dir = glm.normalize(line_dir)

        denom = glm.dot(plane_normal, line_dir)
        t = -(glm.dot(plane_normal, start_pos) + plane_d) / denom

        intersection_point = start_pos + line_dir * t
        return intersection_point

    def apply_forces(self, delta_time: float):
        """ Translates proposed thrust to actual thrust by smoothing the input and finally applies it to velocity """
        # Apply lateral thrust for movement
        """ Horizontal forces"""
        if self.player.proposed_thrust.x != 0.0 or self.player.proposed_thrust.z != 0.0:
            # Calculate desired lateral thrust
            max_thrust = self.player.max_speed * glm.normalize(
                glm.vec3(
                    self.player.proposed_thrust.x,
                    0.0,
                    self.player.proposed_thrust.z
                )
            )

            # Use a lerp function to smoothly interpolate towards the target thrust
            lerp_factor = 1 - glm.exp(-self.player.accelerator * delta_time)

            self.player.thrust.x = self.player.proposed_thrust.x * (
                    1 - lerp_factor) + max_thrust.x * lerp_factor

            self.player.thrust.z = self.player.proposed_thrust.z * (
                    1 - lerp_factor) + max_thrust.z * lerp_factor
        else:
            # Apply braking force to lateral movement
            deceleration = glm.exp(-self.player.accelerator * delta_time)
            self.player.velocity.x *= deceleration
            self.player.velocity.z *= deceleration

            # Ensure that the lateral thrust does not invert due to overshooting the deceleration
            if glm.length(glm.vec3(self.player.thrust.x, 0.0, self.player.thrust.z)) < 0.01:
                self.player.thrust.x = 0.0
                self.player.thrust.z = 0.0

        """ Vertical forces """
        self.player.thrust.y = self.player.proposed_thrust.y

        # Ensure vertical velocity does not invert due to overshooting the deceleration
        if abs(self.player.thrust.y) < 0.01:
            self.player.thrust.y = 0.0

    def simple_resolve_collision(self, collision_face, proposed_thrust, delta_time, nearest_face_name):
        p0, p1, p2 = collision_face

        p0 = glm.vec3(p0)
        p1 = glm.vec3(p1)
        p2 = glm.vec3(p2)

        # Use p0 as the reference point on the plane
        P = p0
        N = glm.cross(p1 - p0, p2 - p0)

        # Normalize the normal vector
        N_normalized = glm.normalize(N)

        V = self.player.position

        # Compute the vector from the player's position to the reference point on the plane
        W = V - P

        # Compute the dot product of W and the normalized normal vector
        dot_product = glm.dot(W, N_normalized)

        # Compute the projection of W onto the normal vector
        projection_onto_normal = dot_product * N_normalized

        # Calculate the new position by correcting only the normal component
        V_corrected = V - projection_onto_normal

        # The player's velocity is in the collision direction
        if (
                (glm.dot(self.player.velocity, N_normalized) >= 0 and not self.player.is_jumping)
                or (glm.dot(proposed_thrust, N_normalized) >= 0 and not self.player.is_jumping)
        ):
            # Correct the player's position component that is normal to the plane
            self.player.position = V_corrected  # * PID_correction

            # Project the player's velocity onto the normal vector
            velocity_projection_onto_normal = glm.dot(self.player.velocity, N_normalized) * N_normalized

            # Subtract the projection from the player's velocity
            proposed_thrust -= velocity_projection_onto_normal
            self.player.velocity -= velocity_projection_onto_normal

        else:
            # The player's velocity is already moving the player in a direction away from the collision surface
            pass

        if nearest_face_name == 'top':
            self.player.is_grounded = True
            self.player.velocity.y = 0.0
        else:
            self.player.is_grounded = False

        self.player.velocity += proposed_thrust

    def limit_speed(self):
        max_speed = self.player.max_speed
        if glm.length(self.player.velocity) > max_speed:
            self.player.velocity = max_speed * glm.normalize(self.player.velocity)

    def update_physics(self, delta_time: float, weapons, player):
        # Apply forces to the player
        self.apply_forces(delta_time)

        # Manage collisions
        nearest_face_vectors = self.check_linear_collision()
        if nearest_face_vectors:
            print("Collision detected!")
            # Collision: Resolve by projecting player onto collision face.
            #            Reject proposed thrust: Restrict entry velocity, but allow player to have exit velocity.
            self.handle_collisions(self.player.thrust, delta_time)
        else:
            # No collision: Accept the proposed thrust
            self.player.velocity += self.player.thrust
            self.apply_gravity(delta_time)

        # Apply speed limiter
        self.limit_speed()

        # Debugging output
        if self.player.is_jumping:
            print("grounded=", self.player.is_grounded, " jumping=", self.player.is_jumping, "\n",
                  "thrust=", self.player.thrust, "\n",
                  "proposed=", self.player.proposed_thrust, "\n",
                  "velocity=", self.player.velocity, "\n")

        # Reset inputs
        self.player.reset_thrust()
        self.update_projectile_trajectory(delta_time=delta_time, weapons=weapons)

    def update_projectile_trajectory(self, delta_time, weapons):
        for weapon in weapons:
            # Age existing tracers and remove expired ones
            self.age_and_remove_expired_tracers(delta_time, weapon)

            for trajectory in weapon.active_trajectories[:]:  # Safe iteration with a shallow copy
                trajectory['elapsed_time'] += delta_time

                current_position = trajectory['positions'][-1]['position']
                current_velocity = trajectory['velocity']
                drag_force = self.calculate_drag_force(current_velocity, weapon.caliber)
                acceleration = drag_force / weapon.caliber.mass - self.gravity
                new_velocity = current_velocity + acceleration * delta_time
                new_position = current_position + new_velocity * delta_time

                trajectory['velocity'] = new_velocity
                trajectory['positions'].append(
                    {'position': new_position, 'lifetime': 0.0})  # Append new position with initial lifetime

                # Create or update the tracer for this trajectory
                weapon.tracers.append({'position': new_position, 'lifetime': 0.0})

                if self.check_projectile_collision([pos['position'] for pos in trajectory['positions']]):
                    print("Projectile collision detected at:", new_position)
                    weapon.active_trajectories.remove(trajectory)
                elif trajectory['elapsed_time'] > weapon.tracer_lifetime:
                    print("Removing trajectory due to time expiration.")
                    weapon.active_trajectories.remove(trajectory)

    def age_and_remove_expired_tracers(self, delta_time, weapon):
        # Increment the lifetime of each tracer position
        for tracer in weapon.tracers:
            tracer['lifetime'] += delta_time

        # Remove tracers whose positions have exceeded their lifetime
        weapon.tracers = [tracer for tracer in weapon.tracers if tracer['lifetime'] < weapon.tracer_lifetime]

    def calculate_drag_force(self, velocity, caliber):
        speed = glm.length(velocity)
        return -0.5 * self.world.air_density * caliber.drag_coefficient * caliber.bullet_area * speed ** 2 * glm.normalize(velocity)

    def is_out_of_bounds(self, position):
        # print("Projectile displacement = ", glm.length(position))
        if glm.length(position) > 511:
            print("Projectile out of bounds.")
            return True
        return False
