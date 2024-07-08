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
        self.toggle_gravity = True
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

    def limit_speed(self):
        max_speed = self.player.max_speed
        if glm.length(self.player.velocity) > max_speed:
            self.player.velocity = max_speed * glm.normalize(self.player.velocity)

    def update_physics(self, delta_time: float, weapons):
        # Apply forces to the player
        self.apply_forces(delta_time)

        # Manage collisions
        for obj in self.world.get_world_objects():
            collision = self.check_player_aabb_collision(obj.aabb)
            if collision:
                self.resolve_aabb_collision(obj)
        self.player.velocity += self.player.thrust
        self.apply_gravity(delta_time)

        # Apply speed limiter
        self.limit_speed()

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
        return -0.5 * self.world.air_density * caliber.drag_coefficient * caliber.bullet_area * speed ** 2 * glm.normalize(
            velocity)

    def is_out_of_bounds(self, position):
        # print("Projectile displacement = ", glm.length(position))
        if glm.length(position) > 512:
            print("Projectile out of bounds.")
            return True
        return False

    def check_player_aabb_collision(self, aabb2):
        player_bb = self.player.bounding_box

        # AABB1 min and max points
        min1_x, min1_y, min1_z = player_bb[0]
        max1_x, max1_y, max1_z = player_bb[1]

        # AABB2 min and max points
        min2_x, min2_y, min2_z = aabb2[0]
        max2_x, max2_y, max2_z = aabb2[1]

        # Check for overlap on the x-axis
        x_overlap = (min1_x <= max2_x) and (max1_x >= min2_x)

        # Check for overlap on the y-axis
        y_overlap = (min1_y <= max2_y) and (max1_y >= min2_y)

        # Check for overlap on the z-axis
        z_overlap = (min1_z <= max2_z) and (max1_z >= min2_z)

        if x_overlap and y_overlap and z_overlap:
            # If all overlaps are true, the AABBs are colliding
            return True
        return False

    def resolve_aabb_collision(self, obj):
        player_bb = self.player.bounding_box
        # AABB1 min and max points
        min1_x, min1_y, min1_z = player_bb[0]
        max1_x, max1_y, max1_z = player_bb[1]
        aabb2 = obj.aabb
        # AABB2 min and max points
        min2_x, min2_y, min2_z = aabb2[0]
        max2_x, max2_y, max2_z = aabb2[1]
        x_depth = min(max1_x, max2_x) - max(min1_x, min2_x)
        y_depth = min(max1_y, max2_y) - max(min1_y, min2_y)
        z_depth = min(max1_z, max2_z) - max(min1_z, min2_z)

        # Case 1: Top-down collision
        if max1_y > max2_y:
            self.player.position.y += y_depth
            self.player.is_grounded = True
            return
        # Case 2: Bottom-up collision
        if min1_y < min2_y:
            self.player.position.y -= y_depth
            return
        # Case 3: Left-right collision
        if min1_x < min2_x:
            self.player.position.x -= x_depth
            return
        # Case 4: Right-left collision
        if max1_x > max2_x:
            self.player.position.x += x_depth
            return
        # Case 5: Front-rear collision
        if max1_z > max2_z:
            self.player.position.z += z_depth
            return
        # Case 6: Rear-front collision
        if min1_z < min2_z:
            self.player.position.z -= z_depth
            return
