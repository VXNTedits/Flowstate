import sys
from collections import defaultdict
from typing import Tuple, List, Set
import glfw
import glm
from pid import PIDController
from model import Model


class Physics:
    EPSILON = 1e-6

    def __init__(self, world_objects, player, interactables: list, world):
        self.max_velocity = 1
        self.world = world_objects
        self.world = world
        self.player = player
        self.gravity = glm.vec3(0, -10, 0)
        self.interactables = interactables
        self.pid_controller = PIDController(kp=1, ki=0.0, kd=0.0)
        self.offset = 0.1
        self.set_gravity = True

    def apply_gravity(self, delta_time: float):
        if not self.player.is_grounded:
            self.player.velocity += self.gravity * delta_time
            self.player.position += self.player.velocity * delta_time + 0.5 * self.gravity * delta_time ** 2
            self.set_gravity = True
        elif self.player.is_grounded:
            self.set_gravity = False

    def resolve_player_collision(self, obstacle_face, delta_time, penetration_threshold=0.1, correction_velocity=0.5):
        # Define player's bounding box corners
        player_height = self.player.player_height
        player_width = self.player.player_width
        player_position = self.player.position
        bounding_margin = 0.1

        corners = [
            glm.vec3(player_position.x - player_width / 2 - bounding_margin, player_position.y,
                     player_position.z - player_width / 2 - bounding_margin),
            glm.vec3(player_position.x + player_width / 2 + bounding_margin, player_position.y,
                     player_position.z - player_width / 2 - bounding_margin),
            glm.vec3(player_position.x - player_width / 2 - bounding_margin, player_position.y,
                     player_position.z + player_width / 2 + bounding_margin),
            glm.vec3(player_position.x + player_width / 2 + bounding_margin, player_position.y,
                     player_position.z + player_width / 2 + bounding_margin),
            glm.vec3(player_position.x - player_width / 2 - bounding_margin,
                     player_position.y + player_height + bounding_margin,
                     player_position.z - player_width / 2 - bounding_margin),
            glm.vec3(player_position.x + player_width / 2 + bounding_margin,
                     player_position.y + player_height + bounding_margin,
                     player_position.z - player_width / 2 - bounding_margin),
            glm.vec3(player_position.x - player_width / 2 - bounding_margin,
                     player_position.y + player_height + bounding_margin,
                     player_position.z + player_width / 2 + bounding_margin),
            glm.vec3(player_position.x + player_width / 2 + bounding_margin,
                     player_position.y + player_height + bounding_margin,
                     player_position.z + player_width / 2 + bounding_margin)
        ]

        # Calculate plane normal
        plane_point = glm.vec3(obstacle_face[0])
        edge1 = glm.vec3(obstacle_face[1]) - glm.vec3(obstacle_face[0])
        edge2 = glm.vec3(obstacle_face[2]) - glm.vec3(obstacle_face[0])
        plane_normal = glm.normalize(glm.cross(edge2, edge1))

        # Check each corner for collisions
        for corner in corners:
            ray_origin = corner
            ray_direction = glm.normalize(self.player.trajectory)
            ndotu = glm.dot(plane_normal, ray_direction)

            if abs(ndotu) < 1e-9:
                continue

            w = ray_origin - plane_point
            si = -glm.dot(plane_normal, w) / ndotu
            intersection_point = ray_origin + si * ray_direction

            # Verify the intersection point lies within the obstacle face bounds
            if not self.is_point_in_triangle(intersection_point, plane_point, plane_point + edge1, plane_point + edge2):
                continue

            # Determine penetration depth and correction
            penetration_depth = glm.distance(ray_origin, intersection_point)
            if penetration_depth > penetration_threshold:
                continue

            error = penetration_depth
            correction = self.pid_controller.calculate(error, delta_time)

            # Clamp correction to avoid large jumps
            max_correction = 1.0
            correction = min(correction, max_correction)

            # Debug information
            print(f'Plane Normal: {plane_normal}')
            print(f'Intersection Point: {intersection_point}')
            print(f'Penetration Depth: {penetration_depth}')
            print(f'Correction: {correction}')

            # Check collision type
            is_vertical_collision = plane_normal.y > 0.5
            is_horizontal_collision = abs(plane_normal.x) > 0.5 or abs(plane_normal.z) > 0.5

            # Further filter horizontal collisions
            is_horizontal_collision = is_horizontal_collision and not is_vertical_collision

            # Determine if player is grounded
            self.player.is_grounded = is_vertical_collision and self.player.velocity.y <= 0.01

            if is_horizontal_collision:
                print('Horizontal collision')

                # Project the player's velocity onto the collision normal
                velocity_normal_component = glm.dot(self.player.velocity, plane_normal) * plane_normal

                # If the velocity component is directed towards the obstacle, negate it
                if glm.dot(velocity_normal_component, plane_normal) > 0:
                    self.player.velocity -= velocity_normal_component

                # Apply less correction for horizontal collisions, clamped to a max value
                horizontal_correction = correction
                self.player.position -= plane_normal * horizontal_correction

            elif is_vertical_collision:
                print('Vertical collision')
                self.player.position += plane_normal * correction
                if not self.player.is_jumping:
                    self.player.velocity.y = 0  # Reset vertical velocity when grounded

        # Final debug position
        print(f'Final Player Position: {self.player.position}')
        print('_____')

    def is_point_in_triangle(self, p, a, b, c):
        v0 = c - a
        v1 = b - a
        v2 = p - a
        dot00 = glm.dot(v0, v0)
        dot01 = glm.dot(v0, v1)
        dot02 = glm.dot(v0, v2)
        dot11 = glm.dot(v1, v1)
        dot12 = glm.dot(v1, v2)
        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        return (u >= 0) and (v >= 0) and (u + v < 1)

    def check_linear_collision(self):
        """
        Checks if the player's movement intersects with any object in the world.
        Returns the face on which the intersection occurred, or None if no collision is detected.
        """
        start_pos = self.player.previous_position
        end_pos = self.player.position
        player_bb = self.player.calculate_player_bounding_box(start_pos, end_pos)

        for obj in self.world.get_objects():
            aabb = obj.aabb
            if self.is_bounding_box_intersecting_aabb(player_bb, aabb):
                nearest_face = self.get_nearest_intersecting_face(start_pos, end_pos, aabb)
                if nearest_face:
                    return nearest_face
        return None

    def is_line_segment_intersecting_aabb(self, start, end, aabb):
        """
        Checks if a line segment (start to end) intersects with an AABB (Axis-Aligned Bounding Box).
        Returns True if there is an intersection, otherwise False.
        """
        (min_x, min_y, min_z), (max_x, max_y, max_z) = aabb

        def slab_check(p0, p1, min_b, max_b):
            """
            Checks intersection of the line segment with a slab along one axis.
            Returns the intersection intervals [t0, t1] along the axis.
            """
            inv_d = 1.0 / (p1 - p0) if (p1 - p0) != 0 else float('inf')
            t0 = (min_b - p0) * inv_d
            t1 = (max_b - p0) * inv_d
            if t0 > t1:
                t0, t1 = t1, t0
            return t0, t1

        tmin, tmax = 0.0, 1.0

        for axis in range(3):
            p0 = start[axis]
            p1 = end[axis]
            min_b = [min_x, min_y, min_z][axis]
            max_b = [max_x, max_y, max_z][axis]

            if p0 == p1:
                if p0 < min_b or p0 > max_b:
                    return False
            else:
                t0, t1 = slab_check(p0, p1, min_b, max_b)
                tmin = max(tmin, t0)
                tmax = min(tmax, t1)
                if tmin > tmax:
                    return False

        return True

    def get_nearest_intersecting_face(self, start_pos, end_pos, aabb):
        (min_x, min_y, min_z), (max_x, max_y, max_z) = aabb
        faces = {
            "left": ((min_x, min_y, min_z), (min_x, max_y, min_z), (min_x, min_y, max_z)),
            "right": ((max_x, min_y, min_z), (max_x, max_y, min_z), (max_x, min_y, max_z)),
            "bottom": ((min_x, min_y, min_z), (max_x, min_y, min_z), (min_x, min_y, max_z)),
            "top": ((min_x, max_y, min_z), (max_x, max_y, min_z), (min_x, max_y, max_z)),
            "front": ((min_x, min_y, min_z), (max_x, min_y, min_z), (min_x, max_y, min_z)),
            "back": ((min_x, min_y, max_z), (max_x, min_y, max_z), (min_x, max_y, max_z))
        }

        intersection_distances = {}

        for face, vertices in faces.items():
            plane_point = glm.vec3(vertices[0])
            edge1 = glm.vec3(vertices[1]) - glm.vec3(vertices[0])
            edge2 = glm.vec3(vertices[2]) - glm.vec3(vertices[0])
            plane_normal = glm.normalize(glm.cross(edge1, edge2))

            for corner in [(min_x, min_y, min_z), (max_x, min_y, min_z), (min_x, max_y, min_z),
                           (min_x, min_y, max_z), (max_x, max_y, max_z), (max_x, min_y, max_z),
                           (min_x, max_y, max_z), (max_x, max_y, min_z)]:
                ray_direction = glm.normalize(glm.vec3(corner) - plane_point)
                ndotu = glm.dot(plane_normal, ray_direction)
                if abs(ndotu) < 1e-6:
                    continue

                w = glm.vec3(corner) - plane_point
                si = -glm.dot(plane_normal, w) / ndotu
                intersection_point = glm.vec3(corner) + si * ray_direction

                if (min_x <= intersection_point.x <= max_x and
                        min_y <= intersection_point.y <= max_y and
                        min_z <= intersection_point.z <= max_z):
                    distance = glm.distance(glm.vec3(corner), intersection_point)
                    intersection_distances[face] = (distance, vertices)

        if not intersection_distances:
            return None

        nearest_face = min(intersection_distances, key=lambda k: intersection_distances[k][0])
        return intersection_distances[nearest_face][1]

    def is_bounding_box_intersecting_aabb(self, player_bb, aabb):
        (min_px, min_py, min_pz) = player_bb[0]
        (max_px, max_py, max_pz) = player_bb[1]
        (min_x, min_y, min_z), (max_x, max_y, max_z) = aabb

        # Check for overlap on each axis
        return (min_px <= max_x and max_px >= min_x and
                min_py <= max_y and max_py >= min_y and
                min_pz <= max_z and max_pz >= min_z)

    def apply_forces(self, delta_time: float):
        # Apply lateral thrust for movement
        if self.player.thrust.x != 0.0 or self.player.thrust.z != 0.0:
            # Calculate desired lateral velocity
            desired_velocity = glm.normalize(
                glm.vec3(self.player.thrust.x, 0.0, self.player.thrust.z)) * self.player.max_speed
            # Use a lerp function to smoothly interpolate towards the target velocity
            lerp_factor = 1 - glm.exp(-self.player.accelerator * delta_time)
            self.player.velocity.x = self.player.velocity.x * (1 - lerp_factor) + desired_velocity.x * lerp_factor
            self.player.velocity.z = self.player.velocity.z * (1 - lerp_factor) + desired_velocity.z * lerp_factor
        else:
            # Apply braking force to lateral movement
            deceleration = glm.exp(-self.player.accelerator * delta_time)
            self.player.velocity.x *= deceleration
            self.player.velocity.z *= deceleration
            # Ensure that the lateral velocity does not invert due to overshooting the deceleration
            if glm.length(glm.vec3(self.player.velocity.x, 0.0, self.player.velocity.z)) ** 2 < 0.01:
                self.player.velocity.x = 0.0
                self.player.velocity.z = 0.0

        # Ensure vertical velocity does not invert due to overshooting the deceleration
        if abs(self.player.velocity.y) < 0.01:
            self.player.velocity.y = 0.0
        self.apply_gravity(delta_time)
        #print(f"Updated Velocity: {self.player.velocity}")

    def adjust_thrust_to_avoid_collision(self, proposed_thrust, delta_time):
        # Start with the proposed thrust
        adjusted_thrust = glm.vec3(proposed_thrust)
        tolerance = 0.1  # Small margin of tolerance to avoid getting stuck
        adjustment_factor = 0.5

        # Check and adjust x-axis collision
        temp_position_x = self.player.position + glm.vec3(proposed_thrust.x, 0, 0) * delta_time
        if proposed_thrust.x != 0 and self.check_collision_with_proposed_movement(self.player.position,
                                                                                  temp_position_x):
            if abs(proposed_thrust.x) > tolerance:
                print('adjusted thrust to reduce x')
                adjusted_thrust.x *= adjustment_factor
            else:
                print('adjusted thrust to x = 0')
                obstacle_face = self.check_linear_collision()
                if obstacle_face:
                    self.resolve_player_collision(obstacle_face, delta_time)

        # Check and adjust y-axis collision
        temp_position_y = self.player.position + glm.vec3(0, proposed_thrust.y, 0) * delta_time
        if proposed_thrust.y != 0 and self.check_collision_with_proposed_movement(self.player.position,
                                                                                  temp_position_y):
            if abs(proposed_thrust.y) > tolerance:
                print('adjusted thrust to reduce y')
                adjusted_thrust.y *= adjustment_factor
            else:
                print('adjusted thrust to y = 0')
                obstacle_face = self.check_linear_collision()
                if obstacle_face:
                    self.resolve_player_collision(obstacle_face, delta_time)

        # Check and adjust z-axis collision
        temp_position_z = self.player.position + glm.vec3(0, 0, proposed_thrust.z) * delta_time
        if proposed_thrust.z != 0 and self.check_collision_with_proposed_movement(self.player.position,
                                                                                  temp_position_z):
            if abs(proposed_thrust.z) > tolerance:
                print('adjusted thrust to reduce z')
                adjusted_thrust.z *= adjustment_factor
            else:
                print('adjusted thrust to z = 0')
                obstacle_face = self.check_linear_collision()
                if obstacle_face:
                    self.resolve_player_collision(obstacle_face, delta_time)

        return adjusted_thrust

    def check_collision_with_proposed_movement(self, start_pos, end_pos):
        # Create a bounding box for the new position
        player_bb = self.player.calculate_player_bounding_box(start_pos, end_pos)
        for obj in self.world.get_objects():
            if self.is_bounding_box_intersecting_aabb(player_bb, obj.aabb):
                return True
        return False

    def validate_proposed_position(self):
        player_bb = self.player.calculate_player_bounding_box(self.player.position, self.player.proposed_thrust)
        if not player_bb or len(player_bb) != 2:
            print('Invalid bounding box:', player_bb)
            return False
        for obj in self.world.get_objects():
            if self.is_bounding_box_intersecting_aabb(player_bb, obj.aabb):
                print('Collision detected, rejecting proposed position')
                return False
        print('No collision, allowing proposed position')
        return True

    def update_physics(self, delta_time: float):
        # Apply forces to the player
        self.apply_forces(delta_time)

        # Propose new position based on current velocity
        proposed_thrust = self.player.velocity * delta_time
        proposed_position = self.player.position + proposed_thrust

        # Adjust thrust to avoid collisions
        adjusted_thrust = self.adjust_thrust_to_avoid_collision(proposed_thrust, delta_time)

        # Update the player's velocity based on the adjusted thrust
        self.player.velocity = adjusted_thrust / delta_time

        # Update the player's position based on the adjusted thrust
        self.player.position += adjusted_thrust * delta_time

        # Reset thrust after applying forces
        self.player.reset_thrust()
