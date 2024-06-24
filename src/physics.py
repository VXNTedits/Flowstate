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

    def __init__(self, world_objects, player, interactables: list, world):
        self.max_velocity = 1
        # self.world = world_objects
        self.world = world
        self.player = player
        self.gravity = glm.vec3(0, -10, 0)
        self.interactables = interactables
        self.pid_controller = PIDController(kp=0.8, ki=0.0, kd=0.0)
        self.offset = 0.1
        self.set_gravity = True
        self.active_trajectories = []

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

    def check_projectile_collision(self, projectile_positions):
        """
        Checks if the projectile intersects with any object in the world.
        Returns the exact collision point if a collision is detected, otherwise None.
        """
        for i in range(len(projectile_positions) - 1):
            #print("Checking collisions on arc: ", projectile_positions)
            start_pos = projectile_positions[i]
            end_pos = projectile_positions[i + 1]
            for obj in self.world.get_world_objects():
                aabb = obj.aabb
                is_intersecting, nearest_face_name, nearest_face_vectors = (
                    self.is_bounding_box_intersecting_aabb((start_pos, end_pos), aabb))

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
        """
        Check if a line segment intersects a plane and return the intersection point.
        """
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
        """
        Check if a point is within the bounds of the plane's rectangular area.
        """
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
        # Apply lateral thrust for movement
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

        self.player.thrust.y = self.player.proposed_thrust.y

        # Ensure vertical velocity does not invert due to overshooting the deceleration
        if abs(self.player.thrust.y) < 0.01:
            self.player.thrust.y = 0.0

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
                adjusted_thrust.x = 0

        # Check and adjust y-axis collision
        temp_position_y = self.player.position + glm.vec3(0, proposed_thrust.y, 0) * delta_time
        if proposed_thrust.y != 0 and self.check_collision_with_proposed_movement(self.player.position,
                                                                                  temp_position_y):
            if abs(proposed_thrust.y) > tolerance:
                print('adjusted thrust to reduce y')
                adjusted_thrust.y *= adjustment_factor
            else:
                print('adjusted thrust to y = 0')
                adjusted_thrust.y = 0

        # Check and adjust z-axis collision
        temp_position_z = self.player.position + glm.vec3(0, 0, proposed_thrust.z) * delta_time
        if proposed_thrust.z != 0 and self.check_collision_with_proposed_movement(self.player.position,
                                                                                  temp_position_z):
            if abs(proposed_thrust.z) > tolerance:
                print('adjusted thrust to reduce z')
                adjusted_thrust.z *= adjustment_factor
            else:
                print('adjusted thrust to z = 0')
                adjusted_thrust.z = 0

        return adjusted_thrust

    def check_collision_with_proposed_movement(self, start_pos, end_pos):
        # Create a bounding box for the new position
        player_bb = self.player.calculate_player_bounding_box(start_pos, end_pos)
        for obj in self.world.get_world_objects():
            if self.is_bounding_box_intersecting_aabb(player_bb, obj.aabb):
                return True
        return False

    def validate_proposed_position(self):
        player_bb = self.player.calculate_player_bounding_box(self.player.position, self.player.proposed_thrust)
        if not player_bb or len(player_bb) != 2:
            print('Invalid bounding box:', player_bb)
            return False
        for obj in self.world.get_world_objects():
            if self.is_bounding_box_intersecting_aabb(player_bb, obj.aabb):
                print('Collision detected, rejecting proposed position')
                return False
        print('No collision, allowing proposed position')
        return True

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

        # PID_correction = self.pid_controller.calculate(dot_product, delta_time)

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

    def adjust_velocity_based_on_wall_collision(self, thrust_direction, delta_time, deceleration_multiplier=1.0):
        print("Checking whether to adjust velocity based on wall collision...")
        start_pos = self.player.position
        print("        Thrust direction = ", thrust_direction)
        max_distance = glm.length(thrust_direction)  # * delta_time
        print("        Max distance = ", max_distance)
        closest_distance = float('inf')
        closest_face = None
        closest_face_name = None
        direction_normalized = glm.normalize(thrust_direction)

        for obj in self.world.get_world_objects():
            aabb = obj.aabb
            (min_x, min_y, min_z), (max_x, max_y, max_z) = aabb

            tmin, tmax = 0.0, max_distance

            for i in range(3):
                if direction_normalized[i] != 0:
                    t1 = (aabb[0][i] - start_pos[i]) / direction_normalized[i]
                    t2 = (aabb[1][i] - start_pos[i]) / direction_normalized[i]

                    tmin_i = min(t1, t2)
                    tmax_i = max(t1, t2)

                    tmin = max(tmin, tmin_i)
                    tmax = min(tmax, tmax_i)

                    if tmin > tmax:
                        break

            if tmin > tmax or tmin > max_distance:
                continue

            intersection_point = start_pos + direction_normalized * tmin
            intersecting_face = None
            face_vectors = None

            if abs(intersection_point.x - min_x) < 1e-5:
                intersecting_face = "left"
                face_vectors = ((min_x, min_y, min_z), (min_x, max_y, min_z), (min_x, min_y, max_z))
            elif abs(intersection_point.x - max_x) < 1e-5:
                intersecting_face = "right"
                face_vectors = ((max_x, min_y, min_z), (max_x, max_y, min_z), (max_x, min_y, max_z))
            elif abs(intersection_point.z - min_z) < 1e-5:
                intersecting_face = "front"
                face_vectors = ((min_x, min_y, min_z), (max_x, min_y, min_z), (min_x, max_y, min_z))
            elif abs(intersection_point.z - max_z) < 1e-5:
                intersecting_face = "back"
                face_vectors = ((min_x, min_y, max_z), (max_x, min_y, max_z), (min_x, max_y, max_z))

            if intersecting_face and tmin < closest_distance:
                closest_distance = tmin
                closest_face = face_vectors
                closest_face_name = intersecting_face

        if closest_face is not None:
            print("    Closest face:", closest_face_name)
            wall_normal = glm.normalize(glm.cross(glm.vec3(closest_face[1]) - glm.vec3(closest_face[0]),
                                                  glm.vec3(closest_face[2]) - glm.vec3(closest_face[0]))
                                        )

            deceleration_factor = self.calculate_deceleration_factor(closest_distance,
                                                                     max_distance,
                                                                     min_distance=0.01,
                                                                     deceleration_multiplier=deceleration_multiplier)
            print("    Deceleration factor = ", glm.round(deceleration_factor))
            self.apply_deceleration(deceleration_factor,
                                    thrust_direction,
                                    wall_normal,
                                    delta_time,
                                    closest_distance,
                                    min_distance=0.1)

    def calculate_deceleration_factor(self, distance, max_distance, min_distance=0.5, deceleration_multiplier=1.0):
        if distance < min_distance:
            return 1.0  # Maximum deceleration when within min_distance

        normalized_distance = (distance - min_distance) / (max_distance - min_distance)
        deceleration = 1.0 - glm.exp(-normalized_distance * deceleration_multiplier)

        # Ensure the deceleration factor is clamped between 0 and 1
        return min(1.0, max(0.0, deceleration))

    def apply_deceleration(self, deceleration_factor, thrust_direction, wall_normal, delta_time, distance,
                           min_distance):
        print("    Applying deceleration...")
        print("        player initial velocity = ", glm.round(self.player.velocity))

        # Calculate the component of thrust in the direction of the wall normal
        normal_component = glm.dot(thrust_direction, wall_normal) * wall_normal

        # If the distance is less than min_distance, set thrust towards the wall to zero
        if distance < min_distance:
            print("        Minimum distance! Setting thrust to zero.")
            if glm.dot(thrust_direction, wall_normal) > 0:
                # Thrust is towards the wall, set it to zero
                normal_component = glm.vec3(0.0, 0.0, 0.0)
                thrust_direction = normal_component
        else:
            # Apply deceleration only to this normal component
            normal_component *= (1.0 - deceleration_factor)
            # Adjust thrust direction by removing the normal component
            thrust_direction -= normal_component
            print("        Computed thrust direction:", glm.round(thrust_direction))

        # Update player's thrust and velocity
        self.player.proposed_thrust += thrust_direction
        self.player.thrust = self.player.proposed_thrust
        self.player.velocity += self.player.thrust

        print("        Applied deceleration:", glm.round(glm.length(normal_component)))
        print("player corrected velocity = ", glm.round(self.player.velocity))
        print("_______________________")

    def update_physics(self, delta_time: float, weapons, player):
        # Apply forces to the player
        self.apply_forces(delta_time)

        # Manage collisions
        nearest_face_vectors = self.check_linear_collision()
        if nearest_face_vectors:
            # Collision: Resolve by projecting player onto collision face.
            #            Reject proposed thrust: Restrict entry velocity, but allow player to have exit velocity.
            self.handle_collisions(self.player.thrust, delta_time)
        else:
            # No collision: Accept the proposed thrust
            self.player.velocity += self.player.thrust
            # self.apply_gravity(delta_time)

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

            # if weapon.tracers:
            #     print("active tracers = \n", weapon.tracers)

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
