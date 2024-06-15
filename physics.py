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
        #print("applying gravity")
        if not self.player.is_grounded:
            self.player.velocity += self.gravity * delta_time
            self.player.position += self.player.velocity * delta_time + 0.5 * self.gravity * delta_time ** 2
            self.set_gravity = True
        elif self.player.is_grounded:
            self.set_gravity = False

    def is_point_inside_convex_polyhedron(self, point: glm.vec3, convex_shape: List[glm.vec3]) -> bool:
        indices = range(0, len(convex_shape), 3)  # Assuming the convex_shape is a flattened list of triangles
        for i in indices:
            p1 = convex_shape[i]
            p2 = convex_shape[i + 1]
            p3 = convex_shape[i + 2]
            normal = glm.normalize(glm.cross(p2 - p1, p3 - p1))
            if glm.dot(normal, point - p1) > 0:
                print(f"Point {point} is outside the triangle formed by {p1}, {p2}, {p3}")
                return False
        return True

    def is_point_in_aabb(self, point, aabb):
        (min_x, min_y, min_z), (max_x, max_y, max_z) = aabb
        return (min_x <= point.x <= max_x) and (min_y <= point.y <= max_y) and (min_z <= point.z <= max_z)

    def resolve_collision_as_cylinder(self, obstacle_aabb):
        print('resolving collision as cylinder')
        (min_x, min_y, min_z), (max_x, max_y, max_z) = obstacle_aabb
        player_pos = self.player.position
        trajectory = glm.normalize(self.player.velocity)  # Direction of movement
        magnitude = glm.length(self.player.velocity)  # Magnitude of movement

        # Define cylinder parameters (adjust as needed)
        cylinder_radius = 0.5  # Half of player's width
        cylinder_height = 1.8  # Player's height

        iteration_count = 0
        max_iterations = 100  # Maximum iterations to avoid infinite loops

        while iteration_count < max_iterations:
            iteration_count += 1

            # Calculate player's cylinder base center position
            base_center_x = player_pos.x
            base_center_y = player_pos.y - cylinder_height / 2  # Bottom of the cylinder
            base_center_z = player_pos.z

            # Check for horizontal collision (circle intersection)
            closest_x = max(min_x, min(base_center_x, max_x))
            closest_z = max(min_z, min(base_center_z, max_z))
            distance_x = base_center_x - closest_x
            distance_z = base_center_z - closest_z
            collision_horizontal = (distance_x ** 2 + distance_z ** 2) < (cylinder_radius ** 2)

            # Check for vertical collision
            collision_vertical = (base_center_y < max_y and (base_center_y + cylinder_height) > min_y)

            if collision_horizontal and collision_vertical:
                # Calculate penetration depth
                penetration_depth_x = min(max_x - base_center_x, base_center_x - min_x)
                penetration_depth_y = min(max_y - base_center_y, (base_center_y + cylinder_height) - min_y)
                penetration_depth_z = min(max_z - base_center_z, base_center_z - min_z)

                # Resolve collision by moving along the trajectory vector
                if penetration_depth_x <= penetration_depth_y and penetration_depth_x <= penetration_depth_z:
                    # Resolve along x-axis
                    resolve_vector = glm.vec3(trajectory.x, 0, 0)
                elif penetration_depth_y <= penetration_depth_x and penetration_depth_y <= penetration_depth_z:
                    # Resolve along y-axis
                    resolve_vector = glm.vec3(0, trajectory.y, 0)
                else:
                    # Resolve along z-axis
                    resolve_vector = glm.vec3(0, 0, trajectory.z)

                resolve_vector = glm.normalize(resolve_vector) * min(penetration_depth_x, penetration_depth_y,
                                                                     penetration_depth_z)
                player_pos += resolve_vector

                print(f"Iteration {iteration_count}: Resolved collision by moving to {player_pos}")
            else:
                print(f"Iteration {iteration_count}: No collision detected")
                break

        # Update the player's position
        self.player.position = player_pos
        print(f"Final player position after resolving collision: {self.player.position}")

    def resolve_player_collision(self, obstacle_face, delta_time, penetration_threshold=1, correction_velocity=0.5):
        # Define player's bounding box corners
        player_height = 2
        player_width = 0.5
        player_position = self.player.position

        corners = [
            glm.vec3(player_position.x - player_width / 2, player_position.y, player_position.z - player_width / 2),
            glm.vec3(player_position.x + player_width / 2, player_position.y, player_position.z - player_width / 2),
            glm.vec3(player_position.x - player_width / 2, player_position.y, player_position.z + player_width / 2),
            glm.vec3(player_position.x + player_width / 2, player_position.y, player_position.z + player_width / 2),
            glm.vec3(player_position.x - player_width / 2, player_position.y + player_height,
                     player_position.z - player_width / 2),
            glm.vec3(player_position.x + player_width / 2, player_position.y + player_height,
                     player_position.z - player_width / 2),
            glm.vec3(player_position.x - player_width / 2, player_position.y + player_height,
                     player_position.z + player_width / 2),
            glm.vec3(player_position.x + player_width / 2, player_position.y + player_height,
                     player_position.z + player_width / 2)
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
            else:
                print('Intersection point out of bounds')

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
                # Adjust plane normal if necessary based on player velocity
                if (plane_normal.x > 0 > self.player.velocity.x) or (plane_normal.x < 0 < self.player.velocity.x):
                    plane_normal.x = -plane_normal.x
                if (plane_normal.z > 0 > self.player.velocity.z) or (plane_normal.z < 0 < self.player.velocity.z):
                    plane_normal.z = -plane_normal.z

                # Apply less correction for horizontal collisions, clamped to a max value
                horizontal_correction = min(correction, max_correction)
                self.player.position -= plane_normal * horizontal_correction

            # Apply corrections based on collision type
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

        for obj in self.world.get_objects():
            aabb = obj.aabb
            if self.is_line_segment_intersecting_aabb(start_pos, end_pos, aabb):
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

    def get_nearest_intersecting_face(self, start, end, aabb):
        """
        Given that a line segment intersects an AABB, determines the nearest intersecting face.
        Returns the coordinates of the nearest face as ((x1, y1, z1), (x2, y2, z2), (x3, y3, z3)).
        """
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

            ray_direction = glm.normalize(end - start)
            ndotu = glm.dot(plane_normal, ray_direction)
            if abs(ndotu) < 1e-6:
                continue

            w = start - plane_point
            si = -glm.dot(plane_normal, w) / ndotu
            intersection_point = start + si * ray_direction

            if (min_x <= intersection_point.x <= max_x and
                    min_y <= intersection_point.y <= max_y and
                    min_z <= intersection_point.z <= max_z):
                distance = glm.distance(start, intersection_point)
                intersection_distances[face] = (distance, vertices)

        if not intersection_distances:
            return None

        nearest_face = min(intersection_distances, key=lambda k: intersection_distances[k][0])
        return intersection_distances[nearest_face][1]

    def update_physics(self, delta_time: float):
        """
        Updates the game state, applying gravity if no collision is detected, and resolving collisions if detected.
        """
        obstacle_face = self.check_linear_collision()
        if obstacle_face:
            self.resolve_player_collision(obstacle_face, delta_time)
        else:
            self.apply_gravity(delta_time)