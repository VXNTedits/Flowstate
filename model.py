import numpy as np
import glm
from typing import List, Tuple
from OpenGL.GL import *

class Model:
    def __init__(self, filepath: str, player=False, draw_convex_only=False, rotation_angles=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0)):
        print(f"Initializing Model with filepath: {filepath}")
        self.vertices, self.indices = self.load_obj(filepath)
        self.vao, self.vbo, self.ebo = self.setup_buffers()
        self.model_matrix = glm.mat4(1.0)  # Initialize model matrix
        self.is_player = player
        self.set_orientation(rotation_angles)
        self.set_position(translation)
        self.draw_convex_only = draw_convex_only
        if self.is_player:
            self.bounding_box = self.calculate_bounding_box()#self.decompose_model()
        else:
            self.convex_components = self.decompose_model()
            self.bounding_box = self.calculate_bounding_box()
            self.voxels = self.decompose_to_voxels(self.vertices, 5)
            self.voxel_size = 5

    def decompose_to_voxels(self, vertices: np.ndarray, voxel_size: float) -> List[glm.vec3]:
        # Reshape vertices from flat array to list of vec3
        vertices = vertices.reshape(-1, 3)
        vertices_list = [glm.vec3(vertex[0], vertex[1], vertex[2]) for vertex in vertices]

        # Calculate the bounding box from the given vertices
        min_corner = glm.vec3(
            min(vertex.x for vertex in vertices_list),
            min(vertex.y for vertex in vertices_list),
            min(vertex.z for vertex in vertices_list)
        )
        max_corner = glm.vec3(
            max(vertex.x for vertex in vertices_list),
            max(vertex.y for vertex in vertices_list),
            max(vertex.z for vertex in vertices_list)
        )

        # Calculate the number of voxels along each axis
        voxel_count_x = int((max_corner.x - min_corner.x) / voxel_size) + 1
        voxel_count_y = int((max_corner.y - min_corner.y) / voxel_size) + 1
        voxel_count_z = int((max_corner.z - min_corner.z) / voxel_size) + 1

        voxels = []
        for x in range(voxel_count_x):
            for y in range(voxel_count_y):
                for z in range(voxel_count_z):
                    voxel_center = glm.vec3(
                        min_corner.x + x * voxel_size + voxel_size / 2,
                        min_corner.y + y * voxel_size + voxel_size / 2,
                        min_corner.z + z * voxel_size + voxel_size / 2
                    )
                    if self.is_point_inside_shape(voxel_center, vertices_list):
                        voxels.append(voxel_center)
        return voxels

    def is_point_inside_shape(self, point: glm.vec3, vertices: List[glm.vec3]) -> bool:
        # Use a ray-casting algorithm to determine if the point is inside the shape
        intersections = 0
        direction = glm.vec3(1, 0, 0)  # Arbitrary direction
        for i in range(0, len(vertices), 3):
            p1 = vertices[i]
            p2 = vertices[i + 1]
            p3 = vertices[i + 2]
            if self.ray_intersects_triangle(point, direction, p1, p2, p3):
                intersections += 1
        return intersections % 2 == 1

    def ray_intersects_triangle(self, origin: glm.vec3, direction: glm.vec3, v0: glm.vec3, v1: glm.vec3,
                                v2: glm.vec3) -> bool:
        # Möller–Trumbore ray-triangle intersection algorithm
        epsilon = 1e-8
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = glm.cross(direction, edge2)
        a = glm.dot(edge1, h)
        if -epsilon < a < epsilon:
            return False  # Ray is parallel to the triangle
        f = 1.0 / a
        s = origin - v0
        u = f * glm.dot(s, h)
        if u < 0.0 or u > 1.0:
            return False
        q = glm.cross(s, edge1)
        v = f * glm.dot(direction, q)
        if v < 0.0 or u + v > 1.0:
            return False
        t = f * glm.dot(edge2, q)
        if t > epsilon:
            return True  # Ray intersects the triangle
        else:
            return False  # Line intersection but not a ray intersection

    def load_obj(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        vertices = []
        normals = []
        faces = []

        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('vn '):
                    parts = line.split()
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.split()
                    face = []
                    for part in parts[1:]:
                        indices = part.split('/')
                        vertex_index = int(indices[0]) - 1
                        normal_index = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else vertex_index
                        face.append((vertex_index, normal_index))
                    faces.append(face)

        vertex_data = []
        for face in faces:
            for vertex_index, normal_index in face:
                vertex_data.extend(vertices[vertex_index])
                vertex_data.extend(normals[normal_index])

        vertex_data = np.array(vertex_data, dtype=np.float32)
        indices = np.arange(len(vertex_data) // 6, dtype=np.uint32)
        return vertex_data, indices

    def set_orientation(self, rotation_angles):
        # Apply rotations around x, y, z axes respectively
        rotation_matrix = glm.mat4(1.0)
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[0]), glm.vec3(1.0, 0.0, 0.0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[1]), glm.vec3(0.0, 1.0, 0.0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[2]), glm.vec3(0.0, 0.0, 1.0))
        self.model_matrix = rotation_matrix * self.model_matrix

    def set_position(self, translation):
        # Apply translation
        translation_matrix = glm.translate(glm.mat4(1.0), glm.vec3(translation[0], translation[1], translation[2]))
        self.model_matrix = translation_matrix * self.model_matrix

    def decompose_model(self) -> List['Model']:
        positions = self.vertices.reshape(-1, 6)[:, :3]
        positions = [glm.vec3(x, y, z) for x, y, z in positions]
        shapes = []
        for i in range(0, len(self.indices), 3):
            triangle = [positions[self.indices[i]], positions[self.indices[i + 1]], positions[self.indices[i + 2]]]
            shapes.append(triangle)

        convex_shapes = []
        for shape in shapes:
            decomposed = self.decompose_to_convex(shape)
            convex_shapes.extend(decomposed)

        return convex_shapes

    def calculate_bounding_box(self) -> List[glm.vec3]:
        positions = self.vertices.reshape(-1, 6)[:, :3]

        if positions.size == 0:
            print("Warning: No vertices found, cannot compute bounding box.")
            return []

        min_x, min_y, min_z = np.min(positions, axis=0)
        max_x, max_y, max_z = np.max(positions, axis=0)

        bounding_box = [
            glm.vec3(min_x, min_y, min_z),
            glm.vec3(max_x, min_y, min_z),
            glm.vec3(max_x, max_y, min_z),
            glm.vec3(min_x, max_y, min_z),
            glm.vec3(min_x, min_y, max_z),
            glm.vec3(max_x, min_y, max_z),
            glm.vec3(max_x, max_y, max_z),
            glm.vec3(min_x, max_y, max_z)
        ]
        return bounding_box

    def compute_edges(self, vertices: List[glm.vec3]) -> List[glm.vec3]:
        edges = []
        for i in range(len(vertices)):
            start = vertices[i]
            end = vertices[(i + 1) % len(vertices)]
            edge = end - start
            edges.append(edge)
        return edges

    def get_normals(self, edges: List[glm.vec3]) -> List[glm.vec3]:
        normals = []
        for edge in edges:
            normal = glm.normalize(glm.cross(edge, glm.vec3(0, 0, 1)))  # Assuming a default normal in Z direction
            normals.append(normal)
        return normals

    def is_convex(self, vertices: List[glm.vec3]) -> bool:
        num_vertices = len(vertices)

        # Ensure there are at least 4 vertices to form a 3D shape
        if num_vertices < 4:
            return False

        # Compute the normal of the first face to use as a reference
        v0 = vertices[0]
        v1 = vertices[1]
        v2 = vertices[2]
        reference_normal = glm.cross(v1 - v0, v2 - v0)

        # Check if all other faces have the same orientation relative to the reference normal
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                for k in range(j + 1, num_vertices):
                    if i != j and j != k and i != k:
                        vi = vertices[i]
                        vj = vertices[j]
                        vk = vertices[k]
                        current_normal = glm.cross(vj - vi, vk - vi)
                        if glm.dot(reference_normal, current_normal) < 0:
                            return False

        return True

    def decompose_to_convex(self, vertices: List[glm.vec3]) -> List[List[glm.vec3]]:
        shapes = [vertices]
        convex_shapes = []

        while shapes:
            shape = shapes.pop()
            if self.is_convex(shape):
                convex_shapes.append(shape)
            else:
                vertex_count = len(shape)
                if vertex_count < 3:
                    continue

                ears = []
                for i in range(vertex_count):
                    prev = (i - 1) % vertex_count
                    curr = i
                    next = (i + 1) % vertex_count
                    if self.is_ear(shape, prev, curr, next):
                        ears.append(curr)

                if not ears:
                    midpoint = vertex_count // 2
                    shapes.append(shape[:midpoint])
                    shapes.append(shape[midpoint:])
                else:
                    ear = ears[0]
                    prev = (ear - 1) % vertex_count
                    next = (ear + 1) % vertex_count
                    new_vertices = shape[:ear] + shape[ear + 1:]
                    shapes.append(new_vertices)
                    convex_shapes.append([shape[prev], shape[ear], shape[next]])

        return convex_shapes

    def is_ear(self, vertices: List[glm.vec3], prev: int, curr: int, next: int) -> bool:
        triangle = [vertices[prev], vertices[curr], vertices[next]]
        cross1 = glm.cross(triangle[1] - triangle[0], triangle[2] - triangle[1])
        cross2 = glm.cross(triangle[2] - triangle[1], triangle[0] - triangle[2])
        cross3 = glm.cross(triangle[0] - triangle[2], triangle[1] - triangle[0])
        if cross1.z <= 0 or cross2.z <= 0 or cross3.z <= 0:
            return False
        for i in range(len(vertices)):
            if i in [prev, curr, next]:
                continue
            if self.point_in_triangle(vertices[i], triangle):
                return False
        return True

    def point_in_triangle(self, pt: glm.vec3, tri: List[glm.vec3]) -> bool:
        def sign(p1: glm.vec3, p2: glm.vec3, p3: glm.vec3) -> float:
            return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)

        d1 = sign(pt, tri[0], tri[1])
        d2 = sign(pt, tri[1], tri[2])
        d3 = sign(pt, tri[2], tri[0])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def setup_buffers(self) -> Tuple[int, int, int]:
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        stride = 6 * self.vertices.itemsize
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * self.vertices.itemsize))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
        return vao, vbo, ebo

    def draw(self):
        if self.draw_convex_only:
            if not self.is_player:
                for component in self.convex_components:
                    glBindVertexArray(component.vao)
                    glDrawElements(GL_TRIANGLES, len(component.indices), GL_UNSIGNED_INT, None)
                    glBindVertexArray(0)
        else:
            glBindVertexArray(self.vao)
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
            glBindVertexArray(0)

