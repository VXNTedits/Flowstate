import numpy as np
import glm
from typing import List, Tuple
from OpenGL.GL import *

class Model:

    def __init__(self, filepath: str, player=False):
        print(f"Initializing Model with filepath: {filepath}")
        self.vertices, self.indices = self.load_obj(filepath)
        self.vao, self.vbo, self.ebo = self.setup_buffers()
        self.model_matrix = glm.mat4(1.0)  # Initialize model matrix
        self.vertices, self.indices = self.load_obj(filepath)
        self.static_normals = self.compute_edges(self.vertices.reshape(-1, 3))
        self.is_player = player
        if self.is_player:
            self.convex_components_list, self.convex_components_obj = self.decompose_model()
        else:
            self.convex_components = self.decompose_model()


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

    def decompose_model(self) -> List['Model']:
        if not self.is_player:
            # Extract only the positional data for convex decomposition
            positions = self.vertices.reshape(-1, 6)[:, :3]
            positions = [glm.vec3(x, y, z) for x, y, z in positions]

            shapes = []
            for i in range(0, len(self.indices), 3):
                triangle = [positions[self.indices[i]],
                            positions[self.indices[i + 1]],
                            positions[self.indices[i + 2]]]
                shapes.append(triangle)

            convex_shapes = []
            for shape in shapes:
                convex_shapes.extend(self.decompose_to_convex(shape))
            return convex_shapes
        else:
            # Basic bounding box calculation for simple rectangular object
            min_x, min_y, min_z = np.min(self.vertices.reshape(-1, 6)[:, :3], axis=0)
            max_x, max_y, max_z = np.max(self.vertices.reshape(-1, 6)[:, :3], axis=0)

            # Define the 8 corners of the bounding box
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

            # Create two triangles for each face of the bounding box (12 triangles in total)
            triangles = [
                [bounding_box[0], bounding_box[1], bounding_box[2]],
                [bounding_box[0], bounding_box[2], bounding_box[3]],
                [bounding_box[4], bounding_box[5], bounding_box[6]],
                [bounding_box[4], bounding_box[6], bounding_box[7]],
                [bounding_box[0], bounding_box[1], bounding_box[5]],
                [bounding_box[0], bounding_box[5], bounding_box[4]],
                [bounding_box[2], bounding_box[3], bounding_box[7]],
                [bounding_box[2], bounding_box[7], bounding_box[6]],
                [bounding_box[0], bounding_box[3], bounding_box[7]],
                [bounding_box[0], bounding_box[7], bounding_box[4]],
                [bounding_box[1], bounding_box[2], bounding_box[6]],
                [bounding_box[1], bounding_box[6], bounding_box[5]]
            ]

            # Convert triangles to Models
            #bounding_shapes = [Model.from_vertices(triangle) for triangle in triangles]
            #return bounding_shapes
            return triangles, [Model.from_vertices(triangle) for triangle in triangles]

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
            normal = glm.normalize(glm.vec3(-edge[1], edge[0], 0))  # Perpendicular vector in 2D
            normals.append(normal)
        return normals


    def is_convex(self, vertices: List[glm.vec3]) -> bool:
        edges = self.compute_edges(vertices)
        normals = self.get_normals(edges)
        signs = []
        for i in range(len(vertices)):
            v1 = vertices[i] - vertices[i - 1]
            v2 = vertices[(i + 1) % len(vertices)] - vertices[i]
            cross_product = glm.cross(v1, v2)
            signs.append(cross_product.z > 0)
        return all(signs) or not any(signs)

    def is_ear(self, vertices: List[glm.vec3], prev: int, curr: int, next: int) -> bool:
        triangle = [vertices[prev], vertices[curr], vertices[next]]
        cross1 = glm.cross(glm.vec3(triangle[1] - triangle[0], 0), glm.vec3(triangle[2] - triangle[1], 0))
        cross2 = glm.cross(glm.vec3(triangle[2] - triangle[1], 0), glm.vec3(triangle[0] - triangle[2], 0))
        cross3 = glm.cross(glm.vec3(triangle[0] - triangle[2], 0), glm.vec3(triangle[1] - triangle[0], 0))
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

    def decompose_to_convex(self, vertices: List[glm.vec3]) -> List['Model']:
        shapes = [vertices]
        convex_shapes = []

        while shapes:
            shape = shapes.pop()
            if self.is_convex(shape):
                convex_shapes.append(Model.from_vertices(shape))
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
                    convex_shapes.append(Model.from_vertices([shape[prev], shape[ear], shape[next]]))

        return convex_shapes

    @classmethod
    def from_vertices(cls, vertices: List[glm.vec3]) -> 'Model':
        obj = cls.__new__(cls)  # Create a new instance without calling __init__
        obj.vertices = np.array(vertices, dtype=np.float32).flatten()
        obj.indices = np.arange(len(vertices), dtype=np.uint32)
        obj.static_normals = obj.get_normals(obj.compute_edges(obj.vertices.reshape(-1, 3)))
        obj.convex_components = [obj]
        return obj

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
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

