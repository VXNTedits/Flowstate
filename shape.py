from typing import List, Tuple

import glm

from model import Model


class Shape:
    def __init__(self, vertices: List[glm.vec3]):
        self.vertices = vertices
        self.edges = self.compute_edges()

    def compute_edges(self) -> List[glm.vec3]:
        edges = []
        for i in range(len(self.vertices)):
            start = self.vertices[i]
            end = self.vertices[(i + 1) % len(self.vertices)]
            edge = end - start
            edges.append(edge)
        return edges

    def get_normals(self) -> List[glm.vec3]:
        normals = []
        for edge in self.edges:
            normal = glm.normalize(glm.vec3(-edge.y, edge.x, 0))  # Perpendicular vector in 2D
            normals.append(normal)
        return normals

    def is_convex(self) -> bool:
        normals = self.get_normals()
        signs = []
        for i in range(len(self.vertices)):
            v1 = self.vertices[i] - self.vertices[i - 1]
            v2 = self.vertices[(i + 1) % len(self.vertices)] - self.vertices[i]
            cross_product = glm.cross(glm.vec3(v1, 0), glm.vec3(v2, 0))
            signs.append(cross_product.z > 0)
        return all(signs) or not any(signs)

    def is_ear(self, prev: int, curr: int, next: int) -> bool:
        triangle = [self.vertices[prev], self.vertices[curr], self.vertices[next]]
        cross1 = glm.cross(glm.vec3(triangle[1] - triangle[0], 0), glm.vec3(triangle[2] - triangle[1], 0))
        cross2 = glm.cross(glm.vec3(triangle[2] - triangle[1], 0), glm.vec3(triangle[0] - triangle[2], 0))
        cross3 = glm.cross(glm.vec3(triangle[0] - triangle[2], 0), glm.vec3(triangle[1] - triangle[0], 0))
        if cross1.z <= 0 or cross2.z <= 0 or cross3.z <= 0:
            return False
        for i in range(len(self.vertices)):
            if i in [prev, curr, next]:
                continue
            if self.point_in_triangle(self.vertices[i], triangle):
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


def decompose_to_convex(vertices: List[glm.vec3]) -> List[Shape]:
    shapes = [Shape(vertices)]
    convex_shapes = []

    while shapes:
        shape = shapes.pop()
        if shape.is_convex():
            convex_shapes.append(shape)
        else:
            vertex_count = len(shape.vertices)
            if vertex_count < 3:
                continue

            ears = []
            for i in range(vertex_count):
                prev = (i - 1) % vertex_count
                curr = i
                next = (i + 1) % vertex_count
                if shape.is_ear(prev, curr, next):
                    ears.append(curr)

            if not ears:
                midpoint = vertex_count // 2
                new_shape1 = Shape(shape.vertices[:midpoint])
                new_shape2 = Shape(shape.vertices[midpoint:])
                shapes.append(new_shape1)
                shapes.append(new_shape2)
            else:
                ear = ears[0]
                prev = (ear - 1) % vertex_count
                next = (ear + 1) % vertex_count
                new_vertices = shape.vertices[:ear] + shape.vertices[ear + 1:]
                new_shape = Shape(new_vertices)
                shapes.append(new_shape)
                triangle = Shape([shape.vertices[prev], shape.vertices[ear], shape.vertices[next]])
                convex_shapes.append(triangle)

    return convex_shapes


def project_shape(shape: Shape, axis: glm.vec3) -> Tuple[float, float]:
    min_proj = float('inf')
    max_proj = float('-inf')
    for vertex in shape.vertices:
        proj = glm.dot(vertex, axis)
        min_proj = min(min_proj, proj)
        max_proj = max(max_proj, proj)
    return min_proj, max_proj


def overlap(proj1: Tuple[float, float], proj2: Tuple[float, float]) -> bool:
    return not (proj1[1] < proj2[0] or proj2[1] < proj1[0])


def sat_collision(shape1: Shape, shape2: Shape) -> bool:
    axes = shape1.get_normals() + shape2.get_normals()
    for axis in axes:
        proj1 = project_shape(shape1, axis)
        proj2 = project_shape(shape2, axis)
        if not overlap(proj1, proj2):
            return False
    return True


def check_collision(model1: Model, model2: Model) -> bool:
    for shape1 in model1.convex_components:
        for shape2 in model2.convex_components:
            if sat_collision(shape1, shape2):
                return True
    return False