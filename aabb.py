import glm

class AABB:
    def __init__(self, min_corner, max_corner):
        self.min = glm.vec3(min_corner)
        self.max = glm.vec3(max_corner)

    def overlaps(self, other):
        return (self.min.x <= other.max.x and self.max.x >= other.min.x and
                self.min.y <= other.max.y and self.max.y >= other.min.y and
                self.min.z <= other.max.z and self.max.z >= other.min.z)

    @staticmethod
    def calculate_aabb(vertices, model_matrix):
        min_corner = glm.vec3(float('inf'))
        max_corner = glm.vec3(float('-inf'))
        for vertex in vertices:
            transformed_vertex = glm.vec3(model_matrix * glm.vec4(vertex, 1.0))
            min_corner = glm.min(min_corner, transformed_vertex)
            max_corner = glm.max(max_corner, transformed_vertex)
        return AABB(min_corner, max_corner)
