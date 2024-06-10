import glm

from shape import Shape


def project_shape(shape, axis):
    min_proj = float('inf')
    max_proj = float('-inf')
    for vertex in shape.vertices:
        proj = glm.dot(vertex, axis)
        min_proj = min(min_proj, proj)
        max_proj = max(max_proj, proj)
    return min_proj, max_proj

def overlap(proj1, proj2):
    return not (proj1[1] < proj2[0] or proj2[1] < proj1[0])

def sat_collision(shape1, shape2):
    axes = shape1.get_normals() + shape2.get_normals()
    for axis in axes:
        proj1 = project_shape(shape1, axis)
        proj2 = project_shape(shape2, axis)
        if not overlap(proj1, proj2):
            return False
    return True

def check_collision(model1, model2):
    for shape1 in model1.convex_components:
        for shape2 in model2.convex_components:
            if sat_collision(shape1, shape2):
                return True
    return False



class AABB:
    def __init__(self, min_corner, max_corner):
        self.min = glm.vec3(min_corner)
        self.max = glm.vec3(max_corner)


    # Check for collisions between models
    def check_collision(model1, model2):
        for shape1 in model1.convex_components:
            for shape2 in model2.convex_components:
                if sat_collision(shape1, shape2):
                    return True
        return False

def decompose_to_convex(vertices, indices):
    shapes = [Shape(vertices, indices)]
    convex_shapes = []

    while shapes:
        shape = shapes.pop()
        if shape.is_convex():
            convex_shapes.append(shape)
        else:
            # Split the shape
            for i in range(len(shape.vertices)):
                if not shape.is_convex():
                    # Find the concave vertex
                    concave_vertex = shape.vertices[i]
                    for j in range(i + 2, len(shape.vertices) + i - 1):
                        # Create a new shape by splitting the original shape
                        new_shape_vertices = shape.vertices[i:j + 1] + shape.vertices[j:i - 1:-1]
                        new_shape = Shape(new_shape_vertices)
                        if new_shape.is_convex():
                            shapes.append(new_shape)
                            break

    return convex_shapes

