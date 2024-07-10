import os
from dataclasses import dataclass
import random

import numpy as np
import glm
from typing import List, Tuple
from OpenGL.GL import *
from utils.file_utils import get_relative_path
from scipy.spatial import ConvexHull

@dataclass
class MaterialOverride:
    kd_override: glm.vec3
    ks_override: glm.vec3
    ns_override: float


class Model:
    default_material = {
        'diffuse': [1.0, 0.0, 0.0],  # Example values
        'specular': [1.0, 1.0, 1.0],  # Example values (or a glm.vec3 if using glm)
        'shininess': 10.0,  # Example value
        'roughness': 0.1,  # Example value for roughness
        'bumpScale': 10.0
    }

    def __init__(self,
                 filepath: str,
                 mtl_filepath: str,
                 player=False,
                 draw_convex_only=False,
                 rotation_angles=glm.vec3(0.0, 0.0, 0.0),
                 translation=glm.vec3(0.0, 0.0, 0.0),
                 kd_override=None,
                 ks_override=None,
                 ns_override=None,
                 scale=1,
                 is_collidable=True,
                 shift_to_centroid=False,
                 roughness_override=None,
                 bump_scale_override=None):

        self.initial_position = translation
        self.impact = False
        self.script_dir = os.path.dirname(os.path.dirname(__file__))

        self.scale = scale
        self.position = translation
        self.orientation = rotation_angles
        self.init_model_matrix(translation, rotation_angles)

        self.is_collidable = is_collidable
        print(f"Initializing Model with filepath: {filepath}")
        self.name = filepath.split('/')[-1].split('.')[0]

        self.shift_to_centroid = shift_to_centroid
        self.vertices, self.indices = self.load_obj(filepath, shift_to_centroid=self.shift_to_centroid)
        self.composite_centroid = self.calculate_centroid()

        self.materials = self.load_mtl(mtl_filepath)
        # Apply overrides if provided
        first_material_key = next(iter(self.materials))
        if kd_override is not None:
            print(f'{self.name}, kd: {kd_override}')
            self.materials[first_material_key]['diffuse'] = kd_override
        if ks_override is not None:
            print(f'{self.name}, ks: {ks_override}')
            self.materials[first_material_key]['specular'] = ks_override
        if ns_override is not None:
            print(f'{self.name}, ns: {ns_override}')
            self.materials[first_material_key]['shininess'] = ns_override
        if roughness_override is not None:
            print(f'{self.name}, roughness: {roughness_override}')
            self.materials[first_material_key]['roughness'] = roughness_override
        if bump_scale_override is not None:
            print(f'{self.name}, bumpScale: {bump_scale_override}')
            self.materials[first_material_key]['bumpScale'] = bump_scale_override

        self.default_material = self.materials[first_material_key]
        self.vao, self.vbo, self.ebo = self.setup_buffers()
        self.is_player = player
        self.set_scale(scale)
        self.set_position(translation)
        self.draw_convex_only = draw_convex_only
        self.centroid = self.calculate_centroid()
        if self.is_player:
            self.bounding_box = self.calculate_bounding_box()
            self.aabb = self.calculate_aabb()
        else:
            if self.is_collidable:
                self.bounding_box = self.calculate_bounding_box()
                self.aabb = self.calculate_aabb()
                self.voxels = None
                self.voxel_size = 5
        print(f"{self.name}'s Materials: {self.materials} ")
        print()
        if kd_override is not None:
            self.materials['diffuse'] = kd_override
        if ks_override is not None:
            self.materials['specular'] = ks_override
        if ns_override is not None:
            self.materials['shininess'] = ns_override
        if roughness_override is not None:
            self.materials['roughness'] = roughness_override
        if bump_scale_override is not None:
            self.materials['bumpScale'] = bump_scale_override

    def convex_decomposition(self, vertices):
        def quickhull(vertices):
            hull = ConvexHull(vertices)
            return hull.vertices, hull.simplices

        def split_non_convex(vertices):
            # This function should implement the logic to split non-convex parts
            # This is a placeholder for the actual implementation
            # The implementation might include finding a plane and splitting the vertices into two sets
            pass

        def is_convex(vertices):
            # Check if a set of vertices forms a convex shape
            hull = ConvexHull(vertices)
            return len(hull.vertices) == len(vertices)
        # Step 1: Compute the initial convex hull
        convex_parts = []
        hull_vertices, hull_simplices = quickhull(vertices)
        convex_parts.append(vertices[hull_vertices])

        # Step 2: Decompose non-convex parts
        non_convex_parts = [vertices]

        while non_convex_parts:
            part = non_convex_parts.pop()
            if not is_convex(part):
                subparts = split_non_convex(part)
                non_convex_parts.extend(subparts)
            else:
                convex_parts.append(part)

        return convex_parts

    def calculate_centroid(self):
        transformed_vertices = []
        for i in range(0, len(self.vertices), 6):
            local_vertex = glm.vec3(self.vertices[i], self.vertices[i + 1], self.vertices[i + 2])
            transformed_vertices.append(local_vertex)

        x_coords = [vertex.x for vertex in transformed_vertices]
        y_coords = [vertex.y for vertex in transformed_vertices]
        z_coords = [vertex.z for vertex in transformed_vertices]

        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        centroid_z = np.mean(z_coords)

        return glm.vec3(centroid_x, centroid_y, centroid_z)

    def load_obj(self, relative_filepath: str, shift_to_centroid=False) -> Tuple[np.ndarray, np.ndarray]:
        filepath = get_relative_path(relative_filepath)
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

        if shift_to_centroid:
            print(f"shifting {self.name} to centroid")
            x_coords = [vertex[0] for vertex in vertices]
            z_coords = [vertex[2] for vertex in vertices]
            centroid_x = sum(x_coords) / len(x_coords)
            centroid_z = sum(z_coords) / len(z_coords)
            for i in range(len(vertices)):
                vertices[i][0] -= centroid_x
                vertices[i][2] -= centroid_z

        vertex_data = []
        for face in faces:
            for vertex_index, normal_index in face:
                vertex_data.extend(vertices[vertex_index])
                vertex_data.extend(normals[normal_index])

        vertex_data = np.array(vertex_data, dtype=np.float32)
        indices = np.arange(len(vertex_data) // 6, dtype=np.uint32)
        return vertex_data, indices

    def load_mtl(self, mtl_filepath: str) -> dict:
        materials = {}
        current_material = None

        with open(mtl_filepath, 'r') as file:
            for line in file:
                if line.startswith('newmtl'):
                    current_material = line.split()[1]
                    materials[current_material] = {'diffuse': [1.0, 0.0, 0.0], 'specular': [0.1, 0.5, 1.0],
                                                   'shininess': 10.0, 'roughness': 0.2, 'bumpScale': 0.0}
                elif current_material:
                    if line.startswith('Kd '):
                        parts = line.split()
                        materials[current_material]['diffuse'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                    elif line.startswith('Ks '):
                        parts = line.split()
                        materials[current_material]['specular'] = [float(parts[1]), float(parts[2]), float(parts[3])]
                    elif line.startswith('Ns '):
                        materials[current_material]['shininess'] = float(line.split()[1])

        print('Parsed materials:', materials)
        return materials

    def setup_buffers(self) -> Tuple[int, int, int]:
        assert self.vertices is not None and len(self.vertices) > 0, "Vertices data should not be empty."
        assert self.indices is not None and len(self.indices) > 0, "Indices data should not be empty."

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

        # Verify the buffer sizes
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        vbo_size = glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE)
        assert vbo_size == self.vertices.nbytes, f"VBO size {vbo_size} does not match vertices data size {self.vertices.nbytes}."

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        ebo_size = glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE)
        assert ebo_size == self.indices.nbytes, f"EBO size {ebo_size} does not match indices data size {self.indices.nbytes}."

        glBindVertexArray(0)

        return vao, vbo, ebo

    def update_buffers(self, vao: int, vbo: int, ebo: int) -> None:
        def check_gl_error():
            error = glGetError()
            if error != GL_NO_ERROR:
                print(f"OpenGL error: {error}")
                assert error == GL_NO_ERROR, f"OpenGL error occurred: {error}"

        print("Updating buffers...")
        print(f"Vertices size: {self.vertices.nbytes}")
        print(f"Indices size: {self.indices.nbytes}")

        assert self.vertices is not None and len(self.vertices) > 0, "Vertices data should not be empty."
        assert self.indices is not None and len(self.indices) > 0, "Indices data should not be empty."

        glBindVertexArray(vao)
        check_gl_error()

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        check_gl_error()
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        check_gl_error()

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        check_gl_error()
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        check_gl_error()

        stride = 6 * self.vertices.itemsize
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        check_gl_error()
        glEnableVertexAttribArray(0)
        check_gl_error()
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * self.vertices.itemsize))
        check_gl_error()
        glEnableVertexAttribArray(1)
        check_gl_error()

        glBindVertexArray(0)
        check_gl_error()

        # Verify the buffer sizes
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        vbo_size = glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE)
        assert vbo_size == self.vertices.nbytes, f"VBO size {vbo_size} does not match vertices data size {self.vertices.nbytes}."

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        ebo_size = glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE)
        assert ebo_size == self.indices.nbytes, f"EBO size {ebo_size} does not match indices data size {self.indices.nbytes}."

        print(self.name, "vao = ", self.vao, " vbo = ", self.vbo, " ebo = ", self.ebo)

    def draw(self, camera=None):
        assert self.vao is not None, "VAO should not be None."
        assert len(self.indices) > 0, "Indices should not be empty."
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        error = glGetError()
        if error != GL_NO_ERROR:
            print(f"OpenGL error during draw: {error}")
            assert error == GL_NO_ERROR, f"OpenGL error occurred during draw: {error}"

    def set_scale(self, scale):
        print("setting scale for ", self.name, " to ", scale)
        if self.model_matrix is None:
            raise ValueError("model_matrix is not initialized")

        if not isinstance(scale, (int, float)):
            raise ValueError("scale must be a numeric value")

        try:
            self.model_matrix = glm.scale(self.model_matrix, glm.vec3(scale, scale, scale))
        except Exception as e:
            raise RuntimeError(f"Failed to scale model_matrix: {e}")

    def set_position(self, translation):
        self.position = glm.vec3(translation[0], translation[1], translation[2])
        self.update_model_matrix()

    def set_orientation(self, rotation, pivot_point=None):
        # Convert rotation to glm.vec3
        rotation_vec = glm.vec3(rotation[0], rotation[1], rotation[2])

        if pivot_point is not None:
            self.set_position(-pivot_point)
            self.orientation = glm.vec3(rotation_vec)
            self.update_model_matrix()
            self.set_position(pivot_point)

        # Apply the rotation to the orientation
        self.orientation = glm.vec3(rotation_vec)

        # Update the model matrix to reflect new position and orientation
        self.update_model_matrix()

    def calculate_bounding_box(self, bounding_margin=0.0) -> list:
        positions = self.vertices.reshape(-1, 6)[:, :3]

        if positions.size == 0:
            print("Warning: No vertices found, cannot compute bounding box.")
            return []

        min_x, min_y, min_z = np.min(positions, axis=0)
        max_x, max_y, max_z = np.max(positions, axis=0)

        # Add margin to the bounding box dimensions
        min_x -= bounding_margin
        min_y -= bounding_margin
        min_z -= bounding_margin
        max_x += bounding_margin
        max_y += bounding_margin
        max_z += bounding_margin

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

        # Transform bounding box vertices by the model matrix
        transformed_bounding_box = []
        for vertex in bounding_box:
            vec4_vertex = glm.vec4(vertex, 1.0)
            transformed_vertex = glm.vec3(self.model_matrix * vec4_vertex)
            transformed_bounding_box.append(transformed_vertex)

        print("Calculated bounding box:", transformed_bounding_box)
        return transformed_bounding_box

    def calculate_aabb(self):
        bounding_box = self.calculate_bounding_box()
        if not bounding_box:
            return (0, 0, 0), (0, 0, 0)  # Return a default AABB if the bounding box is empty

        min_x = min(bounding_box, key=lambda v: v.x).x
        max_x = max(bounding_box, key=lambda v: v.x).x
        min_y = min(bounding_box, key=lambda v: v.y).y
        max_y = max(bounding_box, key=lambda v: v.y).y
        min_z = min(bounding_box, key=lambda v: v.z).z
        max_z = max(bounding_box, key=lambda v: v.z).z

        aabb = (min_x, min_y, min_z), (max_x, max_y, max_z)
        print("Calculated AABB:", aabb)
        return aabb

    def update_model_matrix(self, parent_matrix=None, debug=False):
        # Create translation matrix for the object's position
        translation_matrix = glm.translate(glm.mat4(1.0), self.position)

        # Create rotation matrices for the object's orientation
        rotation_x = glm.rotate(glm.mat4(1.0), glm.radians(self.orientation[0]), glm.vec3(1.0, 0.0, 0.0))
        rotation_y = glm.rotate(glm.mat4(1.0), glm.radians(self.orientation[1]), glm.vec3(0.0, 1.0, 0.0))
        rotation_z = glm.rotate(glm.mat4(1.0), glm.radians(self.orientation[2]), glm.vec3(0.0, 0.0, 1.0))

        # Combine rotations to form the object's rotation matrix
        rotation_matrix = rotation_z * rotation_y * rotation_x

        # Create scaling matrix for the object's scale
        scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(self.scale, self.scale, self.scale))

        # Combine translation, rotation, and scale to form the local model matrix
        local_model_matrix = translation_matrix * rotation_matrix * scale_matrix

        # Handle case where there is no parent matrix
        if parent_matrix is None:
            self.model_matrix = local_model_matrix
        else:
            self.model_matrix = parent_matrix * local_model_matrix

        if debug:
            # print("local model matrix: \n", local_model_matrix)
            print(f"{self.name}'s model matrix updated to:\n{self.model_matrix}")

    def init_model_matrix(self, translation, rotation_angles):
        translation_matrix = glm.translate(glm.mat4(1.0), translation)

        rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(rotation_angles[0]), glm.vec3(1.0, 0.0, 0.0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[1]), glm.vec3(0.0, 1.0, 0.0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[2]), glm.vec3(0.0, 0.0, 1.0))

        scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(self.scale, self.scale, self.scale))

        self.model_matrix = translation_matrix * rotation_matrix * scale_matrix

    def add_crater(self, impact_point, crater_radius, crater_depth):
        # TODO: Implement this
        self.impact = True
        vertices_2d = self.vertices.reshape(-1, 3)

        print("Vertices before crater addition:")
        print(vertices_2d)

        num_crater_vertices = 12
        angle_step = 360.0 / num_crater_vertices

        crater_vertices = []
        for i in range(num_crater_vertices):
            angle = i * angle_step
            radian = glm.radians(angle)
            x = impact_point.x + crater_radius * random.uniform(0.8, 1.2) * glm.cos(radian)
            y = impact_point.y + crater_radius * random.uniform(0.8, 1.2) * glm.sin(radian)
            z = impact_point.z - random.uniform(crater_depth * 0.8, crater_depth * 1.2)
            crater_vertices.append([x, y, z])

        crater_center = [impact_point.x, impact_point.y, impact_point.z - crater_depth]
        crater_vertices.append(crater_center)

        crater_vertices = np.array(crater_vertices)
        vertices_2d = np.vstack((vertices_2d, crater_vertices))

        print("Crater vertices:")
        print(crater_vertices)

        self.vertices = vertices_2d.reshape(-1)

        print("Vertices after crater addition:")
        print(vertices_2d)

        crater_start_index = len(vertices_2d) - len(crater_vertices)
        crater_center_index = crater_start_index + num_crater_vertices
        crater_faces = []
        for i in range(num_crater_vertices):
            next_index = crater_start_index + (i + 1) % num_crater_vertices
            current_index = crater_start_index + i
            crater_faces.append([current_index, next_index, crater_center_index])

        crater_faces = np.array(crater_faces)

        print("Crater faces:")
        print(crater_faces)

        assert self.indices is not None, "Indices data should not be None."

        if self.indices.ndim == 1:
            assert self.indices.size % 3 == 0, "Indices size is not a multiple of 3 and cannot be reshaped."
            self.indices = self.indices.reshape(-1, 3)

        print("Indices before vstack:")
        print(self.indices)

        if self.indices.size == 0:
            print("Warning: self.indices is empty or not initialized. Initializing with crater_faces.")
            self.indices = crater_faces
        else:
            if crater_faces.shape[1] < self.indices.shape[1]:
                extra_columns = self.indices.shape[1] - crater_faces.shape[1]
                extra_data = np.zeros((crater_faces.shape[0], extra_columns), dtype=crater_faces.dtype)
                crater_faces = np.hstack((crater_faces, extra_data))
            elif crater_faces.shape[1] > self.indices.shape[1]:
                extra_columns = crater_faces.shape[1] - self.indices.shape[1]
                extra_data = np.zeros((self.indices.shape[0], extra_columns), dtype=self.indices.dtype)
                self.indices = np.hstack((self.indices, extra_data))

            self.indices = np.vstack((self.indices, crater_faces))

        print("Indices after vstack:")
        print(self.indices)

        assert self.vertices.size % 3 == 0, "Vertices size is not a multiple of 3 after crater addition."
        assert self.indices.size % 3 == 0, "Indices size is not a multiple of 3 after crater addition."

        self.update_buffers(self.vao, self.vbo, self.ebo)

    def translate_vertices(self, translation: glm.vec3):
        """
        Translate the vertices of the model by the given translation vector.

        :param translation: A glm.vec3 vector representing the translation.
        """
        translation_array = np.array([translation.x, translation.y, translation.z], dtype=np.float32)

        # Reshape the vertices array to (N, 3) to apply the translation
        reshaped_vertices = self.vertices.reshape(-1, 3)
        reshaped_vertices += translation_array

        # Flatten back to 1D array
        self.vertices = reshaped_vertices.flatten()
        print(f"Vertices translated by {translation_array}")
        self.update_buffers(self.vao, self.vbo, self.ebo)

    def update_transformation_matrix(self, rotation_angle, rotation_axis):
        """
        Update the transformation (model) matrix based on the rotation angle around the rotation axis.
        """

        # Normalize the rotation axis
        axis = glm.normalize(rotation_axis)

        # Create the rotation matrix
        rotation_matrix = glm.rotate(glm.mat4(1.0), rotation_angle, axis)

        # Create the translation matrices
        translation_matrix = glm.translate(glm.mat4(1.0), self.position)
        inverse_translation_matrix = glm.translate(glm.mat4(1.0), -self.position)

        # Combine translation and rotation into the transformation matrix
        self.model_matrix = translation_matrix * rotation_matrix * inverse_translation_matrix
        self.orientation = rotation_angle * glm.vec3(rotation_axis)
