from dataclasses import dataclass
import numpy as np
import glm
from typing import List, Tuple
from OpenGL.GL import *


@dataclass
class MaterialOverride:
    kd_override: glm.vec3
    ks_override: glm.vec3
    ns_override: float


class Model:
    default_material = {
        'diffuse': [1, 0.0, 0.0],  # Example values
        'specular': [1.0, 1.0, 1.0],  # Example values (or a glm.vec3 if using glm)
        'shininess': 10.0  # Example value
    }

    def __init__(self,
                 filepath: str,
                 mtl_filepath: str,
                 player=False,
                 draw_convex_only=False,
                 rotation_angles=[0.0, 0.0, 0.0],
                 translation=[0.0, 0.0, 0.0],
                 kd_override=None,
                 ks_override=None,
                 ns_override=None,
                 scale=1,
                 is_collidable=True,
                 shift_to_centroid=False):

        self.orientation = rotation_angles
        self.scale = scale
        self.position = translation
        self.model_matrix = self.init_model_matrix(translation, rotation_angles)
        self.update_model_matrix()  #glm.mat4(1.0)  # Initialize model matrix
        self.shift_to_centroid = shift_to_centroid
        self.is_collidable = is_collidable
        print(f"Initializing Model with filepath: {filepath}")
        self.name = filepath
        self.vertices, self.indices = self.load_obj(filepath, shift_to_centroid=self.shift_to_centroid)
        self.materials = self.load_mtl(mtl_filepath)  # if mtl_filepath else {}
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

        self.default_material = self.materials[first_material_key]
        self.vao, self.vbo, self.ebo = self.setup_buffers()
        self.is_player = player
        self.set_scale(scale)
        self.set_position(translation)
        self.draw_convex_only = draw_convex_only
        if self.is_player:
            #self.player_width = 0.5
            #self.player_height = 2
            self.bounding_box = self.calculate_bounding_box()
            self.aabb = self.calculate_aabb()
            #
        else:
            # self.convex_components = self.decompose_model()
            if self.is_collidable:
                self.bounding_box = self.calculate_bounding_box()
                self.aabb = self.calculate_aabb()
                self.voxels = None  # self.decompose_to_voxels(self.vertices, 5)
                self.voxel_size = 5
        print(f"{self.name}'s Materials: {self.materials} ")
        print()
        if kd_override is not None:
            self.materials['diffuse'] = kd_override
        if ks_override is not None:
            self.materials['specular'] = ks_override
        if ns_override is not None:
            self.materials['shininess'] = ns_override
        self.centroid = self.calculate_centroid()

    def calculate_centroid(self):
        # Ensure vertices are transformed by the model matrix to get world coordinates
        transformed_vertices = []
        for i in range(0, len(self.vertices), 6):
            local_vertex = glm.vec3(self.vertices[i], self.vertices[i + 1], self.vertices[i + 2])
            world_vertex = glm.vec3(self.model_matrix * glm.vec4(local_vertex, 1.0))
            transformed_vertices.append(world_vertex)

        # Extract x and z coordinates in world space
        x_coords = [vertex.x for vertex in transformed_vertices]
        z_coords = [vertex.z for vertex in transformed_vertices]

        # Calculate the average of x and z coordinates
        centroid_x = np.mean(x_coords)
        centroid_z = np.mean(z_coords)

        return glm.vec3(centroid_x, 0, centroid_z)

    def load_obj(self, filepath: str, shift_to_centroid=False) -> Tuple[np.ndarray, np.ndarray]:
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
                    materials[current_material] = {'diffuse': [1.0, 0.0, 0.0], 'specular': [0.0, 1.0, 0.0],
                                                   'shininess': 1000.0}
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

    def set_scale(self, scale):
        if self.model_matrix is None:
            raise ValueError("model_matrix is not initialized")

        if not isinstance(scale, (int, float)):
            raise ValueError("scale must be a numeric value")

        try:
            self.model_matrix = glm.scale(self.model_matrix, glm.vec3(scale, scale, scale))
        except Exception as e:
            raise RuntimeError(f"Failed to scale model_matrix: {e}")

    def set_orientation(self, rotation_angles):
        # Apply rotations around x, y, z axes respectively
        rotation_matrix = glm.mat4(1.0)
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[0]), glm.vec3(1.0, 0.0, 0.0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[1]), glm.vec3(0.0, 1.0, 0.0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[2]), glm.vec3(0.0, 0.0, 1.0))
        self.orientation = glm.vec3(rotation_angles)
        self.model_matrix = rotation_matrix * self.model_matrix

    def set_position(self, translation):
        # Ensure the translation is a glm.vec3
        translation_vec = glm.vec3(translation[0], translation[1], translation[2])

        # Create the translation matrix
        translation_matrix = glm.translate(glm.mat4(1.0), translation_vec)

        # Multiply the translation matrix with the current model matrix
        # Note the order of multiplication may need to be reversed depending on the use case
        self.position = translation
        self.model_matrix = translation_matrix * self.model_matrix
        # If you want to set an absolute position, you might need to reset or reinitialize the model matrix
        # self.model_matrix = translation_matrix  # Uncomment this line for absolute positioning

    # Implement other methods as required
    def calculate_bounding_box(self, bounding_margin=0.1) -> list:
        if self.is_player:
            # Return a point at the bottom center of the player model: the player's feet
            return [glm.vec3(0, 0, 0)]  # Assuming the player's feet are at the origin
        else:
            # This handles the world object
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

    # def __getattr__(self, name):
    #     return getattr(self._model, name)
    def update_model_matrix(self, parent_matrix=None):
        translation_matrix = glm.translate(glm.mat4(1.0),
                                           glm.vec3(self.position[0], self.position[1], self.position[2]))

        rotation_x = glm.rotate(glm.mat4(1.0), glm.radians(self.orientation[0]), glm.vec3(1.0, 0.0, 0.0))
        rotation_y = glm.rotate(glm.mat4(1.0), glm.radians(self.orientation[1]), glm.vec3(0.0, 1.0, 0.0))
        rotation_z = glm.rotate(glm.mat4(1.0), glm.radians(self.orientation[2]), glm.vec3(0.0, 0.0, 1.0))

        rotation_matrix = rotation_z * rotation_y * rotation_x

        scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(self.scale, self.scale, self.scale))

        local_model_matrix = translation_matrix * rotation_matrix * scale_matrix

        if parent_matrix is None:
            self.model_matrix = local_model_matrix
        else:
            self.model_matrix = parent_matrix * local_model_matrix

    def init_model_matrix(self, translation, rotation_angles):
        translation_matrix = glm.translate(glm.mat4(1.0), translation)

        rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(rotation_angles[0]), glm.vec3(1.0, 0.0, 0.0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[1]), glm.vec3(0.0, 1.0, 0.0))
        rotation_matrix = glm.rotate(rotation_matrix, glm.radians(rotation_angles[2]), glm.vec3(0.0, 0.0, 1.0))

        scale_matrix = glm.scale(glm.mat4(1.0), glm.vec3(self.scale, self.scale, self.scale))

        self.model_matrix = translation_matrix * rotation_matrix * scale_matrix




