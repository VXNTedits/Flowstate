import numpy as np

import os
import numpy as np

import os
import numpy as np

import os
import numpy as np

import numpy as np
from MaterialLoader import MaterialLoader

import os
import numpy as np
from MaterialLoader import MaterialLoader

import os
import numpy as np
from MaterialLoader import MaterialLoader

import os
import numpy as np
from MaterialLoader import MaterialLoader

class ObjLoader:
    def __init__(self):
        self.material_loader = MaterialLoader()
        self.materials = {}

    def load_obj(self, filepath: str):
        vertices = []
        normals = []
        faces = []
        material_file = None
        current_material = None
        base_dir = os.path.dirname(filepath)

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
                    faces.append((face, current_material))
                elif line.startswith('mtllib '):
                    material_file = os.path.join(base_dir, line.split()[1])
                elif line.startswith('usemtl '):
                    current_material = line.split()[1]

        if material_file:
            self.materials = self.material_loader.load_mtl(material_file)

        if len(normals) == 0:
            normals = self.calculate_normals(vertices, faces)

        vertex_data = []
        for face, material in faces:
            for vertex_index, normal_index in face:
                vertex_data.extend(vertices[vertex_index])
                vertex_data.extend(normals[normal_index])
                if material and 'Kd' in self.materials[material]:
                    vertex_data.extend(self.materials[material]['Kd'])
                else:
                    vertex_data.extend([1.0, 1.0, 1.0])  # Default to white if no material

        vertex_data = np.array(vertex_data, dtype=np.float32)
        indices = np.arange(len(vertex_data) // 9, dtype=np.uint32)  # 3 positions + 3 normals + 3 colors
        return vertex_data, indices, self.materials

    def calculate_normals(self, vertices, faces):
        normals = np.zeros((len(vertices), 3), dtype=np.float32)
        for face, _ in faces:
            v0 = np.array(vertices[face[0][0]])
            v1 = np.array(vertices[face[1][0]])
            v2 = np.array(vertices[face[2][0]])
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / np.linalg.norm(normal)
            for vertex_index, _ in face:
                normals[vertex_index] += normal
        normals = [normal / np.linalg.norm(normal) for normal in normals]
        return normals
