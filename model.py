import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm
import os

import glm
from OpenGL.GL import *
import numpy as np

import glm
from OpenGL.GL import *
import numpy as np

import glm
from OpenGL.GL import *
import numpy as np

import numpy as np
import glm
from OpenGL.GL import *
import os

class Model:
    def __init__(self, filepath: str):
        self.vertices, self.indices = self.load_obj(filepath)
        self.vao, self.vbo, self.ebo = self.setup_buffers()

    def parse_obj(self, filepath: str):
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

        return vertices, normals, faces

    def calculate_normals(self, vertices, faces):
        normals = np.zeros((len(vertices), 3), dtype=np.float32)
        for face in faces:
            v0, v1, v2 = face[0][0], face[1][0], face[2][0]
            p0, p1, p2 = np.array(vertices[v0]), np.array(vertices[v1]), np.array(vertices[v2])
            normal = np.cross(p1 - p0, p2 - p0)
            normal = normal / np.linalg.norm(normal)
            for vertex_index, _ in face:
                normals[vertex_index] += normal
        normals = np.array([n / np.linalg.norm(n) for n in normals])
        return normals

    def load_obj(self, filepath: str):
        vertices, file_normals, faces = self.parse_obj(filepath)

        if len(file_normals) == 0:
            normals = self.calculate_normals(vertices, faces)
        else:
            normals = file_normals

        vertex_data = []
        for face in faces:
            for vertex_index, normal_index in face:
                vertex_data.extend(vertices[vertex_index])
                vertex_data.extend(normals[vertex_index])

        vertex_data = np.array(vertex_data, dtype=np.float32)
        indices = np.arange(len(vertex_data) // 6, dtype=np.uint32)
        return vertex_data, indices

    def setup_buffers(self):
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
