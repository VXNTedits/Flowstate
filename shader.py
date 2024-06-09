import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm
import os

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

import glm
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

class Shader:
    def __init__(self, vertex_path: str, fragment_path: str):
        self.id = self.create_shader_program(vertex_path, fragment_path)

    def create_shader_program(self, vertex_path: str, fragment_path: str) -> int:
        vertex_code = self.read_shader_source(vertex_path)
        fragment_code = self.read_shader_source(fragment_path)
        vertex_shader = self.compile_shader(vertex_code, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(fragment_code, GL_FRAGMENT_SHADER)
        shader_program = glCreateProgram()
        glAttachShader(shader_program, vertex_shader)
        glAttachShader(shader_program, fragment_shader)
        glLinkProgram(shader_program)
        if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(shader_program))
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        return shader_program

    def use(self):
        glUseProgram(self.id)

    def set_uniform_matrix4fv(self, name: str, matrix):
        if isinstance(matrix, glm.mat4):
            location = glGetUniformLocation(self.id, name)
            glUniformMatrix4fv(location, 1, GL_FALSE, glm.value_ptr(matrix))
        else:
            raise TypeError("Expected glm.mat4 type for the matrix parameter")

    def set_uniform3f(self, name: str, vector: glm.vec3):
        location = glGetUniformLocation(self.id, name)
        glUniform3fv(location, 1, glm.value_ptr(vector))

    def read_shader_source(self, path: str) -> str:
        with open(path, 'r') as file:
            return file.read()

    def compile_shader(self, source: str, shader_type: int) -> int:
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(shader))
        return shader

