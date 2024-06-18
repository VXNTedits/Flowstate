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
        # Load vertex/fragment shaders from file
        vertex_code = self.load_shader_code(vertex_path)
        fragment_code = self.load_shader_code(fragment_path)
        # Compile shaders
        vertex_shader = self.compile_shader(vertex_code, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(fragment_code, GL_FRAGMENT_SHADER)
        # Link shaders to create a program
        self.program = self.link_program(vertex_shader, fragment_shader)

    def load_shader_code(self, path):
        with open(path, 'r') as file:
            return file.read()

    def link_program(self, vertex_shader, fragment_shader):
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(program))
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        return program

    def use(self):
        glUseProgram(self.program)

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

    def set_uniform_matrix4fv(self, name, matrix):
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}.")
        else:
            glUniformMatrix4fv(location, 1, GL_FALSE, glm.value_ptr(matrix))

    def set_uniform3f(self, name, vec3):
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}.")
        else:
            glUniform3f(location, vec3.x, vec3.y, vec3.z)

    def set_uniform1i(self, name, value):
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}.")
        else:
            glUniform1i(location, value)

    def set_bump_scale(self, value: float):
        self.set_uniform1f("bumpScale", value)

    def set_roughness(self, value: float):
        self.set_uniform1f("roughness", value)

    def set_uniform1f(self, name, value):
        location = glGetUniformLocation(self.program, name)
