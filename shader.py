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


class ShaderManager:
    _instances = {}

    @staticmethod
    def get_shader(vertex_path, fragment_path):
        key = (vertex_path, fragment_path)
        if key not in ShaderManager._instances:
            ShaderManager._instances[key] = Shader(vertex_path, fragment_path)
        return ShaderManager._instances[key]


class Shader:
    current_program = None

    def __init__(self, vertex_path, fragment_path):
        self.vertex_path = vertex_path
        self.fragment_path = fragment_path
        self.program = self.create_shader_program()

    def load_shader_code(self, path):
        with open(path, 'r') as file:
            return file.read()

    def compile_shader(self, source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(shader))
        return shader

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

    def create_shader_program(self):
        vertex_code = self.load_shader_code(self.vertex_path)
        fragment_code = self.load_shader_code(self.fragment_path)

        vertex_shader = self.compile_shader(vertex_code, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(fragment_code, GL_FRAGMENT_SHADER)

        program = self.link_program(vertex_shader, fragment_shader)
        print(f"Shader program created with ID: {program}")
        return program

    def use(self):
        if Shader.current_program != self.program:
            #print(f"Switching to shader program ID: {self.program}")
            glUseProgram(self.program)
            Shader.current_program = self.program

    def set_uniform_matrix4fv(self, name, matrix):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}.")
        else:
            glUniformMatrix4fv(location, 1, GL_FALSE, glm.value_ptr(matrix))

    def set_uniform3f(self, name, vec3):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}.")
        else:
            glUniform3f(location, vec3.x, vec3.y, vec3.z)

    def set_uniform1i(self, name, value):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}.")
        else:
            glUniform1i(location, value)

    def set_uniform1f(self, name, value):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}.")
        else:
            glUniform1f(location, value)

    def set_bump_scale(self, value: float):
        self.set_uniform1f("bumpScale", value)

    def set_roughness(self, value: float):
        self.set_uniform1f("roughness", value)


