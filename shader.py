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
        self.fragment_path = vertex_path
        self.vertex_path = fragment_path
        self.name = vertex_path.split('/')[-1]
        # Load vertex/fragment shaders from file
        vertex_code = self.load_shader_code(vertex_path)
        fragment_code = self.load_shader_code(fragment_path)
        # Compile shaders
        vertex_shader = self.compile_shader(vertex_code, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(fragment_code, GL_FRAGMENT_SHADER)
        # Link shaders to create a program
        self.program = None
        self.program = self.link_program(vertex_shader, fragment_shader)
        print(f"Shader program {vertex_path} {fragment_path} created with ID: {self.program}")

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
        current_program = glGetIntegerv(GL_CURRENT_PROGRAM)
        if current_program != self.program:
            print(f"Switching to shader program ID: {self.program}")
            glUseProgram(self.program)

    # def ensure_use(self):
    #     current_program = glGetIntegerv(GL_CURRENT_PROGRAM)
    #     if current_program != self.program:
    #         print(f"Warning: Shader program {self.name} {self.program} is not currently in use. Current program is {self.name} {current_program}.")
    #         self.use()

    def create_shader_program(self):
        vertex_code = self.load_shader_code(self.vertex_path)
        fragment_code = self.load_shader_code(self.fragment_path)

        vertex_shader = self.compile_shader(vertex_code, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(fragment_code, GL_FRAGMENT_SHADER)

        self.program = self.link_program(vertex_shader, fragment_shader)
        print(f"Shader program created with ID: {self.program}")

    def read_shader_source(self, path: str) -> str:
        with open(path, 'r') as file:
            return file.read()

    def compile_shader(self, source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(shader))
        return shader

    def set_uniform_matrix4fv(self, name, matrix):
        # Ensure the shader program is in use
        self.use()
        current_program = glGetIntegerv(GL_CURRENT_PROGRAM)
        if current_program != self.program:
            print(f"Warning: Shader program {self.program} is not currently in use. Current program is {current_program}.")
            self.use()

        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}.")
        else:
            #print(f"Setting uniform '{name}' at location {location} in shader program {self.program}.")
            glUniformMatrix4fv(location, 1, GL_FALSE, glm.value_ptr(matrix))

    def set_uniform3f(self, name, vec3):
        # Ensure the shader program is in use
        self.use()
        current_program = glGetIntegerv(GL_CURRENT_PROGRAM)
        if current_program != self.program:
            #print(f"Warning: Shader program {self.program} is not currently in use. Current program is {current_program}.")
            self.use()

        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}.")
        else:
            #print(f"Setting uniform '{name}' at location {location} in shader program {self.program}.")
            glUniform3f(location, vec3.x, vec3.y, vec3.z)

    def set_uniform1i(self, name, value):
        # Ensure the shader program is in use
        self.use()
        current_program = glGetIntegerv(GL_CURRENT_PROGRAM)
        if current_program != self.program:
            print(f"Warning: Shader program {self.program} is not currently in use. Current program is {current_program}.")
            self.use()

        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}.")
        else:
            #print(f"Setting uniform '{name}' at location {location} in shader program {self.program}.")
            glUniform1i(location, value)

    def set_uniform1f(self, name, value):
        # Ensure the shader program is in use
        self.use()
        current_program = glGetIntegerv(GL_CURRENT_PROGRAM)
        if current_program != self.program:
            print(f"Warning: Shader program {self.program} is not currently in use. Current program is {current_program}.")
            self.use()

        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}.")
        else:
            #print(f"Setting uniform '{name}' at location {location} in shader program {self.program}.")
            glUniform1f(location, value)

    def set_bump_scale(self, value: float):
        self.set_uniform1f("bumpScale", value)

    def set_roughness(self, value: float):
        self.set_uniform1f("roughness", value)


