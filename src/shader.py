import os

import glm
from OpenGL.GL import *


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
        self.script_dir = os.path.dirname(os.path.dirname(__file__))
        self.name = vertex_path.split('/')[-1]
        self.vertex_path = self.get_relative_path(vertex_path)
        self.fragment_path = self.get_relative_path(fragment_path)
        self.program = self.create_shader_program(vertex_path, fragment_path)


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

    def create_shader_program(self, vertex_path, fragment_path):
        vertex_code = self.load_shader_code(vertex_path)
        fragment_code = self.load_shader_code(fragment_path)


        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_code)
        glCompileShader(vertex_shader)
        self.check_compile_errors(vertex_shader, "VERTEX")

        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_code)
        glCompileShader(fragment_shader)
        self.check_compile_errors(fragment_shader, "FRAGMENT")

        return self.link_program(vertex_shader, fragment_shader)

    def check_compile_errors(self, shader, type):
        if type == "VERTEX" or type == "FRAGMENT":
            success = glGetShaderiv(shader, GL_COMPILE_STATUS)
            if not success:
                info_log = glGetShaderInfoLog(shader)
                raise Exception(f"ERROR::SHADER_COMPILATION_ERROR of type: {type}\n{info_log}")

    def check_link_errors(self, program):
        success = glGetProgramiv(program, GL_LINK_STATUS)
        if not success:
            info_log = glGetProgramInfoLog(program)
            raise Exception(f"ERROR::PROGRAM_LINKING_ERROR of type: PROGRAM\n{info_log}")

    def use(self):
        if Shader.current_program != self.program:
            #print(f"Switching to shader program ID: {self.program}")
            glUseProgram(self.program)
            Shader.current_program = self.program

    def set_uniform_matrix4fv(self, name, matrix):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}: {self.name}.")
        else:
            glUniformMatrix4fv(location, 1, GL_FALSE, glm.value_ptr(matrix))

    def set_uniform3f(self, name, vec3):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}: {self.name}.")
        else:
            glUniform3f(location, vec3.x, vec3.y, vec3.z)

    def set_uniform3fv(self, name, vec3):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}: {self.name}.")
        else:
            glUniform3fv(location, 1, glm.value_ptr(vec3))

    def set_uniform1i(self, name, value):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}: {self.name}.")
        else:
            glUniform1i(location, value)

    def set_uniform1f(self, name, value):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}: {self.name}.")
        else:
            glUniform1f(location, value)

    def set_uniform_bool(self, name, value):
        glUniform1i(glGetUniformLocation(self.program, name), value)

    def set_bump_scale(self, value: float):
        self.set_uniform1f("bumpScale", value)

    def set_roughness(self, value: float):
        self.set_uniform1f("roughness", value)

    def get_relative_path(self, relative_path):
        return os.path.join(self.script_dir, relative_path)
