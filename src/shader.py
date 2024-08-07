import os

import glm
import numpy as np
from OpenGL.GL import *
from utils.file_utils import *

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
        self.name_vertex = vertex_path.split('/')[-1]
        self.name_fragment = fragment_path.split('/')[-1]
        self.vertex_path = get_relative_path(vertex_path)
        self.fragment_path = get_relative_path(fragment_path)
        self.program = self.create_shader_program(self.vertex_path, self.fragment_path)

    def load_shader_code(self, path):
        with open(path, 'r') as file:
            return file.read()

    def compile_shader(self, source, shader_type):
        shader = glCreateShader(shader_type)
        lines = source.split('\n')
        version_present = any(line.lstrip().startswith('#version') for line in lines)

        if version_present:
            # print(f"#version directive found in source:\n{source}\n")
            preprocessed_source = source
        else:
            if self.use_glsl_430 or 'gl_VertexID' in source or 'layout(binding = 0) buffer' in source:
                version_directive = "#version 430 core\n"
            else:
                version_directive = "#version 330 core\n"
            preprocessed_source = version_directive + source

        shader_type_str = "GL_VERTEX_SHADER" if shader_type == GL_VERTEX_SHADER else "GL_FRAGMENT_SHADER"
        # print(f"Compiling {shader_type_str} with source...\n__________________\n{preprocessed_source}\n")

        glShaderSource(shader, preprocessed_source)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            info_log = glGetShaderInfoLog(shader).decode()
            print(info_log)
            raise RuntimeError("Shader compilation failed!")
        return shader

    def link_program(self, vertex_shader, fragment_shader):
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            info_log = glGetProgramInfoLog(program).decode()
            print(info_log)
            raise RuntimeError("Program linking failed")
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        return program

    def create_shader_program(self, vertex_path, fragment_path):
        vertex_code = self.load_shader_code(vertex_path)
        fragment_code = self.load_shader_code(fragment_path)
        vertex_shader = self.compile_shader(vertex_code, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(fragment_code, GL_FRAGMENT_SHADER)
        return self.link_program(vertex_shader, fragment_shader)

    def check_compile_errors(self, shader, type):
        if type == "PROGRAM":
            success = glGetProgramiv(shader, GL_LINK_STATUS)
            if not success:
                info_log = glGetProgramInfoLog(shader)
                raise Exception(f"ERROR::SHADER_COMPILATION_ERROR of type: {type}\n{info_log}")
        else:
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
            print(f"Uniform '{name}' not found in shader program {self.program}. \n"
                  f"    (Associated fragment shader: {self.name_fragment}.)")
        else:
            if isinstance(matrix, glm.mat4):
                glUniformMatrix4fv(location, 1, GL_FALSE, glm.value_ptr(matrix))
            elif isinstance(matrix, np.ndarray) and matrix.shape == (4, 4):
                glUniformMatrix4fv(location, 1, GL_FALSE, matrix)
            else:
                print(f"Error: The provided matrix is not a valid glm.mat4 or numpy.ndarray of shape (4, 4).")

    def set_uniform3f(self, name, vec3):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}. \n"
                  f"    (Associated fragment shader: {self.name_fragment}.)")
        else:
            glUniform3f(location, vec3.x, vec3.y, vec3.z)

    def set_uniform3fv(self, name, vec3):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}. \n"
                  f"    (Associated fragment shader: {self.name_fragment}.)")
        else:
            glUniform3fv(location, 1, glm.value_ptr(vec3))
            # Debug information
            #print(f"Set uniform '{name}' to value {vec3} at location {location}.")

    def set_uniform3fvec(self, name, vector):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}.")
        else:
            flat_values = [val for vec in vector for val in vec]
            glUniform3fv(location, len(vector), flat_values)

    def set_uniform1f(self, name, value):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}. \n"
                  f"    (Associated fragment shader: {self.name_fragment}.)")
        else:
            glUniform1f(location, value)
            # Debug information
            #print(f"Set uniform '{name}' to value {value} at location {location}.")

    def set_uniform1fv(self, name, values):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}.")
        else:
            glUniform1fv(location, len(values), values)

    def set_uniform1i(self, name, value):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}. \n"
                  f"    (Associated fragment shader: {self.name_fragment}.)")
        else:
            glUniform1i(location, value)

    def set_uniform_sampler2D(self, name, texture_unit):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}. \n"
                  f"    (Associated fragment shader: {self.name_fragment}.)")
        else:
            glUniform1i(location, texture_unit)

    def set_uniform_sampler3D(self, name, texture_unit):
        """
        Sets a 3D texture sampler uniform in the shader program.

        Args:
        name (str): The name of the sampler3D uniform in the shader.
        texture_unit (int): The texture unit to which the sampler is bound.
        """
        # Activate the shader program before setting the uniform
        self.use()

        # Get the location of the uniform variable
        location = glGetUniformLocation(self.program, name)

        # Check if the uniform is found
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}. \n"
                  f"    (Associated fragment shader: {self.name_fragment}.)")
        else:
            # Set the sampler to use the specified texture unit
            glUniform1i(location, texture_unit)

    def set_uniform_bool(self, name, value):
        self.use()
        location = glGetUniformLocation(self.program, name)
        if location == -1:
            print(f"Uniform '{name}' not found in shader program {self.program}. \n"
                  f"    (Associated fragment shader: {self.name_fragment}.)")
        else:
            glUniform1i(location, int(value))

    def set_tracers_uniform(self, tracers):

        trajectories = []

        for tracer in tracers:
            life = tracer[0]
            for position in tracer[1:]:
                trajectories.append(
                    (position,
                     glm.vec3(1.0, 1.0, 1.0),
                     1 / life
                     )
                )

        self.trajectory_array = np.array(
            trajectories,
            dtype=[('position', np.float32, 3),
                   ('color', np.float32, 3),
                   ('intensity', np.float32)
                   ]
        )

        #print("Trajectories:", trajectories)
        #print("Trajectory Array:", self.trajectory_array)

        # Set the SSBO for trajectories
        self.set_ssbo(0, self.trajectory_array)

    def set_bump_scale(self, value: float):
        self.set_uniform1f("bumpScale", value)

    def set_roughness(self, value: float):
        self.set_uniform1f("roughness", value)
