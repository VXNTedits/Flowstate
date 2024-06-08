import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import glm
import os

# Global variables for camera position and orientation
camera_pos = glm.vec3(0.0, 0.0, 3.0)
camera_front = glm.vec3(0.0, 0.0, -1.0)
camera_up = glm.vec3(0.0, 1.0, 0.0)
yaw = -90.0
pitch = 0.0
last_x = 400
last_y = 300
first_mouse = True
camera_speed = 2.5
sensitivity = 0.1
delta_time = 0.0
last_frame = 0.0

# Track key states
keys = {
    glfw.KEY_W: False,
    glfw.KEY_A: False,
    glfw.KEY_S: False,
    glfw.KEY_D: False,
}


# Read shader source code from file
def read_shader_source(filename):
    with open(filename, 'r') as file:
        return file.read()


# Compile a shader and return the shader ID
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    # Check for compilation errors
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        error_message = glGetShaderInfoLog(shader)
        raise RuntimeError(f"Shader compilation failed: {error_message}")

    return shader


# Handle key presses and releases
def key_callback(window, key, scancode, action, mods):
    global keys
    if action == glfw.PRESS:
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        if key in keys:
            keys[key] = True
    elif action == glfw.RELEASE:
        if key in keys:
            keys[key] = False


# Handle mouse movement
def mouse_callback(window, xpos, ypos):
    global yaw, pitch, last_x, last_y, first_mouse, camera_front

    if first_mouse:
        last_x = xpos
        last_y = ypos
        first_mouse = False

    xoffset = xpos - last_x
    yoffset = last_y - ypos  # Reversed since y-coordinates range from bottom to top
    last_x = xpos
    last_y = ypos

    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset
    pitch += yoffset

    # Constrain the pitch angle
    if pitch > 89.0:
        pitch = 89.0
    if pitch < -89.0:
        pitch = -89.0

    # Calculate the new front vector
    front = glm.vec3()
    front.x = np.cos(glm.radians(yaw)) * np.cos(glm.radians(pitch))
    front.y = np.sin(glm.radians(pitch))
    front.z = np.sin(glm.radians(yaw)) * np.cos(glm.radians(pitch))
    camera_front = glm.normalize(front)


# Update the camera position based on the key states
def update_camera_position():
    global camera_pos, delta_time
    camera_speed_adjusted = camera_speed * delta_time

    if keys[glfw.KEY_W]:
        camera_pos += camera_speed_adjusted * camera_front
    if keys[glfw.KEY_S]:
        camera_pos -= camera_speed_adjusted * camera_front
    if keys[glfw.KEY_A]:
        camera_pos -= glm.normalize(glm.cross(camera_front, camera_up)) * camera_speed_adjusted
    if keys[glfw.KEY_D]:
        camera_pos += glm.normalize(glm.cross(camera_front, camera_up)) * camera_speed_adjusted


def parse_obj(filepath):
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


def calculate_normals(vertices, faces):
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


def load_obj(filepath):
    vertices, file_normals, faces = parse_obj(filepath)

    if len(file_normals) == 0:
        normals = calculate_normals(vertices, faces)
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


def main():
    global camera_pos, camera_front, camera_up, delta_time, last_frame

    # Initialize GLFW
    if not glfw.init():
        return

    # Set GLFW window hints (for OpenGL 3.3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)  # For macOS compatibility

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(800, 600, "Movable Camera with Mouse Look and Lighting", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # Set the key callback
    glfw.set_key_callback(window, key_callback)

    # Set the mouse callback
    glfw.set_cursor_pos_callback(window, mouse_callback)

    # Capture the mouse
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    # Load and compile vertex shader
    vertex_shader_source = read_shader_source('vertex_shader.glsl')
    vertex_shader = compile_shader(vertex_shader_source, GL_VERTEX_SHADER)

    # Load and compile fragment shader
    fragment_shader_source = read_shader_source('fragment_shader.glsl')
    fragment_shader = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER)

    # Link shaders into a shader program
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)

    # Check for linking errors
    if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
        error_message = glGetProgramInfoLog(shader_program)
        raise RuntimeError(f"Shader linking failed: {error_message}")

    # Clean up shaders (they are now linked into the program and no longer needed)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    # Load the .obj file from the 'obj' directory
    obj_filepath = os.path.join('obj', 'cube.obj')
    vertex_data, indices = load_obj(obj_filepath)

    # Generate and bind a Vertex Array Object (VAO)
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    # Generate and bind a Vertex Buffer Object (VBO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

    # Generate and bind an Element Buffer Object (EBO)
    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Define the vertex attribute pointers
    stride = 6 * vertex_data.itemsize
    # Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    # Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * vertex_data.itemsize))
    glEnableVertexAttribArray(1)

    # Unbind the VAO (not really necessary)
    glBindVertexArray(0)

    # Enable depth testing
    glEnable(GL_DEPTH_TEST)

    # Use the shader program
    glUseProgram(shader_program)

    # Get the uniform locations
    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    proj_loc = glGetUniformLocation(shader_program, "projection")
    light_pos_loc = glGetUniformLocation(shader_program, "lightPos")
    view_pos_loc = glGetUniformLocation(shader_program, "viewPos")
    light_color_loc = glGetUniformLocation(shader_program, "lightColor")
    object_color_loc = glGetUniformLocation(shader_program, "objectColor")

    # Set light properties
    light_pos = glm.vec3(1.2, 1.0, 2.0)
    light_color = glm.vec3(1.0, 1.0, 1.0)
    object_color = glm.vec3(1.0, 0.5, 0.31)

    # Main loop
    while not glfw.window_should_close(window):
        # Calculate delta time
        current_frame = glfw.get_time()
        delta_time = current_frame - last_frame
        last_frame = current_frame

        # Process input
        update_camera_position()

        # Clear the screen and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Create transformations
        model = glm.mat4(1.0)
        model = glm.rotate(model, glm.radians(glfw.get_time() * 50), glm.vec3(0.0, 1.0, 0.0))
        view = glm.lookAt(camera_pos, camera_pos + camera_front, camera_up)
        projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

        # Pass the matrices to the shader
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(projection))

        # Pass light properties to the shader
        glUniform3fv(light_pos_loc, 1, glm.value_ptr(light_pos))
        glUniform3fv(view_pos_loc, 1, glm.value_ptr(camera_pos))
        glUniform3fv(light_color_loc, 1, glm.value_ptr(light_color))
        glUniform3fv(object_color_loc, 1, glm.value_ptr(object_color))

        # Bind the VAO
        glBindVertexArray(VAO)

        # Draw the object
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        # Unbind the VAO
        glBindVertexArray(0)

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    # Clean up
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteBuffers(1, [EBO])
    glfw.terminate()


if __name__ == "__main__":
    main()
