import glfw

import glfw


class Window:
    def __init__(self, width, height, title, fullscreen=False):
        # Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        # Request OpenGL 4.3 context
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.width = width
        self.height = height

        # Get the primary monitor
        primary_monitor = glfw.get_primary_monitor()
        if not primary_monitor:
            glfw.terminate()
            raise Exception("Failed to get primary monitor")

        if fullscreen:
            # Get the video mode of the primary monitor
            video_mode = glfw.get_video_mode(primary_monitor)
            if not video_mode:
                glfw.terminate()
                raise Exception("Failed to get video mode for primary monitor")

            # Create a fullscreen window
            self.window = glfw.create_window(video_mode.width,
                                             video_mode.height,
                                             title, primary_monitor, None)
            self.width = video_mode.width
            self.height = video_mode.height
        else:
            # Create a windowed mode window
            self.window = glfw.create_window(width, height, title, None, None)

        # Check if the window was created successfully
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        # Make the OpenGL context current
        glfw.make_context_current(self.window)

    def __del__(self):
        # Clean up the window and terminate GLFW
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()

    def set_callbacks(self, key_callback, mouse_callback, mouse_button_callback):
        glfw.set_key_callback(self.window, key_callback)
        glfw.set_cursor_pos_callback(self.window, mouse_callback)
        glfw.set_mouse_button_callback(self.window, mouse_button_callback)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    def should_close(self):
        return glfw.window_should_close(self.window)

    def swap_buffers(self):
        glfw.swap_buffers(self.window)

    def poll_events(self):
        glfw.poll_events()
