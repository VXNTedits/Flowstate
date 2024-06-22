import glfw

class Window:
    def __init__(self, width, height, title, fullscreen=False):
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        self.width = width
        self.height = height

        primary_monitor = glfw.get_primary_monitor()
        if not primary_monitor:
            raise Exception("Failed to get primary monitor")

        if fullscreen:
            video_mode = glfw.get_video_mode(primary_monitor)
            if not video_mode:
                raise Exception("Failed to get video mode for primary monitor")
            self.window = glfw.create_window(video_mode.size.width, video_mode.size.height, title, primary_monitor, None)
            self.width = video_mode.size.width
            self.height = video_mode.size.height
        else:
            self.window = glfw.create_window(width, height, title, None, None)

        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        glfw.make_context_current(self.window)

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
