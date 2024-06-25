import glfw


class InputHandler:
    def __init__(self, camera, player, physics):
        self.camera = camera
        self.physics = physics
        self.player = player
        self.keys = {
            glfw.KEY_W: False,
            glfw.KEY_A: False,
            glfw.KEY_S: False,
            glfw.KEY_D: False,
            glfw.KEY_SPACE: False,
            glfw.KEY_F: False
        }
        self.left_mouse_button_pressed = False

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_V:
                self.camera.toggle_view(self.player.position)
            if key in self.keys:
                self.keys[key] = True
        elif action == glfw.RELEASE:
            if key in self.keys:
                self.keys[key] = False

    def mouse_callback(self, window, xpos, ypos):
        if not hasattr(self, 'last_x') or not hasattr(self, 'last_y'):
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = True
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False
        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos  # Reversed since y-coordinates range from bottom to top
        self.last_x = xpos
        self.last_y = ypos
        self.player.process_mouse_movement(xoffset, yoffset)

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.left_mouse_button_pressed = True
                self.player.handle_left_click(self.left_mouse_button_pressed)
            elif action == glfw.RELEASE:
                self.left_mouse_button_pressed = False
                self.player.handle_left_click(self.left_mouse_button_pressed)

    def is_left_mouse_button_pressed(self):
        return self.left_mouse_button_pressed

    def process_input(self, player, delta_time):
        directions = []

        if self.keys.get(glfw.KEY_W):
            directions.append('FORWARD')

        if self.keys.get(glfw.KEY_S):
            directions.append('BACKWARD')

        if self.keys.get(glfw.KEY_A):
            directions.append('LEFT')

        if self.keys.get(glfw.KEY_D):
            directions.append('RIGHT')

        if self.keys.get(glfw.KEY_SPACE):
            directions.append('JUMP')

        if self.keys.get(glfw.KEY_F):
            directions.append('INTERACT')

        if directions:
            self.handle_input(directions, delta_time)

    def handle_input(self, directions, delta_time):
        # print(directions)
        self.player.propose_updated_thrust(directions, delta_time)
