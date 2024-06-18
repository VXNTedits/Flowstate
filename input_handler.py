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
                self.camera.toggle_view(self.player.position, self.player.get_rotation_matrix())
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
                self.player.handle_left_click()
            elif action == glfw.RELEASE:
                self.left_mouse_button_pressed = False

    def process_input(self, player, delta_time):
        if self.keys.get(glfw.KEY_W):
            self.handle_input('FORWARD', delta_time)

        if self.keys.get(glfw.KEY_S):
            #print('s')
            self.handle_input('BACKWARD', delta_time)

        if self.keys.get(glfw.KEY_A):
            #print('a')
            self.handle_input('LEFT', delta_time)

        if self.keys.get(glfw.KEY_D):
            #print('d')
            self.handle_input('RIGHT', delta_time)

        if self.keys.get(glfw.KEY_SPACE):
            #print('input: space')
            self.handle_input('JUMP', delta_time)

        if self.keys.get(glfw.KEY_F):
            #print('input: interact')
            self.handle_input('INTERACT', delta_time)

    def handle_input(self, direction, delta_time):
        #print(direction)
        self.player.propose_updated_thrust(direction, delta_time)
