import glfw

import glfw

class InputHandler:
    def __init__(self, camera, player):
        self.camera = camera
        self.player = player
        self.keys = {
            glfw.KEY_W: False,
            glfw.KEY_A: False,
            glfw.KEY_S: False,
            glfw.KEY_D: False,
        }

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
        self.camera.process_mouse_movement(xoffset, yoffset)

    def update(self, delta_time):
        if self.keys[glfw.KEY_W]:
            self.player.update_position('FORWARD', delta_time)
        if self.keys[glfw.KEY_S]:
            self.player.update_position('BACKWARD', delta_time)
        if self.keys[glfw.KEY_A]:
            self.player.update_position('LEFT', delta_time)
        if self.keys[glfw.KEY_D]:
            self.player.update_position('RIGHT', delta_time)
