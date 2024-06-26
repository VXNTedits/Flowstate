import sys

import glm

class Camera:
    def __init__(self, position, up, yaw=-90.0, pitch=0.0):
        self.aspect_ratio = 800/600
        self.zoom = 1
        self.position = position
        self.up = up
        self.yaw = yaw
        self.pitch = pitch
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.right = glm.vec3()
        self.world_up = up
        self.update_camera_vectors()
        self.first_person = True  # Default to first-person view

    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, self.up)

    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        sensitivity = 0.1
        xoffset *= sensitivity
        yoffset *= sensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        if constrain_pitch:
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0

        self.update_camera_vectors()

    def get_projection_matrix(self):
        # Return the projection matrix
        return glm.perspective(glm.radians(self.zoom), self.aspect_ratio, 0.01, 1000.0)

    def update_camera_vectors(self):
        front = glm.vec3()
        front.x = glm.cos(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        front.y = glm.sin(glm.radians(self.pitch))
        front.z = glm.sin(glm.radians(self.yaw)) * glm.cos(glm.radians(self.pitch))
        self.front = glm.normalize(front)
        self.right = glm.normalize(glm.cross(self.front, self.world_up))
        self.up = glm.normalize(glm.cross(self.right, self.front))


    def toggle_view(self, player_position):
        if self.first_person:
            self.set_third_person(player_position)
        else:
            self.set_first_person(player_position)
        self.first_person = not self.first_person

    def set_first_person(self, head_position):
        offset = glm.vec3(0.0, 1.8, 0.0)  # Adjust offset as needed for correct eye level
        self.position = head_position # + offset

    def set_third_person(self, head_position):
        offset = glm.vec3(4, 2, 4)  # Adjust offset for third-person view
        self.position = head_position + offset

    def set_camera_position(self, position):
        self.position = position
        self.update_camera_vectors()

    def update(self, delta_time):
        pass
