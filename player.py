import glm
from model import Model

import glm
from model import Model

import glm
from model import Model

import glm
from model import Model

import glm
from model import Model


class Player(Model):
    def __init__(self, model_path: str, camera):
        self._model = Model(model_path)
        self._position = glm.vec3(10.0, 10.0, -10.0)  # Initialize at the specified position
        self.previous_position = glm.vec3(10.0, 10.0, -10.0)
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.speed = 2.5
        self.camera = camera
        self.thrust = glm.vec3(0.0, 0.0, 0.0)
        self._velocity = glm.vec3(0, 0, 0)
        self.vertices, self.indices = Model.load_obj(self, model_path)

        # Apply a rotation to make the model stand vertically
        self.model_matrix = glm.rotate(glm.mat4(1.0), glm.radians(-90), glm.vec3(1.0, 0.0, 0.0))
        self.update_model_matrix()

    def update(self, delta_time: float):
        # Update position based on thrust and delta_time
        self._position += self.thrust * delta_time

        # Ensure the camera position is updated to stay synchronized with the player
        if self.camera.first_person:
            self.camera.set_first_person(self._position)
        else:
            self.camera.set_third_person(self._position)

        self.update_model_matrix()
        print(f"Player position: {self._position} Camera position: {self.camera.position}")

    def update_position(self, direction: str, delta_time: float):
        self.thrust = glm.vec3(0.0, 0.0, 0.0)
        front = glm.vec3(glm.cos(glm.radians(self.camera.yaw)), 0, glm.sin(glm.radians(self.camera.yaw)))
        right = glm.normalize(glm.cross(front, self.up))

        if direction == 'FORWARD':
            self.thrust += front * self.speed * delta_time
        if direction == 'BACKWARD':
            self.thrust -= front * self.speed * delta_time
        if direction == 'LEFT':
            self.thrust -= right * self.speed * delta_time
        if direction == 'RIGHT':
            self.thrust += right * self.speed * delta_time

        self.previous_position = self._position
        self._position += self.thrust

        if self.camera.first_person:
            self.camera.set_first_person(self._position)  # Update camera to first-person view
        else:
            self.camera.set_third_person(self._position)  # Update camera to third-person view

        self.update_model_matrix()
        print(f"Player position: {self._position} Camera position: {self.camera.position}")

    def update_model_matrix(self):
        translation = glm.translate(glm.mat4(1.0), self._position)
        rotation = glm.rotate(glm.mat4(1.0), glm.radians(self.camera.yaw), glm.vec3(0.0, -1.0, 0.0))
        self._model.model_matrix = translation * rotation * self.model_matrix

    def set_position(self, position):
        self._position = position
        self.update_model_matrix()

    def draw(self):
        self._model.draw()

    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, value: Model):
        self._model = value

    @property
    def velocity(self) -> glm.vec3:
        return self._velocity

    @velocity.setter
    def velocity(self, value: glm.vec3):
        self._velocity = value

    @property
    def position(self) -> glm.vec3:
        return self._position

    @position.setter
    def position(self, value: glm.vec3):
        self._position = value
