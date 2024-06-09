from interfaces import PlayerInterface, WorldInterface
import glm
from model import Model

class Player(PlayerInterface):
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
        self.camera.position = self._position  # Ensure camera follows player
        self.update_model_matrix()
        print(f"Player position: {self._position} Camera position: {self.camera.position}")

    def update_model_matrix(self):
        self._model.model_matrix = glm.translate(glm.mat4(1.0), self._position)

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
