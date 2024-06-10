import glm

from model import Model

class Player(Model):
    def __init__(self, model_path: str, camera):
        self.camera = camera  # Ensure camera is assigned first
        self.model = Model(model_path, player=True)  # Change _model to model
        self._position = glm.vec3(10.0, 10.2, -10.0)  # Initialize at the specified position
        self.previous_position = glm.vec3(10.0, 10.0, -10.0)
        self.front = glm.vec3(0.0, 0.0, -1.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.speed = 2.5
        self.thrust = glm.vec3(0.0, 0.0, 0.0)
        self._velocity = glm.vec3(0, 0, 0)
        self.yaw = camera.yaw  # Initialize yaw to camera's yaw
        self._rotation = glm.vec3(camera.pitch,camera.yaw,0)
        self.vertices, self.indices = Model.load_obj(self, model_path)
        self.convex_components_list = self.model.convex_components_list
        self.convex_components_obj = self.model.convex_components_obj
        self.bounding_box = self.convex_components_list
        # Apply a rotation to make the model stand vertically
        self.model_matrix = glm.rotate(glm.mat4(1.0), glm.radians(-90), glm.vec3(1.0, 0.0, 0.0))
        #self.set_origin(glm.vec3(-0.025, 0.2, 0.0))
        self.update_model_matrix()

    def set_origin(self, new_origin):
        """Set the player object's (0,0,0) coordinate to new_origin."""
        self.model.translate(new_origin)  # Use self.model
        self._position = new_origin
        self.update_model_matrix()

    def update(self, delta_time: float):
        # Update position based on thrust and delta_time
        self._position += self.thrust * delta_time

        # Ensure the camera position is updated to stay synchronized with the player
        self.update_camera_position()
        self.update_bounding_box_position()
        self.update_model_matrix()

        #print(f"Player position: {self._position} Camera position: {self.camera.position}")

    def update_position(self, direction: str, delta_time: float):
        self.thrust = glm.vec3(0.0, 0.0, 0.0)
        front = glm.vec3(glm.cos(glm.radians(self.yaw)), 0, glm.sin(glm.radians(self.yaw)))
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

        self.update_camera_position()

        self.update_model_matrix()
        print(f"Player position: {self._position} Camera position: {self.camera.position}")

    def update_model_matrix(self):
        # Combine all rotations into a single transformation matrix
        model_rotation = (
                glm.rotate(glm.mat4(1.0), glm.radians(-self.yaw), glm.vec3(0.0, 1.0, 0.0)) *
                glm.rotate(glm.mat4(1.0), glm.radians(90), glm.vec3(0.0, 1.0, 0.0)) *
                glm.rotate(glm.mat4(1.0), glm.radians(-90), glm.vec3(1.0, 0.0, 0.0))
        )
        translation = glm.translate(glm.mat4(1.0), self._position)
        self.model.model_matrix = translation * model_rotation

    def get_rotation_matrix(self):
        return glm.rotate(glm.mat4(1.0), glm.radians(self.yaw), glm.vec3(0.0, 1.0, 0.0))

    def set_position(self, position):
        self._position = position
        self.update_model_matrix()

    def draw(self):
        self.model.draw()  # Use self.model

    # In the Player class, ensure camera yaw is updated to match player yaw
    def process_mouse_movement(self, xoffset, yoffset):
        self.camera.process_mouse_movement(xoffset, yoffset)
        self.yaw = self.camera.yaw  # Update player's yaw based on camera's yaw
        self.update_camera_position()

    def update_camera_position(self):
        # Always update camera yaw to match player yaw
        self.camera.yaw = self.yaw
        self.camera.update_camera_vectors()
        if self.camera.first_person:
            self.camera.set_first_person(self._position, self.get_rotation_matrix())
        else:
            self.camera.set_third_person(self._position, self.get_rotation_matrix())

    @property
    def velocity(self) -> glm.vec3:
        return self._velocity

    @velocity.setter
    def velocity(self, value: glm.vec3):
        self._velocity = value

    @property
    def position(self) -> glm.vec3:
        return self._position

    @property
    def rotation(self) -> glm.vec3:
        return self._rotation

    @rotation.setter
    def rotation(self, value: glm.vec3):
        self._rotation = value
        self.update_bounding_box_position()  # Update bounding box when rotation changes

    @position.setter
    def position(self, value: glm.vec3):
        self._position = value
        self.update_bounding_box_position()

    def update_bounding_box_position(self):
        # Get the current transformation matrix
        translation_matrix = glm.translate(glm.mat4(1.0), self._position)
        rotation_matrix = glm.rotate(glm.mat4(1.0), self.rotation.y,
                                     glm.vec3(0.0, 1.0, 0.0))  # Assuming rotation around Y-axis
        # Initialize the transformed bounding box
        transformed_bounding_box = []
        # Apply transformations to each vertex of the bounding box
        for v_obj in self.bounding_box:
            transformed_v_obj = []  # Initialize a new transformed vertex object
            for v_i in v_obj:
                transformed_vertex = translation_matrix * rotation_matrix * v_i
                transformed_v_obj.append(transformed_vertex)
            transformed_bounding_box.append(transformed_v_obj)

        # Update the bounding box with the newly transformed vertices
        self.bounding_box = transformed_bounding_box
