from model import Model

class World(Model):
    def __init__(self, filepath: str, rotation_angles=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0)):
        super().__init__(filepath, rotation_angles=rotation_angles, translation=translation)


