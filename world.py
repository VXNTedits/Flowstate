from model import Model

class World(Model):
    def __init__(self, filepath: str, rotation_angles=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0)):
        self.world_aabb = None
        super().__init__(filepath, rotation_angles=rotation_angles, translation=translation)
        self.height = translation[1]
        print('world bounding box init ',self.bounding_box)
        # Extract the transformed bounding box vertices for the world
        #self.calculate_world_bounding_box()

        # Find the axis-aligned bounding box (AABB) for the world

    def calculate_world_bounding_box(self):
        print(self.vertices)
        min_x = min(self.vertices, key=lambda v: v.x).x
        min_y = min(self.vertices, key=lambda v: v.y).y
        min_z = min(self.vertices, key=lambda v: v.z).z
        max_x = max(self.vertices, key=lambda v: v.x).x
        max_y = max(self.vertices, key=lambda v: v.y).y
        max_z = max(self.vertices, key=lambda v: v.z).z
        self.world_aabb = (min_x, min_y, min_z, max_x, max_y, max_z)
