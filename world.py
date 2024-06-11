from model import Model


class World(Model):
    def __init__(self, filepaths: list, rotations: list, translations: list):
        super().__init__(filepaths[0], rotation_angles=rotations[0], translation=translations[0])
        self.objects = []
        self.world_aabb = None
        self.is_player = False
               # Initialize each object and store it in the objects list
        for filepath, rotation, translation in zip(filepaths, rotations, translations):
            obj = Model(filepath, rotation_angles=rotation, translation=translation)
            self.objects.append(obj)

        # Calculate world bounding box considering all objects
        self.calculate_world_bounding_box()

    def calculate_world_bounding_box(self):
        min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
        max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

        for obj in self.objects:
            bbox = obj.bounding_box
            min_x = min(min_x, bbox[0].x)
            min_y = min(min_y, bbox[0].y)
            min_z = min(min_z, bbox[0].z)
            max_x = max(max_x, bbox[1].x)
            max_y = max(max_y, bbox[1].y)
            max_z = max(max_z, bbox[1].z)

        self.world_aabb = (min_x, min_y, min_z, max_x, max_y, max_z)
        print('World AABB:', self.world_aabb)

    def get_objects(self):
        return self.objects