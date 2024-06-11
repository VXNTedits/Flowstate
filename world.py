from model import Model


class World(Model):
    def __init__(self, filepaths: list, mtl_filepaths: list, rotations: list, translations: list, material_overrides):
        super().__init__(filepaths[0],  mtl_filepaths[0], rotation_angles=rotations[0], translation=translations[0],
                         kd_override=material_overrides[0][0], ks_override=material_overrides[0][1],
                         ns_override=material_overrides[0][2])

        self.objects = []
        self.world_aabb = None
        self.is_player = False
        # Initialize each object and store it in the objects list
        for filepath, mtl_filepath, rotation, translation, material_overrides in zip(filepaths, mtl_filepaths, rotations, translations,
                                                                       material_overrides):
            obj = Model(filepath, mtl_filepath, rotation_angles=rotation, translation=translation, kd_override=material_overrides[0],
                        ks_override=material_overrides[1], ns_override=material_overrides[2])
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
