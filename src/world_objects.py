import os

from src.model import Model
from dataclasses import dataclass
import glm
from utils.file_utils import get_relative_path

@dataclass
class MaterialOverride:
    kd_override: glm.vec3
    ks_override: glm.vec3
    ns_override: float

class WorldObjects:
    def __init__(self, filepaths: list, mtl_filepaths: list, rotations: list, translations: list, material_overrides: list, scales: list):
        self.models = []

        for i in range(len(filepaths)):
            material_override = material_overrides[i]
            model = Model(get_relative_path(filepaths[i]), get_relative_path(mtl_filepaths[i]),
                          rotation_angles=rotations[i],
                          translation=translations[i],
                          kd_override=material_override.kd_override,
                          ks_override=material_override.ks_override,
                          ns_override=material_override.ns_override,
                          scale=scales[i]
                          )
            self.models.append(model)

        self.objects = []
        self.world_aabb = None
        self.is_player = False

        # Initialize each object and store it in the objects list
        for filepath, mtl_filepath, rotation, translation, material_override, scale in zip(filepaths, mtl_filepaths, rotations, translations, material_overrides, scales):
            obj = Model(get_relative_path(filepath), get_relative_path(mtl_filepath),
                        rotation_angles=rotation,
                        translation=translation,
                        kd_override=material_override.kd_override,
                        ks_override=material_override.ks_override,
                        ns_override=material_override.ns_override,
                        scale=scale
                        )
            self.objects.append(obj)

        #self.model_matrix =

        # Calculate world bounding box considering all objects
        #self.calculate_world_bounding_box()

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
        #print('World AABB:', self.world_aabb)

    def get_objects(self):
        return self.objects

    def update(self, delta_time):
        pass
