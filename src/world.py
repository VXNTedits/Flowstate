import os
import glm
from src.composite_model import CompositeModel
from model import Model
from utils.file_utils import *


class World(CompositeModel):
    def __init__(self, name, air_density, *args, **kwargs):
        # Construct the directory path
        self.air_density = air_density
        directory = os.path.join(get_project_root(), 'res', name)

        # Initialize lists to hold file paths and model parameters
        filepaths = []
        mtl_filepaths = []  # Assuming .mtl files are in the same directory
        rotations = []
        translations = []
        material_overrides = []
        scales = []

        # Parse the directory to find .obj files
        for filename in os.listdir(directory):
            if filename.endswith('.obj'):
                filepath = os.path.join(directory, filename)
                mtl_filepath = filepath.replace('.obj', '.mtl')
                # Add model initialization parameters
                filepaths.append(filepath)
                mtl_filepaths.append(mtl_filepath)
                rotations.append(glm.vec3(0, 0, 0))  # Default rotation
                translations.append(glm.vec3(0, -2, 0))  # 50, 0, -30 Default translation
                material_overrides.append(None)  # Default material
                scales.append(1)  # Default scale

        # Ensure there is at least one file to initialize
        if filepaths:
            # Initialize the superclass with the first model
            super().__init__(filepath=filepaths[0],
                             mtl_filepath=mtl_filepaths[0],
                             rotation_angles=rotations[0],
                             translation=translations[0],
                             kd_override=material_overrides[0],
                             ks_override=material_overrides[0],
                             ns_override=material_overrides[0],
                             scale=scales[0],
                             shift_to_centroid=False,
                             is_collidable=True,
                             *args,
                             **kwargs)

            # Add remaining models using add_model method
            for i in range(1, len(filepaths)):
                model = Model(filepath=filepaths[i],
                              mtl_filepath=mtl_filepaths[i],
                              rotation_angles=rotations[0],
                              translation=translations[0],
                              kd_override=material_overrides[0],
                              ks_override=material_overrides[0],
                              ns_override=material_overrides[0],
                              scale=scales[0],
                              is_collidable=True)
                self.add_world_model(model, scale=scales[0],
                                     relative_position=translations[0],
                                     relative_rotation=rotations[0])

    def get_world_objects(self):
        world_objects = [model for model, _, _ in self.models]
        return world_objects

    def update(self, delta_time):
        pass

    def handle_shoot_event(self, impact_point):
        # Find the object at the impact point or create one
        impacted_obj = self.find_or_create_impact_object(impact_point)

        # Set impact-related attributes
        impacted_obj.impact = True
        impacted_obj.impact_point = impact_point
        impacted_obj.crater_radius = 2.0  # Example value, set appropriately
        impacted_obj.crater_depth = 1.0  # Example value, set appropriately

        # Trigger a render update (or flag for the next render call)
        self.render_update_needed = True

    def find_or_create_impact_object(self, impact_point):
        # Find the object at the impact point or create a new one if needed
        for obj in self.get_world_objects():
            if self.contains_point(obj, impact_point):
                return obj

        # If no object found, create a new one (or handle as needed)
        new_obj = self.create_new_object_at(impact_point)
        self.add_world_model(new_obj, scale=1, relative_position=impact_point, relative_rotation=glm.vec3(0, 0, 0))
        return new_obj

    def contains_point(self, obj, point):
        # Placeholder method to determine if the object contains the point
        # Implement your logic to check if the object contains the point
        return False

    def create_new_object_at(self, impact_point):
        # Placeholder method to create a new object at the impact point
        # Implement your logic to create a new object
        model = Model(filepath="path/to/model.obj",
                      mtl_filepath="path/to/material.mtl",
                      rotation_angles=glm.vec3(0, 0, 0),
                      translation=impact_point,
                      kd_override=None,
                      ks_override=None,
                      ns_override=None,
                      scale=1,
                      is_collidable=True)
        return model
