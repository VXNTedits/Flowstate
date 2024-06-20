import os
import glm
from src.composite_model import CompositeModel
from model import Model


class World(CompositeModel):
    def __init__(self, name, *args, **kwargs):
        # Construct the directory path
        directory = os.path.join('./res', name)

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
                translations.append(glm.vec3(50, 0, -30))  # 50, 0, -30 Default translation
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
            #self.world_objects += self

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
                #self.world_objects += model

    def get_world_objects(self):
       #return self.world_objects
        world_objects = [model for model, _, _ in self.models]
        return world_objects

    def update(self, delta_time):
        pass
