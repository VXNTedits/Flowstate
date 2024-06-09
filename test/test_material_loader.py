from MaterialLoader import MaterialLoader

def test_material_loader():
    material_loader = MaterialLoader()
    materials = material_loader.load_mtl('cube.mtl')
    for material_name, properties in materials.items():
        print(f"Material: {material_name}")
        for prop, value in properties.items():
            print(f"  {prop}: {value}")

test_material_loader()
