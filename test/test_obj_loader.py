from ObjLoader import ObjLoader

def test_obj_loader():
    obj_loader = ObjLoader()
    vertex_data, indices = obj_loader.load_obj('cube.obj')
    print(vertex_data)
    print(indices)

test_obj_loader()
