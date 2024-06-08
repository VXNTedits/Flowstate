import os


def parse_obj_file(file_path):
    vertices = []
    normals = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('vn '):
                normals.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                face = line.strip().split()[1:]
                face_indices = [list(map(int, vert.split('/'))) for vert in face]
                faces.append(face_indices)

    return vertices, normals, faces

# Path to the uploaded OBJ file
file_path = r"C:\Users\leona\OneDrive\Projects\Game\obj\world1.obj"

# Parse the OBJ file
vertices, normals, faces = parse_obj_file(file_path)

# Display the first few elements to inspect
print("Vertices:", vertices[:5])
print("Normals:", normals[:5])
print("Faces:", faces[:5])
